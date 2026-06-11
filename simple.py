import os
import warnings
import logging

# Suppress Python warnings (torch.distributed checkpoint UserWarnings, etc.)
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# Silence HuggingFace Hub messages (e.g. unauthenticated request warning)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import math
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info

from modules.muon import Muon

from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

from modules.layers import RMSNorm, RotaryEmbedding, Block
from modules.utils import (
    load_checkpoint,
    save_checkpoint_async,
    train_or_load_bpe,
)

# EPYC 9355: 48 cores / 96 threads. Cap PyTorch + MKL thread pools to avoid
# contention with DataLoader workers (we use 12 workers below).
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

 
# Blackwell: expandable segments reduce fragmentation over long runs
# with large KV caches and variable-length allocations.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

def build_document_block_mask(idx, eot_id):
    """Block-diagonal causal BlockMask for FlexAttention.

    Documents are packed end-to-end and separated by `eot_id` (the eot token is
    the last token of its document). A token starts a new document iff the
    previous token was an eot, so a cumulative count of those boundaries gives
    each token a document id. A query may attend to earlier keys that share its
    document id — i.e. causal AND same-document.

    Built eagerly (outside the compiled model) and passed into the forward pass;
    `flex_attention` fuses the mask into the attention kernel. Returns a
    BlockMask broadcast over heads (H=None).
    """
    bsz, seq_len = idx.shape
    # prev_is_eot[b, i] = (idx[b, i-1] == eot_id), with False at position 0.
    prev_is_eot = F.pad((idx == eot_id)[:, :-1], (1, 0), value=False)
    doc_ids = prev_is_eot.cumsum(dim=1)                       # [B, seq]

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (doc_ids[b, q_idx] == doc_ids[b, kv_idx])

    return create_block_mask(
        mask_mod, B=bsz, H=None, Q_LEN=seq_len, KV_LEN=seq_len,
        device=idx.device, _compile=True,
    )


class SimpleTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=256,
        n_layers=12,
        n_heads=12,
        n_embd=768,
        eot_id=None,
        activation_checkpointing=False,
    ):
        super().__init__()

        # Recompute block activations in backward instead of storing them —
        # trades ~25-30% step time for a large activation-memory cut. Memory
        # knob only: gradients are unchanged. Applies to the training path;
        # cached generation never checkpoints.
        self.activation_checkpointing = activation_checkpointing

        if n_embd % n_heads != 0:
            raise ValueError("n_embd must be divisible by n_heads")

        # Token id that terminates each packed document. When set, the forward
        # pass builds a block-diagonal document mask so attention never crosses
        # a document boundary within a packed block.
        self.eot_id = eot_id
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.head_dim = n_embd // n_heads

        self.token_emb = nn.Embedding(vocab_size, n_embd)

        self.blocks = nn.ModuleList([
            Block(n_embd, n_heads, n_layers) for _ in range(n_layers)
        ])

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=block_size)

        self.ln_f = RMSNorm(n_embd)

        # Untied LM head: a separate output projection (no longer sharing the
        # token-embedding weight). Costs embedding-sized params but typically
        # improves loss at this scale.
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # Blocks initialize their own weights in Block.__init__; here we only
        # init the embedding and the (now untied) LM head, so we don't clobber
        # the blocks' scaled residual-projection init.
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, idx, block_mask=None, past_kvs=None, use_cache=False, return_hidden=False):
        bsz, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError("Sequence length exceeds block_size")

        x = self.token_emb(idx)
        new_kvs = [] if use_cache else None

        # Document masking (a FlexAttention BlockMask) is built by the caller and
        # applies only to the full-context training/eval path. During cached
        # generation each call extends a single sequence, so the plain causal
        # SDPA path in Block handles it and block_mask stays None.

        # RoPE cos/sin are shared across all blocks: compute once for this step.
        offset = past_kvs[0][0].size(2) if past_kvs is not None else 0
        cos, sin = self.rope(seq_len, offset=offset)

        checkpointing = (
            self.activation_checkpointing and self.training
            and torch.is_grad_enabled() and not use_cache
        )
        for layer, block in enumerate(self.blocks):
            past_kv = past_kvs[layer] if past_kvs is not None else None
            if checkpointing:
                x, new_kv = torch.utils.checkpoint.checkpoint(
                    block, x, cos, sin, block_mask, past_kv, use_cache,
                    use_reentrant=False,
                )
            else:
                x, new_kv = block(x, cos, sin, block_mask=block_mask, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                new_kvs.append(new_kv)

        x = self.ln_f(x)
        # Training/eval uses LigerFusedLinearCrossEntropy, which fuses the
        # lm_head projection into the loss — so return the hidden states and let
        # the caller apply the (weight-fused) loss without ever materializing the
        # full logits. Generation still needs real logits to sample from.
        if return_hidden:
            return x, new_kvs
        logits = self.lm_head(x)
        return logits, new_kvs

    @torch.no_grad()
    def generate(
        self,
        idx,
        temperature=1.0,
        top_k=None,
        top_p=None,
        min_p=None,
        repetition_penalty=1.0,
        eos_token_id=None,
        max_new_tokens=None,
    ):
        """
        Sampling hierarchy (applied in order, all optional):
          1. repetition penalty — downweight tokens already seen in the prefix
          2. temperature        — scale logits before any filtering
          3. top-k              — keep only the k highest-logit tokens
          4. top-p              — nucleus: smallest set whose cumulative prob >= p
          5. min-p              — keep tokens where prob(i) >= min_p * prob(argmax)
                            adaptive: bar rises when model is confident,
                            falls when uncertain — best for killing rep loops

        Generation runs until every sequence in the batch has emitted
        `eos_token_id`, or until `max_new_tokens` tokens have been appended
        (whichever comes first). Already-finished rows are padded with EOS so the
        returned tensor stays rectangular. A valid `eos_token_id` is required;
        without it (and without a `max_new_tokens` cap) generation would loop
        forever.
        """
        was_training = self.training
        self.eval()

        try:
            if idx.size(1) == 0:
                raise ValueError("Prompt must contain at least one token")

            if eos_token_id is None and max_new_tokens is None:
                raise ValueError(
                    "generation needs a stop condition: pass eos_token_id, "
                    "max_new_tokens, or both"
                )

            past_kvs = None
            finished = torch.zeros(idx.size(0), 1, dtype=torch.bool, device=idx.device)
            num_generated = 0
            while True:
                if max_new_tokens is not None and num_generated >= max_new_tokens:
                    break
                idx_cond = idx[:, -self.block_size:]
                if past_kvs is None:
                    logits, past_kvs = self.forward(idx_cond, use_cache=True)
                else:
                    logits, past_kvs = self.forward(idx[:, -1:], past_kvs=past_kvs, use_cache=True)
 
                logits = logits[:, -1, :]  # (B, vocab)

                if repetition_penalty is not None and repetition_penalty > 1.0:
                    seen_token_logits = torch.gather(logits, 1, idx)
                    seen_token_logits = torch.where(
                        seen_token_logits < 0,
                        seen_token_logits * repetition_penalty,
                        seen_token_logits / repetition_penalty,
                    )
                    logits.scatter_(1, idx, seen_token_logits)
 
                if temperature <= 0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    if eos_token_id is not None:
                        next_token = torch.where(finished, eos_token_id, next_token)
                        finished = finished | (next_token == eos_token_id)
                    idx = torch.cat([idx, next_token], dim=1)
                    num_generated += 1
                    if eos_token_id is not None and bool(finished.all()):
                        break
                    continue

                logits = logits / temperature
 
                # Top-K
                if top_k is not None and top_k > 0:
                    actual_k = min(top_k, logits.size(-1))
                    values, _ = torch.topk(logits, actual_k)
                    cutoff = values[:, -1].unsqueeze(-1)
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
 
                # Top-P (nucleus)
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
                    softmax_logits = F.softmax(sorted_logits, dim=-1)
                    cumprobs = torch.cumsum(softmax_logits, dim=-1)
                    # Shift right so the token that pushes cumsum over p is kept
                    sorted_remove = cumprobs - softmax_logits >= top_p
                    remove = sorted_remove.scatter(1, sorted_idx, sorted_remove)
                    logits = logits.masked_fill(remove, float("-inf"))
 
                # Min-P
                if min_p is not None and 0.0 < min_p < 1.0:
                    probs = F.softmax(logits, dim=-1)
                    max_prob = probs.max(dim=-1, keepdim=True).values
                    logits = logits.masked_fill(probs < min_p * max_prob, float("-inf"))
 
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                if eos_token_id is not None:
                    next_token = torch.where(finished, eos_token_id, next_token)
                    finished = finished | (next_token == eos_token_id)
                idx = torch.cat([idx, next_token], dim=1)
                num_generated += 1
                if eos_token_id is not None and bool(finished.all()):
                    break

            return idx
        finally:
            if was_training:
                self.train()

class TokenDataset(Dataset):
    def __init__(self, token_ids, block_size):
        length = len(token_ids) - (len(token_ids) - 1) % block_size
        self.token_ids = token_ids[:length]
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.token_ids) - self.block_size)

    def __getitem__(self, idx):
        x = self.token_ids[idx : idx + self.block_size]
        y = self.token_ids[idx + 1 : idx + self.block_size + 1]
        return x, y

# Reserve the first VAL_DOCS documents of the stream for a fixed validation
# set; training skips past them so it never sees val docs.
VAL_DOCS = 2000


def stream_dataset(dataset_path, name=None, split="train", skip=0, take=None):
    """Stream a dataset. skip/take operate on documents.

    Streaming avoids downloading the full shard set up front; HF fetches
    parquet shards on demand and caches them as workers consume the iterator.
    """
    from datasets import load_dataset
    ds = load_dataset(
        dataset_path,
        name,
        split=split,
        streaming=True,
    )
    if skip:
        ds = ds.skip(skip)
    if take is not None:
        ds = ds.take(take)
    return ds


def precompute_val_tokens(tokenizer, eot_id, dataset_path, dataset_name, dataset_split="train"):
    """Tokenize the held-out val docs once at startup into a single tensor."""
    print(f"Tokenizing {VAL_DOCS} val docs from {dataset_path} for validation...")
    tokens = []
    batch, BATCH = [], 256
    def flush():
        if not batch:
            return
        for e in tokenizer.encode_batch(batch):
            tokens.extend(e.ids)
            tokens.append(eot_id)
        batch.clear()
    for ex in stream_dataset(dataset_path, dataset_name, split=dataset_split, take=VAL_DOCS):
        t = ex["text"].strip()
        if not t:
            continue
        batch.append(t)
        if len(batch) >= BATCH:
            flush()
    flush()
    print(f"Val tokens: {len(tokens)/1e6:.1f}M")
    return torch.tensor(tokens, dtype=torch.long)


class StreamingTokenDataset(IterableDataset):
    """
    Streams the configured dataset, tokenizes on the fly, packs the token stream into a
    rolling buffer, and yields non-overlapping (x, y) blocks for next-token
    prediction. Each DataLoader worker takes a distinct shard of the stream
    so workers don't replay the same documents.
    """

    def __init__(self, tokenizer, block_size, eot_id, dataset_path, dataset_name, dataset_split="train", skip_docs=0):
        super().__init__()
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.eot_id = eot_id
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.skip_docs = skip_docs

    def __iter__(self):
        worker_info = get_worker_info()
        ds = stream_dataset(self.dataset_path, self.dataset_name, split=self.dataset_split, skip=self.skip_docs)
        if worker_info is not None:
            ds = ds.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        bs = self.block_size
        buf = []
        for ex in ds:
            t = ex["text"].strip()
            if not t:
                continue
            buf.extend(self.tokenizer.encode(t).ids)
            buf.append(self.eot_id)
            while len(buf) >= bs + 1:
                x = torch.tensor(buf[:bs],      dtype=torch.long)
                y = torch.tensor(buf[1:bs + 1], dtype=torch.long)
                yield x, y
                # Advance by bs; carry the last token so the next block's y
                # remains a strict 1-shift of x without re-tokenizing.
                buf = buf[bs:]


# Fused linear cross entropy: Liger fuses the final lm_head projection with the
# cross-entropy reduction, so the full [B*T, vocab] logits are never
# materialized (a major activation-memory saving at large vocab). The model's
# forward therefore returns the pre-projection hidden states on the training
# path, and the lm_head weight is handed to the loss here.
# z-loss (lse_square_scale) penalizes large logit log-sum-exp, stabilizing the
# output distribution — the standard companion to dropping the tanh logit
# softcap (PaLM, Chameleon).
_liger_flce = LigerFusedLinearCrossEntropyLoss(lse_square_scale=1e-4)

def fused_linear_cross_entropy(hidden, lm_head_weight, targets):
    return _liger_flce(
        lm_head_weight,
        hidden.reshape(-1, hidden.size(-1)),
        targets.reshape(-1),
    )



def estimate_loss(model, val_loader, device, eot_id, eval_iters=20):
    model.eval()
    losses = []
    # Match the training dtype: without autocast the eval forward materializes
    # full fp32 logits (B*T, vocab), which is ~2x the bf16 footprint and spikes
    # peak memory well above the training path.
    autocast_ctx = torch.amp.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda"))
    with torch.no_grad():
        for i, (xb, yb) in enumerate(val_loader):
            if i >= eval_iters: break
            # New CUDA-graph step each iteration: we read loss.item() per pass,
            # so let the graph reuse its static output buffer instead of
            # clobbering a tensor that may still be referenced.
            torch.compiler.cudagraph_mark_step_begin()
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            block_mask = build_document_block_mask(xb, eot_id) if eot_id is not None else None
            with autocast_ctx:
                hidden, _ = model(xb, block_mask=block_mask, use_cache=False, return_hidden=True)
                loss = fused_linear_cross_entropy(hidden, model.lm_head.weight, yb)
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses) if losses else 0.0

def main():
    parser = argparse.ArgumentParser(description="SimpleTransformerLM")
 
    # Training
    parser.add_argument("--max-steps",    type=int,   default=1000)
    parser.add_argument("--batch-size",   type=int,   default=1)
    parser.add_argument("--block-size",   type=int,   default=2048)
    # DCP writes a directory of shards (not a single file), so the default is a
    # directory name rather than a .pt path.
    parser.add_argument("--checkpoint",   type=str,   default="simple_checkpoint")
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--vocab-size",     type=int, default=32768)
    # BPE cache; the tokenizer is corpus-specific, so point this at a distinct
    # file when you change DATASET_PATH rather than reusing one trained on
    # another corpus.
    parser.add_argument("--tokenizer-path", type=str, default="bpe.json")
    parser.add_argument("--compile-mode", type=str,   default="default",
                        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--grad-accum",   type=int, default=99)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--prompt",       type=str, default="Once upon a time")

    args = parser.parse_args()
 
    torch.manual_seed(0)
    random.seed(0)
 
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    # Blackwell: TF32 gives near-FP32 quality at significantly higher throughput.
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.backends.cudnn.benchmark        = True
 
    if device == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Dataset selection — the single knob for switching corpora. DATASET_PATH
    # and DATASET_NAME are passed straight to HF `load_dataset(path, name, ...)`,
    # so any streaming dataset that exposes a "text" column works. Everything
    # downstream (BPE training, the val split, the streaming loader) reads these
    # three constants and is otherwise dataset-agnostic.
    DATASET_PATH = "roneneldan/TinyStories"
    DATASET_NAME = None
    DATASET_SPLIT = "train"
 
    # Train BPE on a bounded sample of the stream — 50k docs is plenty for a
    # 32k vocab and avoids streaming more shards than the tokenizer needs,
    # regardless of which dataset is configured above.
    def corpus_iter():
        for ex in stream_dataset(DATASET_PATH, DATASET_NAME, split=DATASET_SPLIT, take=50_000):
            t = ex["text"].strip()
            if t:
                yield t

    tokenizer = train_or_load_bpe(
        corpus_iter(),
        vocab_size=args.vocab_size,
        save_path=args.tokenizer_path,
    )
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    encode = lambda s: tokenizer.encode(s).ids
    decode = lambda ids: tokenizer.decode(ids)

    block_size = args.block_size

    train_dataset = StreamingTokenDataset(
        tokenizer, block_size, eot_id, DATASET_PATH, DATASET_NAME, dataset_split=DATASET_SPLIT, skip_docs=VAL_DOCS,
    )
    val_data = precompute_val_tokens(tokenizer, eot_id, DATASET_PATH, DATASET_NAME, dataset_split=DATASET_SPLIT)
    val_dataset = TokenDataset(val_data, block_size)

    # EPYC 9355: 48C/96T — 2 workers per loader leaves headroom for the main
    # process and the OS without causing core contention. Each train worker
    # streams its own shard of the dataset so they don't replay docs.
    loader_kwargs = dict(
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
    )

    # IterableDataset can't be shuffled by the loader — the streamed row order
    # is already arbitrary across shards, so this is fine for pretraining.
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    val_loader   = DataLoader(
        val_dataset,   batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
 
    # Pad vocab size to a multiple of 64 for aligned matmuls
    vocab_size        = tokenizer.get_vocab_size()
    padded_vocab_size = ((vocab_size + 63) // 64) * 64
 
    model = SimpleTransformerLM(
        vocab_size=padded_vocab_size,
        block_size=block_size,
        eot_id=eot_id,
        activation_checkpointing=args.activation_checkpointing,
    ).to(device)
 
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    # max-autotune: Blackwell has enough SRAM for the autotuner to find optimal
    # tile configs. First-step compile will be slow (~5-10 min).
    # Drop fullgraph=True if modules.layers has graph breaks.
    if args.compile_mode != "none":
        # Silence the autotuner's per-kernel benchmark tables on stderr.
        from torch._inductor import config as inductor_config, select_algorithm
        select_algorithm.PRINT_AUTOTUNE = False
        inductor_config.max_autotune_report_choices_stats = False
        # checkpoint() introduces graph breaks, so fullgraph must be off then.
        model = torch.compile(model, mode=args.compile_mode,
                              fullgraph=not args.activation_checkpointing)
 
    # Hybrid optimizer: Muon for 2D body matrices, AdamW for everything else.
    # AdamW params are split into two groups: 1D/embedding tensors (norms,
    # LayerScale, QK-gain, embeddings) get weight_decay=0 because decaying
    # scale parameters fights normalization and decaying embeddings harms rare
    # tokens. Only the 2D body matrices warrant decay, and those go to Muon.
    muon_params, adamw_wd_params, adamw_no_wd_params = [], [], []
    # QK-norm weights are 2D (n_heads, head_dim) but are scale parameters, not
    # body matrices — they must not go to Muon and must not be weight-decayed.
    is_scale = lambda n: ("q_norm" in n or "k_norm" in n)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_body_matrix = (
            p.ndim == 2
            and "token_emb" not in name
            and "lm_head" not in name
            and not is_scale(name)
        )
        if is_body_matrix:
            muon_params.append(p)
        elif p.ndim >= 2 and not is_scale(name):  # embedding matrices (token_emb, lm_head)
            adamw_wd_params.append(p)
        else:              # 1D scale params + 2D QK-norm: norms, LayerScale, QK-gain/norm
            adamw_no_wd_params.append(p)

    LR_PEAK = 1e-3

    # adjust_lr_fn="match_rms_adamw" is the Moonshot/Kimi recipe: Muon's
    # internal update scaling is calibrated so the same LR works as AdamW.
    muon_opt = Muon(
        muon_params,
        lr=LR_PEAK,
        momentum=0.95,
        nesterov=True,
        weight_decay=0.1,
        adjust_lr_fn="match_rms_adamw",
    )
    adamw_opt = torch.optim.AdamW(
        [
            {"params": adamw_wd_params,    "weight_decay": 0.1},
            {"params": adamw_no_wd_params, "weight_decay": 0.0},
        ],
        lr=LR_PEAK,
        betas=(0.9, 0.99),
        fused=(device == "cuda"),
    )
    optimizers = [muon_opt, adamw_opt]

    print(
        f"Optimizer split: Muon on {len(muon_params)} matrices "
        f"(wd=0.1), AdamW on {len(adamw_wd_params)} embedding tensors "
        f"(wd=0.1) + {len(adamw_no_wd_params)} 1D tensors (wd=0)."
    )

    max_steps     = args.max_steps
    eval_interval = args.eval_interval
    warmup_steps  = min(99, max(0, max_steps - 1))

    def get_lr(step):
        """Linear warmup to LR_PEAK, then cosine down to 10% of peak."""
        if step < warmup_steps:
            return LR_PEAK * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.1 * LR_PEAK + 0.5 * (LR_PEAK - 0.1 * LR_PEAK) * (1 + math.cos(math.pi * progress))

    # Async checkpoint via PyTorch DCP: save_checkpoint_async stages state on the
    # main thread and writes in the background. We hold the returned Future and
    # pass it back each call so the previous write finishes before the next.
    ckpt_future = None

    step = 0
    train_iter = iter(train_loader)
    if args.resume:
        loaded_step = load_checkpoint(model, optimizers, args.checkpoint, device)
        step = loaded_step + 1
        if step >= max_steps:
            print(
                f"Checkpoint step {loaded_step} already reaches/exceeds "
                f"--max-steps={max_steps}; skipping training."
            )
 
    # BF16 autocast: Blackwell tensor cores have dedicated BF16 throughput paths.
    # GradScaler not needed for BF16 (no underflow risk unlike FP16).
    autocast_ctx = torch.amp.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda"))
 
    while step < max_steps:
        # CUDA graphs (reduce-overhead / max-autotune) reuse static output
        # buffers across invocations. We retain references to graphed outputs
        # past the step boundary (loss.item() at eval, loss accumulated across
        # micro-steps), so mark the step start to let the graph reclaim and
        # reuse that memory safely instead of overwriting live tensors.
        #
        # NOTE: this must stay once-per-optimizer-step, NOT per micro-step.
        # Gradient accumulation reads each micro-step's grad buffer to add the
        # next one; marking per micro-step lets the graph reclaim that buffer
        # early, which corrupts the accumulation ("accessing tensor output of
        # CUDAGraphs that has been overwritten"). The cost is that the graph
        # pool keeps one activation region per micro-step, so peak reserved
        # memory scales with --grad-accum under cudagraph modes. To keep memory
        # flat with grad-accum, drop cudagraphs (--compile-mode default or
        # max-autotune-no-cudagraphs) instead.
        torch.compiler.cudagraph_mark_step_begin()

        lr = get_lr(step)
        for opt in optimizers:
            for param_group in opt.param_groups:
                param_group["lr"] = lr

        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        loss = 0.0
        for micro_step in range(args.grad_accum):
            if micro_step > 0:
                try:
                    xb, yb = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    xb, yb = next(train_iter)
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

            block_mask = build_document_block_mask(xb, eot_id) if eot_id is not None else None
            with autocast_ctx:
                hidden, _ = model(xb, block_mask=block_mask, use_cache=False, return_hidden=True)
                micro_loss = fused_linear_cross_entropy(hidden, model.lm_head.weight, yb) / args.grad_accum

            micro_loss.backward()
            loss += micro_loss.detach()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()

        if step % eval_interval == 0:
            val_loss  = estimate_loss(model, val_loader, device, eot_id)
            peak_alloc = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
            peak_reserved = torch.cuda.max_memory_reserved() / 1e9 if device == "cuda" else 0.0
            if device == "cuda": torch.cuda.reset_peak_memory_stats()
            print(
                f"step {step:04d} | lr {lr:.2e} | "
                f"train loss {loss.item():.4f} | val loss {val_loss:.4f} | "
                f"peak active {peak_alloc:.1f}GB | peak reserved {peak_reserved:.1f}GB"
            )
            ckpt_future = save_checkpoint_async(
                model, optimizers, step, args.checkpoint, prev_future=ckpt_future
            )

        step += 1

    if ckpt_future is not None:
        ckpt_future.result()

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------
    print("\n--- Generating Sample Text ---\n")
    context = torch.tensor([encode(args.prompt)], device=device)

    configs = {
        "top_k=40":                dict(temperature=0.8, top_k=40,   top_p=None, min_p=None, repetition_penalty=1.1),
        "top_p=0.9":               dict(temperature=0.8, top_k=None, top_p=0.9,  min_p=None, repetition_penalty=1.1),
        "min_p=0.05":              dict(temperature=0.8, top_k=None, top_p=None, min_p=0.05, repetition_penalty=1.1),
        "top_p=0.9 + min_p=0.05":  dict(temperature=0.8, top_k=None, top_p=0.9,  min_p=0.05, repetition_penalty=1.1),
        "RepPenalty=1.2":          dict(temperature=0.8, top_k=None, top_p=None, min_p=None, repetition_penalty=1.2),
        "Temp=0.5":                dict(temperature=0.5, top_k=None, top_p=None, min_p=None, repetition_penalty=1.1),
        "Temp=1.5":                dict(temperature=1.5, top_k=None, top_p=None, min_p=None, repetition_penalty=1.1),
    }
 
    for label, cfg in configs.items():
        print(f"[{label}]")
        with torch.no_grad():
            out = model.generate(
                context.clone(),
                eos_token_id=eot_id,
                max_new_tokens=256,
                **cfg,
            )
        print(decode(list(out[0].tolist())))
        print()

if __name__ == "__main__":
    main()
