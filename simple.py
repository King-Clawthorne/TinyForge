import os
import math
import random
import argparse
import copy
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info

from modules.muon import Muon

from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss

from modules.layers import RMSNorm, RotaryEmbedding, apply_rope
from modules.utils import (
    load_checkpoint,
    save_checkpoint,
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
    "expandable_segments:True,max_split_size_mb:512",
)

class SimpleTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=256,
        n_layers=12,
        n_heads=12,
        n_embd=768,
        logit_softcap=30.0,
    ):
        super().__init__()
        self.logit_softcap = logit_softcap

        if n_embd % n_heads != 0:
            raise ValueError("n_embd must be divisible by n_heads")

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.head_dim = n_embd // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.token_emb = nn.Embedding(vocab_size, n_embd)

        self.ln1 = nn.ModuleList([RMSNorm(n_embd) for _ in range(n_layers)])
        self.ln2 = nn.ModuleList([RMSNorm(n_embd) for _ in range(n_layers)])

        self.qkv = nn.ModuleList([nn.Linear(n_embd, 3 * n_embd, bias=False) for _ in range(n_layers)])
        self.proj = nn.ModuleList([nn.Linear(n_embd, n_embd, bias=False) for _ in range(n_layers)])

        # QK-gain: learnable per-head multiplicative gain on the attention scale.
        # Init to 1.0 so initial behavior matches the fixed 1/sqrt(d_k) baseline.
        self.qk_gain = nn.ParameterList([
            nn.Parameter(torch.ones(n_heads)) for _ in range(n_layers)
        ])

        self.w_up = nn.ModuleList()
        self.w_gate = nn.ModuleList()
        self.w_down = nn.ModuleList()

        for _ in range(n_layers):
            hidden_dim = int(8 * n_embd / 3)
            hidden_dim = ((hidden_dim + 127) // 128) * 128
            self.w_up.append(nn.Linear(n_embd, hidden_dim, bias=False))
            self.w_gate.append(nn.Linear(n_embd, hidden_dim, bias=False))
            self.w_down.append(nn.Linear(hidden_dim, n_embd, bias=False))

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=block_size)

        # LayerScale (CaiT, Touvron et al. 2021) — learnable per-channel residual gates.
        # Init to 0.1 for a ~100M-class model; use 1e-4 for deeper stacks (>24 layers).
        _ls_init = 0.1
        self.ls_attn = nn.ParameterList([
            nn.Parameter(torch.full((n_embd,), _ls_init)) for _ in range(n_layers)
        ])
        self.ls_mlp = nn.ParameterList([
            nn.Parameter(torch.full((n_embd,), _ls_init)) for _ in range(n_layers)
        ])
 

        self.ln_f = RMSNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

        for i in range(n_layers):
            nn.init.normal_(
                self.proj[i].weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * n_layers),
            )
            nn.init.normal_(
                self.w_down[i].weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * n_layers),
            )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, past_kvs=None, use_cache=False):
        bsz, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError("Sequence length exceeds block_size")

        x = self.token_emb(idx)
        new_kvs = [] if use_cache else None
        for layer in range(self.n_layers):
            residual = x
            x_norm = self.ln1[layer](x)
            qkv = self.qkv[layer](x_norm)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(bsz, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            k = k.view(bsz, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            v = v.view(bsz, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
            offset = past_kvs[layer][0].size(2) if past_kvs is not None else 0
            cos, sin = self.rope(q.size(2), offset=offset)
            q, k = apply_rope(q, k, cos, sin)

            past_len = 0
            if past_kvs is not None:
                past_k, past_v = past_kvs[layer]
                past_len = past_k.size(2)
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)

            # QK-gain: fold the learnable per-head gain into q so it scales the
            # q·k score before softmax — equivalent to the old score_mod's
            # `score * gain[h]` but expressed as a plain tensor op SDPA can fuse.
            q = q * self.qk_gain[layer].view(1, self.n_heads, 1, 1).to(q.dtype)

            attn_mask = None
            is_causal = True
            if past_len > 0:
                total_len = past_len + seq_len
                query_positions = past_len + torch.arange(seq_len, device=q.device)
                key_positions = torch.arange(total_len, device=q.device)
                attn_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                is_causal = False

            # flex_attention uses Triton kernels that don't yet compile cleanly on
            # Blackwell (sm_120a); cuDNN fused attention is faster there anyway.
            with sdpa_kernel(
                [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
                 SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
                set_priority=True,
            ):
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, dropout_p=0.0,
                    is_causal=is_causal, scale=self.scale,
                )

            y = y.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, self.n_embd)
            y = self.proj[layer](y)
            x = residual + self.ls_attn[layer] * y
            residual = x
            x_norm = self.ln2[layer](x)
            up = self.w_up[layer](x_norm)
            gate = self.w_gate[layer](x_norm)
            mlp_out = self.w_down[layer](LigerSiLUMulFunction.apply(gate, up))
            x = residual + self.ls_mlp[layer] * mlp_out
            if use_cache: new_kvs.append((k, v))

        x = self.ln_f(x)
        logits = self.lm_head(x)
        if self.logit_softcap is not None and self.logit_softcap > 0:
            cap = self.logit_softcap
            # Fold the softcap to avoid extra full-size (B*T, vocab) temporaries.
            # div_ and tanh_ run in-place on the logits buffer; the final `* cap`
            # must stay out-of-place because TanhBackward needs an unmodified
            # copy of the tanh output for the backward pass.
            logits = logits.div_(cap).tanh_() * cap
        return logits, new_kvs

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
        min_p=None,
        repetition_penalty=1.0,
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
        """
        was_training = self.training
        self.eval()
 
        try:
            if idx.size(1) == 0:
                raise ValueError("Prompt must contain at least one token")
 
            past_kvs = None
            for _ in range(max_new_tokens):
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
                    idx = torch.cat([idx, next_token], dim=1)
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
                idx = torch.cat([idx, next_token], dim=1)
 
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
    Streams FineWeb-Edu, tokenizes on the fly, packs the token stream into a
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


_liger_ce = LigerCrossEntropyLoss()

def chunked_cross_entropy(logits, targets):
    return _liger_ce(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))



def estimate_loss(model, val_loader, device, eval_iters=20):
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
            with autocast_ctx:
                logits, _ = model(xb, use_cache=False)
                loss = chunked_cross_entropy(logits, yb)
            losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses) if losses else 0.0

def main():
    parser = argparse.ArgumentParser(description="SimpleTransformerLM — FineWeb-Edu (streaming)")
 
    # Training
    parser.add_argument("--max-steps",    type=int,   default=9999)
    parser.add_argument("--batch-size",   type=int,   default=99)
    parser.add_argument("--block-size",   type=int,   default=2048)
    parser.add_argument("--checkpoint",   type=str,   default="simple_checkpoint.pt")
    parser.add_argument("--resume",       action="store_true")
    parser.add_argument("--vocab-size",     type=int, default=32768)
    parser.add_argument("--tokenizer-path", type=str, default="fineweb_edu_bpe.json")
    parser.add_argument("--compile-mode", type=str,   default="default",
                        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
    parser.add_argument("--eval-interval", type=int, default=999)
    parser.add_argument("--grad-accum",   type=int, default=1)
    parser.add_argument("--prompt",       type=str, default="The capital of France")
    parser.add_argument("--max-new-tokens", type=int, default=100)

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

    DATASET_PATH = "roneneldan/TinyStories"
    DATASET_NAME = None
    DATASET_SPLIT = "train"
 
    # Train BPE on a sample of the stream — 50k docs (~60M chars) is plenty
    # for a 32k vocab and avoids materializing the full 10BT shard set.
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
    # streams its own shard of FineWeb-Edu so they don't replay docs.
    loader_kwargs = dict(
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True,
    )

    # IterableDataset can't be shuffled by the loader — FineWeb-Edu's row order
    # is already arbitrary across shards, so this is fine for pretraining.
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, **loader_kwargs)
 
    # Pad vocab size to a multiple of 64 for aligned matmuls
    vocab_size        = tokenizer.get_vocab_size()
    padded_vocab_size = ((vocab_size + 63) // 64) * 64
 
    model = SimpleTransformerLM(
        vocab_size=padded_vocab_size,
        block_size=block_size,
    ).to(device)
 
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.1f}M")

    # max-autotune: Blackwell has enough SRAM for the autotuner to find optimal
    # tile configs. First-step compile will be slow (~5-10 min).
    # Drop fullgraph=True if modules.layers has graph breaks.
    if args.compile_mode != "none":
        model = torch.compile(model, mode=args.compile_mode, fullgraph=True)
 
    # Hybrid optimizer: Muon for 2D body matrices, AdamW for everything else.
    # AdamW params are split into two groups: 1D/embedding tensors (norms,
    # LayerScale, QK-gain, embeddings) get weight_decay=0 because decaying
    # scale parameters fights normalization and decaying embeddings harms rare
    # tokens. Only the 2D body matrices warrant decay, and those go to Muon.
    muon_params, adamw_wd_params, adamw_no_wd_params = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_body_matrix = (
            p.ndim == 2
            and "token_emb" not in name
            and "lm_head" not in name
        )
        if is_body_matrix:
            muon_params.append(p)
        elif p.ndim >= 2:  # embedding matrix (token_emb / lm_head, tied)
            adamw_wd_params.append(p)
        else:              # 1D: norm weights, LayerScale, QK-gain
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

    # Async checkpoint: snapshot state dicts on the main thread, write to disk
    # in a background thread so the training loop isn't blocked by I/O.
    _ckpt_thread: threading.Thread | None = None

    def save_checkpoint_async(model, optimizers, step, path):
        nonlocal _ckpt_thread
        if _ckpt_thread is not None and _ckpt_thread.is_alive():
            _ckpt_thread.join()
        # Move tensors to CPU before deepcopy so the snapshot lives in RAM,
        # not VRAM — otherwise we'd double peak GPU memory at every checkpoint.
        snapshot = {
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dicts": [
                {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                 for k, v in o.state_dict().items()}
                for o in optimizers
            ],
            "step": step,
        }
        def _write():
            torch.save(snapshot, path)
            print(f"Checkpoint saved to {path} at step {step}")
        _ckpt_thread = threading.Thread(target=_write, daemon=True)
        _ckpt_thread.start()

    step = 0
    cache_clear_step = 4          # fresh: empty_cache on the 3rd step
    train_iter = iter(train_loader)
    if args.resume:
        loaded_step = load_checkpoint(model, optimizers, args.checkpoint, device)
        step = loaded_step + 1
        cache_clear_step += loaded_step
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

            with autocast_ctx:
                logits, _ = model(xb, use_cache=False)
                micro_loss = chunked_cross_entropy(logits, yb) / args.grad_accum

            micro_loss.backward()
            loss += micro_loss.detach()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()

        if step == cache_clear_step:
            torch.cuda.empty_cache()
            print(f"CUDA cache cleared to reduce fragmentation for KV cache.")

        if step % eval_interval == 0:
            val_loss  = estimate_loss(model, val_loader, device)
            peak_alloc = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
            peak_reserved = torch.cuda.max_memory_reserved() / 1e9 if device == "cuda" else 0.0
            if device == "cuda": torch.cuda.reset_peak_memory_stats()
            print(
                f"step {step:04d} | lr {lr:.2e} | "
                f"train loss {loss.item():.4f} | val loss {val_loss:.4f} | "
                f"peak active {peak_alloc:.1f}GB | peak reserved {peak_reserved:.1f}GB"
            )
            save_checkpoint_async(model, optimizers, step, args.checkpoint)

        step += 1

    if _ckpt_thread is not None:
        _ckpt_thread.join()

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
                max_new_tokens=args.max_new_tokens,
                **cfg,
            )
        print(decode(list(out[0].tolist())))
        print()

if __name__ == "__main__":
    main()
