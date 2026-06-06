# Cohen-Transformer

## Situation

Cohen-Transformer is a compact PyTorch decoder-only transformer language model with a modern training stack: a custom BPE tokenizer, streaming pretraining on TinyStories, a Muon + AdamW hybrid optimizer, and KV-cache-accelerated generation. It is designed to fit a ~100M-parameter model on a single Blackwell-class GPU while keeping the implementation small enough to read end to end.

## Task

Provide a clear, concise project overview and usage guide that helps a developer understand the model's architecture, the training pipeline, and how to run or extend it.

## Action

- Implemented a minimal ~100M-parameter decoder-only transformer in `simple.py`, built from a reusable `Block` module: multi-head attention, SwiGLU MLP, RMSNorm, rotary position embeddings (RoPE) with dynamic NTK-aware context scaling, LayerScale residual gating, per-head QK-norm plus learnable QK-gain, an untied LM head, and KV-cache-aware generation with top-k / top-p / min-p / repetition-penalty samplers that runs until every sequence in the batch emits EOS (a valid `eos_token_id` is required — EOS is the only stop condition).
- Uses a vendored Muon optimizer (`modules/muon.py`) for the body's 2D weight matrices (Newton-Schulz–orthogonalized momentum with the Moonshot `match_rms_adamw` LR adjustment, weight decay 0.1), paired with AdamW for the embedding / LM-head matrices (weight decay 0.1) and the 1D scale parameters — RMSNorm gains, LayerScale, and QK-gain/QK-norm (weight decay 0).
- Streams `roneneldan/TinyStories` through a tokenize-on-the-fly `IterableDataset`: per-worker shard splitting so workers never replay the same documents, a fixed held-out validation set (the first 2000 documents, tokenized once at startup), and a 32k custom byte-level BPE trained on a 50k-document sample of the stream.
- Runs attention through `F.scaled_dot_product_attention` with the cuDNN fused backend prioritized via `sdpa_kernel(..., set_priority=True)`, applying per-head QK-norm before RoPE and folding the learnable per-head QK-gain into `q` so it scales the score before softmax. Training/prefill uses the fast causal flag; KV-cache decode swaps in an explicit lower-right causal mask.
- Packs documents end-to-end separated by `<|endoftext|>` and builds a block-diagonal document mask (`_build_document_mask`) on the training/eval path, so attention never crosses a document boundary within a packed block; the mask takes SDPA's `is_causal` slot since it already encodes causality.
- Stabilized the output distribution with z-loss (Liger cross-entropy `lse_square_scale`) in place of a tanh logit softcap.
- Centralized tokenizer training and (multi-optimizer) checkpointing in `modules/utils.py`, using PyTorch's built-in async Distributed Checkpoint (`dcp.async_save`) so saves write in the background without blocking training; core building blocks (`RMSNorm`, `RotaryEmbedding`, `QKNorm`, `Block`, `apply_rope`) in `modules/layers.py`.

## Result

- A self-contained training script that reads top-to-bottom, runs on a single Blackwell-class GPU, and reaches modern recipe parity (Muon + streaming TinyStories + BF16 + `torch.compile` + cuDNN SDPA) without any framework on top of PyTorch.
- Tokenizer cache, multi-optimizer checkpoint format, streaming dataset, and a compact model definition all reusable for further experiments.
- Solid foundation for trying architectural variants (GQA, deeper stacks, alternative residual schemes) against a known-good baseline.

## Quick Start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Train:

   ```bash
   python simple.py --max-steps 9999 --batch-size 99 --block-size 2048
   ```

3. Resume from a checkpoint and skip straight to generation:

   ```bash
   python simple.py --resume --max-steps 0 --prompt "The capital of France"
   ```

## Files

- `simple.py` — model, training loop, TinyStories streaming dataset, document masking, cuDNN SDPA path, and generation
- `modules/layers.py` — `RMSNorm`, `RotaryEmbedding` (NTK-aware), `QKNorm`, `Block`, `apply_rope`
- `modules/muon.py` — Muon optimizer implementation
- `modules/utils.py` — BPE training/loading, async DCP checkpoint save/load (single or list of optimizers; writes a checkpoint directory), dataset helpers
