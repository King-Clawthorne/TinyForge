# Cohen-Transformer

## Situation

Cohen-Transformer is a compact PyTorch decoder-only transformer language model with a modern training stack: a custom BPE tokenizer, streaming pretraining on FineWeb-Edu, a Muon + AdamW hybrid optimizer, and KV-cache-accelerated generation. It is designed to fit a ~100M-parameter model on a single Blackwell-class GPU while keeping the implementation small enough to read end to end.

## Task

Provide a clear, concise project overview and usage guide that helps a developer understand the model's architecture, the training pipeline, and how to run or extend it.

## Action

- Implemented a minimal ~100M-parameter decoder-only transformer in `simple.py` with multi-head attention, SwiGLU MLP, RMSNorm, rotary position embeddings (RoPE), LayerScale residual gating, QK-gain, logit softcap, weight-tied embeddings/lm_head, and KV-cache-aware generation with top-k / top-p / min-p / repetition-penalty samplers.
- Switched the optimizer to `torch.optim.Muon` for the body's 2D weight matrices (Newton-Schulz–orthogonalized momentum with the Moonshot `match_rms_adamw` LR adjustment) paired with AdamW for embeddings, RMSNorm gains, and LayerScale vectors.
- Swapped training data from TinyStories to a streaming `HuggingFaceFW/fineweb-edu` `sample-10BT` pipeline: per-worker shard splitting on an `IterableDataset`, a pre-tokenized held-out val set, and a 32k custom BPE trained on a 50k-doc sample.
- Added an offline pre-tokenization mode (`--pretokenize`) that writes a flat uint16 `.bin` plus sidecar metadata, and a `MemmapTokenDataset` that mmaps it across DataLoader workers — eliminates the in-loader tokenization bottleneck for repeat runs.
- Migrated the training attention path to `torch.nn.attention.flex_attention` with a cached causal `BlockMask` and QK-gain folded into the `score_mod` closure, so mask + per-head gain fuse into one Triton kernel. KV-cache decode stays on SDPA (q_len=1, BlockMask rebuild would dominate).
- Added `AsyncCheckpointer`: snapshots model + optimizer state to CPU synchronously, then `torch.save`s on a background thread with atomic temp-file rename. At most one outstanding write at a time so a slow disk applies backpressure instead of piling snapshots in RAM.
- Centralized tokenizer training, (multi-optimizer) checkpointing, async checkpointing, and the pre-tokenization writer in `modules/utils.py`; core building blocks (`RMSNorm`, `RotaryEmbedding`, `apply_rope`) in `modules/layers.py`.

## Result

- A self-contained training script that reads top-to-bottom, runs on a single Blackwell-class GPU, and reaches modern recipe parity (Muon + FineWeb-Edu + BF16/FP8 + `torch.compile` + flex_attention + async checkpoints) without any framework on top of PyTorch.
- Tokenizer cache, multi-optimizer checkpoint format, streaming + mmap datasets, and an FP8 + flex_attention–friendly model definition all reusable for further experiments.
- Solid foundation for trying architectural variants (GQA, deeper stacks, alternative residual schemes) against a known-good baseline.

## Quick Start

1. Install dependencies (PyTorch nightly recommended for Blackwell / flex_attention; `torchao` only needed for `--fp8`):

   ```bash
   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
   pip install datasets tokenizers
   pip install torchao  # optional, for --fp8
   ```

2. (Optional, one-time) Pre-tokenize FineWeb-Edu to a flat `.bin` so future runs skip in-loader tokenization:

   ```bash
   python simple.py --pretokenize
   ```

3. Train (auto-uses `.bin` files if present; otherwise streams + tokenizes on the fly):

   ```bash
   python simple.py --max-steps 9999 --batch-size 99 --block-size 2048
   ```

4. Train with FP8 body matmuls on Blackwell (~1.5–1.8× step-time speedup):

   ```bash
   python simple.py --fp8 --max-steps 9999 --batch-size 192 --block-size 2048
   ```

5. Resume from a checkpoint and skip straight to generation:

   ```bash
   python simple.py --resume --max-steps 0 --prompt "The capital of France"
   ```

## Files

- `simple.py` — model, training loop, FineWeb-Edu streaming dataset, memmap dataset, pre-tokenization CLI, FP8 wiring, flex_attention path, and generation
- `modules/layers.py` — `RMSNorm`, `RotaryEmbedding`, `apply_rope`
- `modules/utils.py` — BPE training/loading, sync + async checkpoint save/load (single or list of optimizers), `pretokenize_to_bin` writer, dataset helpers

## Notes

This README follows the STAR method: Situation, Task, Action, Result.
