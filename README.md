# Cohen-Transformer

## Situation

Cohen-Transformer is a compact PyTorch decoder-only transformer language model with a modern training stack: a custom BPE tokenizer, streaming pretraining on FineWeb-Edu, a Muon + AdamW hybrid optimizer, and KV-cache-accelerated generation. It is designed to fit a ~100M-parameter model on a single Blackwell-class GPU while keeping the implementation small enough to read end to end.

## Task

Provide a clear, concise project overview and usage guide that helps a developer understand the model's architecture, the training pipeline, and how to run or extend it.

## Action

- Implemented a minimal ~100M-parameter decoder-only transformer in `simple.py` with multi-head attention, SwiGLU MLP, RMSNorm, rotary position embeddings (RoPE), LayerScale residual gating, QK-gain, logit softcap, weight-tied embeddings/lm_head, and KV-cache-aware generation with top-k / top-p / min-p / repetition-penalty samplers.
- Switched the optimizer to `torch.optim.Muon` for the body's 2D weight matrices (Newton-Schulz–orthogonalized momentum with the Moonshot `match_rms_adamw` LR adjustment) paired with AdamW for embeddings, RMSNorm gains, and LayerScale vectors.
- Swapped training data from TinyStories to a streaming `HuggingFaceFW/fineweb-edu` `sample-10BT` pipeline: per-worker shard splitting on an `IterableDataset`, an on-the-fly tokenized held-out val set, and a 32k custom BPE trained on a 50k-doc sample.
- Runs attention through `F.scaled_dot_product_attention` with the cuDNN fused backend prioritized via `sdpa_kernel(..., set_priority=True)`, folding the learnable per-head QK-gain into `q` so it scales the score before softmax. Training/prefill uses the fast causal flag; KV-cache decode swaps in an explicit lower-right causal mask.
- Centralized tokenizer training and (multi-optimizer) checkpoint save/load in `modules/utils.py`; core building blocks (`RMSNorm`, `RotaryEmbedding`, `apply_rope`) in `modules/layers.py`.

## Result

- A self-contained training script that reads top-to-bottom, runs on a single Blackwell-class GPU, and reaches modern recipe parity (Muon + FineWeb-Edu + BF16/FP8 + `torch.compile` + cuDNN SDPA) without any framework on top of PyTorch.
- Tokenizer cache, multi-optimizer checkpoint format, streaming dataset, and an FP8-friendly model definition all reusable for further experiments.
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

- `simple.py` — model, training loop, FineWeb-Edu streaming dataset, FP8 wiring, cuDNN SDPA path, and generation
- `modules/layers.py` — `RMSNorm`, `RotaryEmbedding`, `apply_rope`
- `modules/muon.py` — Muon optimizer implementation
- `modules/utils.py` — BPE training/loading, checkpoint save/load (single or list of optimizers), dataset helpers
