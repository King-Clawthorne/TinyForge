# Cohen-Transformer

## Situation
Cohen-Transformer is a compact PyTorch-based transformer language model implementation designed for efficient autoregressive generation and training. It includes a simple decoder-only model, rotary position embeddings, RMS normalization, and training utilities for tokenizer management and checkpointing.

## Task
The goal is to provide a clear, concise project overview and usage guide that helps a developer understand what this code does, how it is structured, and how to run or extend it.

## Action
- Implemented a minimal transformer language model in `simple.py` with multi-head attention, gated MLP, LayerScale residual gating, and rotary embeddings.
- Included tokenizer training and checkpoint utilities in `modules/utils.py` to support dataset preparation, model saving, and resume training.
- Organized the model components in `modules/layers.py` and the utility functions in `modules/utils.py` for modular structure and easier extension.

## Result
- Developers can inspect `simple.py` to understand model architecture and generation logic.
- Tokenizer caching and checkpoint handling are available out of the box for training workflows.
- This repository provides a solid foundation for experimenting with transformer LM training and generation on custom data.

## Quick Start
1. Install dependencies:
   ```bash
   pip install torch datasets tokenizers
   ```
2. Review `simple.py` for model and generation details.
3. Use `modules/utils.py` helpers to train or load a BPE tokenizer and save/load checkpoints.

## Files
- `simple.py` — main transformer language model implementation
- `modules/layers.py` — transformer building blocks like `RMSNorm`, `RotaryEmbedding`, and RoPE utilities
- `modules/utils.py` — tokenizer, checkpoint, and dataset helper functions

## Notes
This README follows the STAR method to present the project in a structured way: Situation, Task, Action, Result.
