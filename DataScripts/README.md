# DataScripts — factorial ablation study

Scripts that generate all data for the paper *"Which modern transformer
tricks are redundant?"* — a factorial study of how modern training-recipe
components interact at ~100M-parameter scale. All outputs land in
`../DataOutput/`.

## Factors under study (all binary)

| factor | OFF | ON |
|---|---|---|
| `optimizer` | pure AdamW | Muon (2D body) + AdamW hybrid |
| `qk_norm` | — | per-head RMSNorm on q/k (Gemma 2) |
| `qk_gain` | — | learnable per-head attention-score gain |
| `layerscale` | plain residual | LayerScale gates (CaiT) |
| `value_residual` | — | ResFormer value residual |
| `z_loss` | — | `lse_square_scale=1e-4` |
| `softcap` | — | tanh logit softcap (30.0) |

Model size, data (TinyStories streaming), tokenizer, LR schedule, and budget
are held fixed across all runs.

## Workflow

Run everything from the **repo root**:

```bash
# 0. One-time: download + tokenize the corpus to local .bin files
#    (training runs then need no network at all)
python DataScripts/prepare_data.py

# 1. Preview the run list (~22 configs per seed) without training
python DataScripts/run_matrix.py --dry-run

# 2. Run the full matrix (sequential, one GPU; resumable — re-running
#    skips any config that already has a summary.json)
python DataScripts/run_matrix.py

# 3. Aggregate into paper tables + figures
python DataScripts/analyze_results.py
```

Useful knobs on `run_matrix.py` (passed through to every run):
`--max-steps` (default 1000), `--batch-size` (4), `--grad-accum` (16),
`--block-size` (2048), `--seeds` (1), `--compile-mode` (default).
Default budget is ~131M tokens per run. For a cheap pilot first:
`python DataScripts/run_matrix.py --max-steps 200 --grad-accum 8`.

A single configuration can also be run directly, e.g.:

```bash
python DataScripts/train_ablation.py --optimizer muon --qk-norm --z-loss
```

## Outputs (`DataOutput/`)

- `matrix_manifest.json` — the full experiment matrix
- `runs/<run_id>/config.json` — resolved config of the run
- `runs/<run_id>/log.jsonl` — per-eval records (step, train/val loss, lr, grad norm, tokens, wall time)
- `runs/<run_id>/summary.json` — final/best val loss, divergence flag, timings
- `analysis/results.csv` — one row per run
- `analysis/main_effects.csv` — per-factor Δ val loss over matched pairs
- `analysis/interactions.csv` — 2×2 interaction terms for the targeted pairs
- `analysis/summary.md` — human-readable digest
- `analysis/loss_curves*.png` — figures (requires matplotlib)

## Files

- `ablation_model.py` — `AblationTransformerLM` with every factor toggleable
- `train_ablation.py` — one training run; reuses the data pipeline from `simple.py`
- `run_matrix.py` — builds + executes the experiment matrix (resumable)
- `analyze_results.py` — tables, interaction analysis, figures

Diverged runs (non-finite loss) stop early, are flagged in `summary.json`,
and are excluded from effect/interaction estimates — divergence itself is a
result worth reporting.
