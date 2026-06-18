# DataScripts — factorial ablation study

Scripts that generate all data for the paper *"Which modern transformer
tricks are redundant?"* — a factorial study of how modern training-recipe
components interact at ~100M-parameter scale. All outputs land in
`../DataOutput/`.

## Factors under study (all binary)

| factor | OFF | ON |
| --- | --- | --- |
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

## Verdict-flip sweep (z-loss / softcap)

The main matrix fixes scale, schedule, and vocab at one operating point, where
z-loss and the tanh logit softcap come out inert-to-harmful — expected, since
both are output-softmax stabilizers and an 8k-vocab / 1000-step softmax has
little log-sum-exp blowup to stabilize. `flip_sweep.py` finds *where the verdict
flips* by replicating the z-loss × softcap 2×2 along the three axes that drive
logit magnitude:

```bash
# vocab is the dominant lever (8192 → 32768); --auto-prepare tokenizes
# any missing vocab (downloads are cached after the first time)
python DataScripts/flip_sweep.py --axis vocab --auto-prepare
python DataScripts/flip_sweep.py --axis all        # also schedule + scale
python DataScripts/flip_analyze.py                 # per-regime effects + flip point
```

The vocab axis tops out at 32768: BPE on TinyStories saturates at ~26.9k
tokens (only ~20.5k unique pre-tokenized words), so a larger target vocab is
backed by the identical tokenizer and is not a distinct regime. The other axes
are `schedule` (max-steps 500 → 4000) and `scale` (4×4×256 → 16×16×1024). The
sweep is resumable and continues past a failed cell, so a re-run fills only the
gaps.

Runs land in `DataOutput/flip_runs/` (separate from the main matrix, so
`analyze_results.py` is unaffected). `flip_analyze.py` writes
`analysis/flip_effects.csv` and prints, per axis, the z-loss/softcap main effect
at each regime (Δ final val loss, ON−OFF; negative = useful) and the first
regime where each crosses into "useful".

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
- `flip_sweep.py` — z-loss / softcap verdict-flip sweep over vocab, schedule, scale (resumable)
- `flip_analyze.py` — per-regime z-loss/softcap main effects + the flip point
- `prepare_data.py` — download + BPE-tokenize TinyStories to `DataOutput/tokens/*.bin`

Diverged runs (non-finite loss) stop early, are flagged in `summary.json`,
and are excluded from effect/interaction estimates — divergence itself is a
result worth reporting.
