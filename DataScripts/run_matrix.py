"""Experiment matrix for the factorial ablation study.

Builds the run list and executes each configuration via train_ablation.py
(sequentially, one GPU). Resumable: runs whose summary.json already exists
in DataOutput/runs/ are skipped. The full matrix is written to
DataOutput/matrix_manifest.json before anything runs.

Design (7 binary factors would be 128 runs at full factorial; we use a
targeted fraction, ~26 unique configs per seed):

  1. Anchors: all-OFF vanilla baseline (AdamW), all-ON full modern recipe (Muon).
  2. Add-one-in:    baseline + each single factor          (7 runs)
  3. Leave-one-out: full recipe - each single factor       (7 runs)
  4. Targeted interaction cells from plan.txt:
       Muon x QK-norm                 (2x2, corners shared with 1-2)
       LayerScale x value-residual    (2x2, corners shared with 1-2)
       z-loss x softcap x QK-gain     (2x2x2, corners shared with 1-2)

Usage (from repo root):
  python DataScripts/run_matrix.py --dry-run          # print the run list
  python DataScripts/run_matrix.py                    # run everything
  python DataScripts/run_matrix.py --seeds 3          # 3 seeds per config
  python DataScripts/run_matrix.py --max-steps 500    # cheaper pilot
  python DataScripts/run_matrix.py --start-at 5       # begin at the 5th job
"""

import sys
import json
import argparse
import itertools
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from DataScripts.common import run_id_from_config, BOOL_FACTORS


def base_config(**overrides):
    cfg = {"optimizer": "adamw", **{f: False for f in BOOL_FACTORS}}
    cfg.update(overrides)
    return cfg


def full_config(**overrides):
    cfg = {"optimizer": "muon", **{f: True for f in BOOL_FACTORS}}
    cfg.update(overrides)
    return cfg


def build_matrix():
    configs = []

    def add(cfg, group):
        configs.append({**cfg, "group": group})

    # 1. Anchors
    add(base_config(), "anchor")
    add(full_config(), "anchor")

    # 2. Add-one-in from the vanilla baseline
    add(base_config(optimizer="muon"), "add_one")
    for f in BOOL_FACTORS:
        add(base_config(**{f: True}), "add_one")

    # 3. Leave-one-out from the full recipe
    add(full_config(optimizer="adamw"), "leave_one_out")
    for f in BOOL_FACTORS:
        add(full_config(**{f: False}), "leave_one_out")

    # 4a. Muon x QK-norm (others off)
    for opt, norm in itertools.product(["adamw", "muon"], [False, True]):
        add(base_config(optimizer=opt, qk_norm=norm), "muon_x_qknorm")

    # 4b. LayerScale x value-residual (others off, AdamW)
    for ls, vr in itertools.product([False, True], repeat=2):
        add(base_config(layerscale=ls, value_residual=vr), "ls_x_vres")

    # 4c. z-loss x softcap x QK-gain (others off, AdamW)
    for z, cap, gain in itertools.product([False, True], repeat=3):
        add(base_config(z_loss=z, softcap=cap, qk_gain=gain), "zloss_x_cap_x_gain")

    # Dedupe by factor settings, keeping the first group label.
    seen, unique = {}, []
    for cfg in configs:
        key = (cfg["optimizer"],) + tuple(cfg[f] for f in BOOL_FACTORS)
        if key in seen:
            seen[key]["group"] += f"+{cfg['group']}"
        else:
            seen[key] = cfg
            unique.append(cfg)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Run the ablation matrix")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--start-at", type=int, default=1,
                        help="Skip to the Nth job (1-based, matching the "
                             "[N/total] index in the run list). Jobs before N "
                             "are skipped outright without checking results.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "DataOutput"))
    # Pass-through training knobs (defaults match train_ablation.py)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=15)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--activation-checkpointing", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Fail fast if the pretokenized data isn't there (train_ablation.py
    # default vocab is 8192); otherwise all jobs would fail one by one.
    if not args.dry_run and not list((out_dir / "tokens").glob("meta_*.json")):
        sys.exit("Pretokenized data not found in DataOutput/tokens.\n"
                 "Run once first:  python DataScripts/prepare_data.py")

    matrix = build_matrix()
    jobs = []
    for seed in range(args.seeds):
        for cfg in matrix:
            jobs.append({**cfg, "seed": seed,
                         "run_id": run_id_from_config({**cfg, "seed": seed})})

    manifest = {
        "n_configs": len(matrix),
        "n_jobs": len(jobs),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "block_size": args.block_size,
        "tokens_per_run": args.max_steps * args.batch_size * args.grad_accum * args.block_size,
        "jobs": jobs,
    }
    (out_dir / "matrix_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{len(matrix)} unique configs x {args.seeds} seed(s) = {len(jobs)} runs")
    print(f"~{manifest['tokens_per_run'] / 1e6:.0f}M tokens per run")
    print(f"Manifest written to {out_dir / 'matrix_manifest.json'}\n")

    done = skipped = failed = 0
    for i, job in enumerate(jobs):
        run_id = job["run_id"]
        if i + 1 < args.start_at:
            print(f"[{i + 1}/{len(jobs)}] SKIP (before --start-at "
                  f"{args.start_at}): {run_id}")
            skipped += 1
            continue
        if (runs_dir / run_id / "summary.json").exists():
            print(f"[{i + 1}/{len(jobs)}] SKIP (done): {run_id}")
            skipped += 1
            continue
        cmd = [sys.executable, str(REPO_ROOT / "DataScripts" / "train_ablation.py"),
               "--optimizer", job["optimizer"],
               "--seed", str(job["seed"]),
               "--max-steps", str(args.max_steps),
               "--batch-size", str(args.batch_size),
               "--grad-accum", str(args.grad_accum),
               "--block-size", str(args.block_size),
               "--compile-mode", args.compile_mode,
               "--output-dir", str(runs_dir),
               "--run-id", run_id]
        if args.activation_checkpointing:
            cmd.append("--activation-checkpointing")
        for f in BOOL_FACTORS:
            if job[f]:
                cmd.append("--" + f.replace("_", "-"))

        print(f"[{i + 1}/{len(jobs)}] {'DRY ' if args.dry_run else ''}RUN "
              f"({job['group']}): {run_id}")
        if args.dry_run:
            continue
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if result.returncode == 0:
            done += 1
        else:
            failed += 1
            print(f"  !! run failed with exit code {result.returncode}, continuing")

    if not args.dry_run:
        print(f"\nFinished: {done} ran, {skipped} skipped, {failed} failed.")
        print("Next: python DataScripts/analyze_results.py")


if __name__ == "__main__":
    main()
