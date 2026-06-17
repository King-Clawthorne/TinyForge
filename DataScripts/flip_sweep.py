"""Find where the z-loss / softcap verdict flips.

The main factorial matrix (run_matrix.py) holds scale, schedule, and vocab
*fixed* at one operating point (12L/768d, 1000 steps, vocab 8192). At that
point z-loss and the tanh logit softcap come out inert-to-harmful — which is
expected, because both are *output-softmax stabilizers*: they fight log-sum-exp
blowup of the LM-head logits, and an 8k-vocab / short-schedule softmax has
little blowup to fight. PaLM (vocab 256k) and Gemma needed them; a 100M model
on TinyStories at vocab 8k does not.

This script asks the natural follow-up: along which axis, and at what value,
does the verdict flip from "harmful/inert" to "useful"? It replicates the
z-loss x softcap 2x2 cell across a small one-factor-at-a-time grid over the
three axes that drive logit magnitude:

  - vocab    : 8192 -> 16384 -> 32768 -> 65536   (the dominant lever)
  - schedule : max-steps 500 -> 1000 -> 2000 -> 4000
  - scale    : (n_layers, n_heads, n_embd) tiers

Everything else is held at a cheap, fixed base regime. For each grid point we
train the four cells {z0cap0, z1cap0, z0cap1, z1cap1}; flip_analyze.py then
reports the z-loss and softcap main effects per regime and marks the sign flip.

Runs land in DataOutput/flip_runs/ (kept separate from the main matrix so
analyze_results.py is unaffected). Resumable: a cell whose summary.json exists
is skipped.

The non-default vocab sizes each need their own pretokenized corpus. Either run
  python DataScripts/prepare_data.py --vocab-size 16384   (and 32768, 65536)
up front, or pass --auto-prepare to have this script invoke it for any missing
vocab (downloads are cached after the first time).

Usage (from repo root):
  python DataScripts/flip_sweep.py --dry-run
  python DataScripts/flip_sweep.py --axis vocab --auto-prepare
  python DataScripts/flip_sweep.py --axis all
  python DataScripts/flip_analyze.py
"""

import sys
import json
import argparse
import itertools
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# The base regime: deliberately small/short/narrow, i.e. the corner where the
# existing study finds z-loss/softcap inert. Each axis sweep moves ONE knob
# away from this base while holding the rest here.
BASE = {
    "vocab_size": 8192,
    "max_steps": 1000,
    "n_layers": 8,
    "n_heads": 8,
    "n_embd": 512,
    "lr_peak": 1e-3,
    "block_size": 1024,
    "batch_size": 8,
    "grad_accum": 4,
}

# One-factor-at-a-time grids. The base value of each axis is included so every
# sweep shares the base corner (and so the base 2x2 is computed once).
AXES = {
    "vocab":    {"vocab_size": [8192, 16384, 32768, 65536]},
    "schedule": {"max_steps":  [500, 1000, 2000, 4000]},
    # Scale tiers keep n_embd divisible by n_heads and head_dim sensible.
    "scale":    {"scale_tier": [
        (4, 4, 256),
        (8, 8, 512),
        (12, 12, 768),
        (16, 16, 1024),
    ]},
}

# The 2x2 stabilizer cell: (z_loss, softcap) with every other factor OFF and
# the optimizer fixed to AdamW (matching the zloss_x_cap block in run_matrix.py,
# QK-gain off). Holding everything else off isolates the two stabilizers.
STAB_CELLS = list(itertools.product([False, True], [False, True]))  # (z_loss, softcap)


def regime_tag(axis, value):
    """Compact, filesystem-safe label for one grid point."""
    if axis == "scale":
        nl, nh, ne = value
        return f"scale-{nl}x{nh}x{ne}"
    return f"{axis}-{value}"


def point_params(axis, value):
    """Resolve the full training config for one grid point (base + the swept knob)."""
    p = dict(BASE)
    if axis == "scale":
        p["n_layers"], p["n_heads"], p["n_embd"] = value
    elif axis == "vocab":
        p["vocab_size"] = value
    elif axis == "schedule":
        p["max_steps"] = value
    else:
        raise ValueError(f"unknown axis {axis}")
    return p


def build_jobs(axes, seeds):
    """One job per (regime point) x (z_loss, softcap) cell x seed.

    Each axis is self-contained: it carries its own anchor point (the base
    corner) under its own tag, so the schedule and scale sweeps still have a
    low end to flip from. The base corner therefore trains once per swept axis
    — a few cheap extra runs in exchange for a complete, independently
    orderable series per axis.
    """
    jobs = []
    for axis in axes:
        (knob, values), = AXES[axis].items()
        for value in values:
            params = point_params(axis, value)
            tag = regime_tag(axis, value)
            for z, cap in STAB_CELLS:
                for seed in range(seeds):
                    jobs.append({
                        "axis": axis,
                        "regime": tag,
                        "params": params,
                        "z_loss": z,
                        "softcap": cap,
                        "seed": seed,
                        "run_id": f"{tag}_z{int(z)}_cap{int(cap)}_seed{seed}",
                    })
    return jobs


def needed_vocabs(jobs):
    return sorted({j["params"]["vocab_size"] for j in jobs})


def main():
    parser = argparse.ArgumentParser(
        description="z-loss / softcap verdict-flip sweep")
    parser.add_argument("--axis", choices=["vocab", "schedule", "scale", "all"],
                        default="vocab",
                        help="Which axis to sweep (default: vocab, the dominant lever)")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--auto-prepare", action="store_true",
                        help="Run prepare_data.py for any missing vocab size")
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "DataOutput"))
    args = parser.parse_args()

    axes = ["vocab", "schedule", "scale"] if args.axis == "all" else [args.axis]
    jobs = build_jobs(axes, args.seeds)

    out_dir = Path(args.output_dir)
    runs_dir = out_dir / "flip_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    tokens_dir = out_dir / "tokens"

    n_points = len({j["regime"] for j in jobs})
    print(f"Sweeping axes {axes}: {n_points} regime points x "
          f"{len(STAB_CELLS)} cells x {args.seeds} seed(s) = {len(jobs)} runs\n")

    manifest = {
        "base": BASE,
        "axes": axes,
        "cells": [{"z_loss": z, "softcap": c} for z, c in STAB_CELLS],
        "seeds": args.seeds,
        "n_runs": len(jobs),
        "jobs": [{k: j[k] for k in ("axis", "regime", "params", "z_loss",
                                    "softcap", "seed", "run_id")} for j in jobs],
    }
    (out_dir / "flip_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Ensure pretokenized data exists for every vocab we'll touch.
    missing = [v for v in needed_vocabs(jobs)
               if not (tokens_dir / f"meta_{v}.json").exists()]
    if missing and not args.dry_run:
        if args.auto_prepare:
            for v in missing:
                print(f"Preparing pretokenized corpus for vocab {v}...")
                r = subprocess.run(
                    [sys.executable, str(REPO_ROOT / "DataScripts" / "prepare_data.py"),
                     "--vocab-size", str(v)],
                    cwd=str(REPO_ROOT))
                if r.returncode != 0:
                    sys.exit(f"prepare_data.py failed for vocab {v}")
        else:
            cmds = "\n".join(
                f"  python DataScripts/prepare_data.py --vocab-size {v}"
                for v in missing)
            sys.exit(
                f"Pretokenized data missing for vocab size(s): {missing}\n"
                f"Run these once (or pass --auto-prepare):\n{cmds}")

    done = skipped = failed = 0
    for i, job in enumerate(jobs):
        run_id = job["run_id"]
        if (runs_dir / run_id / "summary.json").exists():
            print(f"[{i + 1}/{len(jobs)}] SKIP (done): {run_id}")
            skipped += 1
            continue
        p = job["params"]
        cmd = [sys.executable, str(REPO_ROOT / "DataScripts" / "train_ablation.py"),
               "--optimizer", "adamw",
               "--seed", str(job["seed"]),
               "--vocab-size", str(p["vocab_size"]),
               "--max-steps", str(p["max_steps"]),
               "--n-layers", str(p["n_layers"]),
               "--n-heads", str(p["n_heads"]),
               "--n-embd", str(p["n_embd"]),
               "--lr-peak", str(p["lr_peak"]),
               "--block-size", str(p["block_size"]),
               "--batch-size", str(p["batch_size"]),
               "--grad-accum", str(p["grad_accum"]),
               "--compile-mode", args.compile_mode,
               "--output-dir", str(runs_dir),
               "--run-id", run_id]
        if job["z_loss"]:
            cmd.append("--z-loss")
        if job["softcap"]:
            cmd.append("--softcap")
        if args.activation_checkpointing:
            cmd.append("--activation-checkpointing")

        print(f"[{i + 1}/{len(jobs)}] {'DRY ' if args.dry_run else ''}RUN "
              f"({job['regime']}): {run_id}")
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
        print("Next: python DataScripts/flip_analyze.py")


if __name__ == "__main__":
    main()
