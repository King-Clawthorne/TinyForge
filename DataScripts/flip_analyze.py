"""Locate the z-loss / softcap verdict flip across the sweep.

Reads DataOutput/flip_runs/*/summary.json + config.json (written by
train_ablation.py via flip_sweep.py) and, for each regime point, computes the
z-loss and softcap main effects from the 2x2 (z_loss x softcap) cell:

  z-loss effect  = mean over softcap in {off,on} of  val(z=1) - val(z=0)
  softcap effect = mean over z-loss  in {off,on} of  val(cap=1) - val(cap=0)
  interaction    = [val(1,1) - val(0,1)] - [val(1,0) - val(0,0)]

Sign convention (val loss, ON minus OFF): negative = the component HELPS,
positive = it HURTS, ~0 = inert. The "flip" is where an effect crosses zero
from positive/inert toward negative as the swept knob grows.

Regimes are ordered along each axis (vocab, schedule, scale) and the first
point whose effect turns useful (<= --flip-threshold, default -0.002) is
flagged. Diverged cells drop their regime's effect for that factor.

Usage (from repo root): python DataScripts/flip_analyze.py
"""

import csv
import json
import argparse
import statistics
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_cells(runs_dir):
    """Return {regime: {(z_loss, softcap): record}} plus regime metadata."""
    regimes = defaultdict(dict)
    meta = {}
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        s = json.loads(summary_path.read_text())
        cfg_path = summary_path.parent / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        run_id = s["run_id"]
        # run_id == "<regime>_z{0,1}_cap{0,1}_seed{n}"
        base, _seed = run_id.rsplit("_seed", 1)
        regime, _, cappart = base.partition("_z")  # cappart like "0_cap1"
        zc, _, capc = cappart.partition("_cap")
        cell = (bool(int(zc)), bool(int(capc)))
        s["_cfg"] = cfg
        # Average over seeds within a cell.
        regimes[regime].setdefault(cell, []).append(s)
        meta.setdefault(regime, {
            "axis": _infer_axis(regime),
            "vocab_size": cfg.get("vocab_size"),
            "max_steps": cfg.get("max_steps"),
            "n_layers": cfg.get("n_layers"),
            "n_embd": cfg.get("n_embd"),
            "n_params": s.get("n_params"),
        })
    return regimes, meta


def _infer_axis(regime):
    return regime.split("-", 1)[0]


def _cell_mean(records):
    vals = [r["final_val_loss"] for r in records
            if not r.get("diverged") and r["final_val_loss"] == r["final_val_loss"]]
    return statistics.mean(vals) if vals else None


def effects_for_regime(cells):
    """Return (z_effect, cap_effect, interaction) or Nones if a cell is missing."""
    m = {}
    for cell, recs in cells.items():
        m[cell] = _cell_mean(recs)
    if any(m.get(c) is None for c in
           [(False, False), (True, False), (False, True), (True, True)]):
        return None, None, None
    z_eff = 0.5 * ((m[(True, False)] - m[(False, False)]) +
                   (m[(True, True)] - m[(False, True)]))
    cap_eff = 0.5 * ((m[(False, True)] - m[(False, False)]) +
                     (m[(True, True)] - m[(True, False)]))
    inter = ((m[(True, True)] - m[(False, True)]) -
             (m[(True, False)] - m[(False, False)]))
    return z_eff, cap_eff, inter


def sort_key(meta_entry):
    axis = meta_entry["axis"]
    if axis == "vocab":
        return meta_entry["vocab_size"] or 0
    if axis == "schedule":
        return meta_entry["max_steps"] or 0
    if axis == "scale":
        return meta_entry["n_params"] or meta_entry["n_embd"] or 0
    return 0


def verdict(eff, threshold):
    if eff is None:
        return "n/a"
    if eff <= threshold:
        return "useful"
    if eff >= -threshold:
        return "harmful"
    return "inert"


def main():
    parser = argparse.ArgumentParser(description="Find the z-loss/softcap flip")
    parser.add_argument("--runs-dir", default=str(REPO_ROOT / "DataOutput" / "flip_runs"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "DataOutput" / "analysis"))
    parser.add_argument("--flip-threshold", type=float, default=-0.002,
                        help="An effect <= this (val-loss units) counts as 'useful'")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    regimes, meta = load_cells(runs_dir)
    if not regimes:
        print(f"No flip runs found in {runs_dir}. Run flip_sweep.py first.")
        return

    rows = []
    for regime, cells in regimes.items():
        z_eff, cap_eff, inter = effects_for_regime(cells)
        rows.append({
            "regime": regime,
            "axis": meta[regime]["axis"],
            "vocab_size": meta[regime]["vocab_size"],
            "max_steps": meta[regime]["max_steps"],
            "n_params_M": (round(meta[regime]["n_params"] / 1e6, 1)
                           if meta[regime]["n_params"] else None),
            "z_loss_effect": z_eff,
            "softcap_effect": cap_eff,
            "interaction": inter,
            "z_loss_verdict": verdict(z_eff, args.flip_threshold),
            "softcap_verdict": verdict(cap_eff, args.flip_threshold),
            "_sort": sort_key(meta[regime]),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "flip_effects.csv"
    fields = ["axis", "regime", "vocab_size", "max_steps", "n_params_M",
              "z_loss_effect", "softcap_effect", "interaction",
              "z_loss_verdict", "softcap_verdict"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["axis"], r["_sort"])):
            w.writerow(r)

    # Console digest, grouped by axis and ordered along it, with flip points.
    print("z-loss / softcap verdict flip "
          f"(effect = Δ final val loss, ON−OFF; negative = useful; "
          f"flip threshold {args.flip_threshold:+})\n")
    by_axis = defaultdict(list)
    for r in rows:
        by_axis[r["axis"]].append(r)
    for axis in ("vocab", "schedule", "scale"):
        axis_rows = sorted(by_axis.get(axis, []), key=lambda r: r["_sort"])
        if not axis_rows:
            continue
        print(f"== axis: {axis} ==")
        print(f"{'regime':<18} {'z-loss':>9} {'softcap':>9} {'inter':>9}  verdicts")
        for r in axis_rows:
            ze = f"{r['z_loss_effect']:+.4f}" if r["z_loss_effect"] is not None else "   n/a"
            ce = f"{r['softcap_effect']:+.4f}" if r["softcap_effect"] is not None else "   n/a"
            it = f"{r['interaction']:+.4f}" if r["interaction"] is not None else "   n/a"
            print(f"{r['regime']:<18} {ze:>9} {ce:>9} {it:>9}  "
                  f"z={r['z_loss_verdict']}, cap={r['softcap_verdict']}")
        for label, key in (("z-loss", "z_loss_effect"),
                           ("softcap", "softcap_effect")):
            flip = next((r for r in axis_rows
                         if r[key] is not None and r[key] <= args.flip_threshold), None)
            if flip:
                print(f"  -> {label} flips to USEFUL at {flip['regime']} "
                      f"({flip[key]:+.4f})")
            else:
                print(f"  -> {label} never crosses useful in this sweep")
        print()

    print(f"Per-regime effects written to {csv_path}")


if __name__ == "__main__":
    main()
