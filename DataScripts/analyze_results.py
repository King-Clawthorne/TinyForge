"""Aggregate ablation results into paper-ready tables and figures.

Reads DataOutput/runs/*/summary.json and log.jsonl, writes to
DataOutput/analysis/:

  results.csv          - one row per run (all factors + losses)
  main_effects.csv     - per-factor effect on final val loss, computed over
                         matched pairs (configs identical except that factor)
  interactions.csv     - 2x2 cell means + interaction terms for the targeted
                         pairs (Muon x QK-norm, LayerScale x value-residual,
                         z-loss x softcap, z-loss x QK-gain, softcap x QK-gain)
  summary.md           - human-readable digest of all of the above
  loss_curves.png      - val-loss curves for all runs (needs matplotlib)
  loss_curves_anchors.png - baseline vs full recipe vs leave-one-out

Usage (from repo root): python DataScripts/analyze_results.py
"""

import csv
import json
import argparse
import statistics
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent

FACTORS = ["optimizer", "qk_norm", "qk_gain", "layerscale",
           "value_residual", "z_loss", "softcap"]
TARGET_PAIRS = [
    ("optimizer", "qk_norm"),
    ("layerscale", "value_residual"),
    ("z_loss", "softcap"),
    ("z_loss", "qk_gain"),
    ("softcap", "qk_gain"),
]


def factor_on(summary, factor):
    v = summary[factor]
    return v == "muon" if factor == "optimizer" else bool(v)


def config_key(summary, exclude=()):
    return tuple(
        (f, factor_on(summary, f)) for f in FACTORS if f not in exclude
    ) + (("seed", summary["seed"]),)


def load_runs(runs_dir):
    runs = []
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        s = json.loads(summary_path.read_text())
        curve = []
        log_path = summary_path.parent / "log.jsonl"
        if log_path.exists():
            for line in log_path.read_text().splitlines():
                if line.strip():
                    curve.append(json.loads(line))
        s["curve"] = curve
        runs.append(s)
    return runs


def write_results_csv(runs, path):
    fields = ["run_id", *FACTORS, "seed", "final_val_loss", "best_val_loss",
              "diverged", "tokens_seen", "wall_time_s", "n_params"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in runs:
            w.writerow({k: r.get(k) for k in fields})


def main_effects(runs):
    """For each factor: mean Δ(final val loss) over matched on/off pairs."""
    effects = []
    for factor in FACTORS:
        groups = defaultdict(dict)
        for r in runs:
            if r.get("diverged"):
                continue
            groups[config_key(r, exclude=(factor,))][factor_on(r, factor)] = r
        deltas = []
        for pair in groups.values():
            if True in pair and False in pair:
                deltas.append(pair[True]["final_val_loss"]
                              - pair[False]["final_val_loss"])
        if deltas:
            effects.append({
                "factor": factor,
                "n_pairs": len(deltas),
                "mean_delta": statistics.mean(deltas),
                "std_delta": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "min_delta": min(deltas),
                "max_delta": max(deltas),
            })
    return effects


def interactions(runs):
    """2x2 cell means and interaction terms over matched quadruples."""
    rows = []
    for fa, fb in TARGET_PAIRS:
        cells = defaultdict(lambda: defaultdict(list))
        for r in runs:
            if r.get("diverged"):
                continue
            key = config_key(r, exclude=(fa, fb))
            cells[key][(factor_on(r, fa), factor_on(r, fb))].append(
                r["final_val_loss"])
        terms = []
        cell_means = defaultdict(list)
        for quad in cells.values():
            if len(quad) == 4:
                m = {c: statistics.mean(v) for c, v in quad.items()}
                # interaction = (effect of A with B on) - (effect of A with B off)
                terms.append((m[(True, True)] - m[(False, True)])
                             - (m[(True, False)] - m[(False, False)]))
                for c, v in m.items():
                    cell_means[c].append(v)
        if terms:
            row = {"factor_a": fa, "factor_b": fb, "n_quads": len(terms),
                   "interaction": statistics.mean(terms)}
            for (a, b), vals in sorted(cell_means.items()):
                row[f"mean_a{int(a)}_b{int(b)}"] = statistics.mean(vals)
            rows.append(row)
    return rows


def write_dict_csv(rows, path):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "factor", k))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_curves(runs, analysis_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for r in runs:
        if not r["curve"]:
            continue
        steps = [p["step"] for p in r["curve"]]
        vals = [p["val_loss"] for p in r["curve"]]
        ax.plot(steps, vals, alpha=0.5, linewidth=1, label=r["run_id"])
    ax.set_xlabel("step")
    ax.set_ylabel("val loss")
    ax.set_title("Validation loss, all runs")
    if len(runs) <= 12:
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(analysis_dir / "loss_curves.png", dpi=150)
    plt.close(fig)

    # Anchors + leave-one-out: the headline figure.
    def is_anchor_or_loo(r):
        on = sum(factor_on(r, f) for f in FACTORS)
        return on in (0, len(FACTORS), len(FACTORS) - 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    for r in runs:
        if not r["curve"] or not is_anchor_or_loo(r):
            continue
        on = sum(factor_on(r, f) for f in FACTORS)
        style = {"linewidth": 2.5} if on in (0, len(FACTORS)) else \
                {"linewidth": 1, "alpha": 0.7}
        steps = [p["step"] for p in r["curve"]]
        vals = [p["val_loss"] for p in r["curve"]]
        ax.plot(steps, vals, label=r["run_id"], **style)
    ax.set_xlabel("step")
    ax.set_ylabel("val loss")
    ax.set_title("Baseline vs full recipe vs leave-one-out")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(analysis_dir / "loss_curves_anchors.png", dpi=150)
    plt.close(fig)
    print(f"Plots written to {analysis_dir}")


def write_summary_md(runs, effects, inter_rows, path):
    lines = ["# Ablation study results\n"]
    n_div = sum(1 for r in runs if r.get("diverged"))
    lines.append(f"{len(runs)} runs total, {n_div} diverged.\n")

    finished = [r for r in runs if not r.get("diverged")]
    if finished:
        ranked = sorted(finished, key=lambda r: r["final_val_loss"])
        lines.append("## Runs ranked by final val loss\n")
        lines.append("| rank | run | final val loss | best val loss |")
        lines.append("|---|---|---|---|")
        for i, r in enumerate(ranked, 1):
            lines.append(f"| {i} | {r['run_id']} | "
                         f"{r['final_val_loss']:.4f} | {r['best_val_loss']:.4f} |")
        lines.append("")

    if effects:
        lines.append("## Main effects (Δ final val loss, ON minus OFF; "
                     "negative = component helps)\n")
        lines.append("| factor | mean Δ | std | n pairs |")
        lines.append("|---|---|---|---|")
        for e in sorted(effects, key=lambda e: e["mean_delta"]):
            lines.append(f"| {e['factor']} | {e['mean_delta']:+.4f} | "
                         f"{e['std_delta']:.4f} | {e['n_pairs']} |")
        lines.append("")

    if inter_rows:
        lines.append("## Interactions (positive = redundant/antagonistic, "
                     "negative = synergistic)\n")
        lines.append("| A | B | interaction | n quads |")
        lines.append("|---|---|---|---|")
        for r in inter_rows:
            lines.append(f"| {r['factor_a']} | {r['factor_b']} | "
                         f"{r['interaction']:+.4f} | {r['n_quads']} |")
        lines.append("")

    if n_div:
        lines.append("## Diverged runs\n")
        for r in runs:
            if r.get("diverged"):
                lines.append(f"- {r['run_id']}")
        lines.append("")

    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation results")
    parser.add_argument("--runs-dir", default=str(REPO_ROOT / "DataOutput" / "runs"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "DataOutput" / "analysis"))
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    analysis_dir = Path(args.out_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(runs_dir)
    if not runs:
        print(f"No completed runs found in {runs_dir} "
              "(expected <run_id>/summary.json). Run run_matrix.py first.")
        return

    write_results_csv(runs, analysis_dir / "results.csv")
    effects = main_effects(runs)
    write_dict_csv(effects, analysis_dir / "main_effects.csv")
    inter_rows = interactions(runs)
    write_dict_csv(inter_rows, analysis_dir / "interactions.csv")
    write_summary_md(runs, effects, inter_rows, analysis_dir / "summary.md")
    plot_curves(runs, analysis_dir)

    print(f"Analyzed {len(runs)} runs.")
    print(f"Tables and summary written to {analysis_dir}")


if __name__ == "__main__":
    main()
