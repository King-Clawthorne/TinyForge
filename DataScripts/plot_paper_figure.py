"""Publication figure for the paper: seed-averaged validation-loss curves for
the two anchors (AdamW baseline, full Muon recipe) and the leave-one-out runs
(full recipe minus one component). One line per configuration instead of one
per run, legend outside the axes, y-axis zoomed to where the curves separate.

Writes DataOutput/analysis/loss_curves_paper.png.  Run from repo root.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "DataOutput" / "runs"
OUT = REPO_ROOT / "DataOutput" / "analysis" / "loss_curves_paper.png"

FACTORS = ["optimizer", "qk_norm", "qk_gain", "layerscale",
           "value_residual", "z_loss", "softcap"]
LABELS = {"optimizer": "Muon", "qk_norm": "QK-norm", "qk_gain": "QK-gain",
          "layerscale": "LayerScale", "value_residual": "value-residual",
          "z_loss": "z-loss", "softcap": "softcap"}


def on(summary, factor):
    v = summary[factor]
    return v == "muon" if factor == "optimizer" else bool(v)


def load():
    """Return {run_id: (summary, curve)} for every completed run."""
    runs = {}
    for sp in sorted(RUNS_DIR.glob("*/summary.json")):
        s = json.loads(sp.read_text())
        curve = []
        lp = sp.parent / "log.jsonl"
        if lp.exists():
            for line in lp.read_text().splitlines():
                if line.strip():
                    curve.append(json.loads(line))
        runs[s["run_id"]] = (s, curve)
    return runs


def classify(s):
    """Return a label and z-order/style class for the configs we want to show:
    baseline, full recipe, or full-recipe-minus-one. Otherwise None."""
    n_on = sum(on(s, f) for f in FACTORS)
    if n_on == 0:
        return "AdamW baseline (all off)", "anchor"
    if n_on == len(FACTORS):
        return "Full Muon recipe (all on)", "anchor"
    if n_on == len(FACTORS) - 1:
        dropped = next(f for f in FACTORS if not on(s, f))
        return f"Full recipe minus {LABELS[dropped]}", "loo"
    return None, None


def seed_average(curves):
    """Per-step mean and min/max envelope of val_loss across seeds."""
    by_step = defaultdict(list)
    for curve in curves:
        for p in curve:
            by_step[p["step"]].append(p["val_loss"])
    steps = sorted(by_step)
    mean = [statistics.mean(by_step[s]) for s in steps]
    lo = [min(by_step[s]) for s in steps]
    hi = [max(by_step[s]) for s in steps]
    return steps, mean, lo, hi


def main():
    runs = load()
    # Group runs (over seeds) by their display label.
    groups = defaultdict(lambda: {"curves": [], "cls": None, "final": []})
    for s, curve in runs.values():
        label, cls = classify(s)
        if label is None or not curve:
            continue
        groups[label]["curves"].append(curve)
        groups[label]["cls"] = cls
        groups[label]["final"].append(s["final_val_loss"])

    # Order by final loss (best first) so leave-one-out colors track quality.
    ordered = sorted(groups.items(),
                     key=lambda kv: statistics.mean(kv[1]["final"]))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.get_cmap("turbo")
    loo_items = [kv for kv in ordered if kv[1]["cls"] == "loo"]
    color_of = {lbl: cmap(i / max(1, len(loo_items) - 1))
                for i, (lbl, _) in enumerate(loo_items)}

    # Legend reads top-to-bottom: the two reference anchors first, then the
    # leave-one-out runs ordered best to worst. Anchors lead because every
    # leave-one-out is "the full recipe minus one component" relative to them.
    anchors = [kv for kv in ordered if kv[1]["cls"] == "anchor"]
    anchors.sort(key=lambda kv: 0 if "baseline" in kv[0] else 1)
    plot_order = anchors + loo_items

    for label, g in plot_order:
        steps, vals, lo, hi = seed_average(g["curves"])
        if g["cls"] == "anchor":
            is_full = "Full" in label
            color = "#0b6e4f" if is_full else "black"
            zorder = 5
        else:
            color = color_of[label]
            zorder = 3
        # Dashed lines so the shaded min--max seed band reads through them.
        ax.fill_between(steps, lo, hi, color=color, alpha=0.25,
                        linewidth=0, zorder=zorder - 1)
        ax.plot(steps, vals, label=label, linewidth=1.8, color=color,
                linestyle="--", zorder=zorder)

    ax.set_xlabel("step")
    ax.set_ylabel("validation loss (seed mean)")
    ax.set_title("Baseline vs. full recipe vs. leave-one-out")
    ax.set_xlim(0, 1000)
    ax.set_ylim(1.6, 2.6)  # zoom to where the curves separate
    ax.grid(True, which="both", alpha=0.25)
    leg = ax.legend(title="Configuration (one line per config, seed mean)",
                    fontsize=9.5, title_fontsize=10,
                    loc="center left", bbox_to_anchor=(1.01, 0.5),
                    labelspacing=0.7, handlelength=2.6, borderpad=0.8,
                    frameon=True, framealpha=0.9, edgecolor="0.8")
    leg.get_title().set_multialignment("left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
