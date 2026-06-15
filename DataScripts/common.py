"""Shared, dependency-free definitions for the ablation study.

Kept import-light on purpose: run_matrix.py needs run ids for --dry-run and
skip-detection on machines that don't have the training deps installed.
"""

FACTORS = ["optimizer", "qk_norm", "qk_gain", "layerscale",
           "value_residual", "z_loss", "softcap"]

BOOL_FACTORS = FACTORS[1:]


def run_id_from_config(cfg):
    parts = [f"opt-{cfg['optimizer']}"]
    short = {"qk_norm": "norm", "qk_gain": "gain", "layerscale": "ls",
             "value_residual": "vres", "z_loss": "zloss", "softcap": "cap"}
    for f, s in short.items():
        parts.append(f"{s}{int(bool(cfg[f]))}")
    parts.append(f"seed{cfg['seed']}")
    return "_".join(parts)
