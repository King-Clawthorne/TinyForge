"""Single ablation training run for the factorial study.

Trains one configuration of AblationTransformerLM on the pretokenized local
corpus (run DataScripts/prepare_data.py once first) and writes
machine-readable results to DataOutput/runs/<run_id>/:

  config.json   - full resolved configuration of the run
  log.jsonl     - one JSON object per eval point (step, losses, lr, tokens, time)
  summary.json  - final results (final/best val loss, params, wall time)

Toggles (all default OFF; the full modern recipe is everything ON + --optimizer muon):
  --optimizer {adamw,muon}  Muon on 2D body matrices + AdamW elsewhere, vs pure AdamW
  --qk-norm                 per-head RMSNorm on q/k
  --qk-gain                 learnable per-head attention-score gain
  --layerscale              LayerScale residual gates
  --value-residual          ResFormer value residual
  --z-loss                  lse_square_scale=1e-4 in the fused loss
  --softcap                 tanh logit softcap (30.0) in the fused loss

Run from the repo root, e.g.:
  python DataScripts/train_ablation.py --optimizer muon --qk-norm --z-loss
"""

import os
import sys
import json
import time
import math
import random
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from liger_kernel.transformers.fused_linear_cross_entropy import (
    LigerFusedLinearCrossEntropyLoss,
)

from modules.muon import Muon
from DataScripts.ablation_model import AblationTransformerLM
from DataScripts.common import FACTORS, run_id_from_config


class BinTokenDataset(Dataset):
    """Non-overlapping (x, y) blocks from a memory-mapped uint16 token file
    written by prepare_data.py. No tokenization, no network: every run reads
    identical batches in identical order."""

    def __init__(self, path, block_size):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        return (len(self.tokens) - 1) // self.block_size

    def __getitem__(self, i):
        a = np.asarray(
            self.tokens[i * self.block_size : (i + 1) * self.block_size + 1],
            dtype=np.int64)
        return torch.from_numpy(a[:-1]), torch.from_numpy(a[1:])


def build_optimizers(model, optimizer_kind, lr_peak):
    """Muon hybrid (matching simple.py) or pure AdamW with the same wd policy."""
    is_scale = lambda n: ("q_norm" in n or "k_norm" in n)
    body, embed, no_wd = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "token_emb" not in name and "lm_head" not in name and not is_scale(name):
            body.append(p)
        elif p.ndim >= 2 and not is_scale(name):
            embed.append(p)
        else:
            no_wd.append(p)

    fused = torch.cuda.is_available()
    if optimizer_kind == "muon":
        muon_opt = Muon(body, lr=lr_peak, momentum=0.95, nesterov=True,
                        weight_decay=0.1, adjust_lr_fn="match_rms_adamw")
        adamw_opt = torch.optim.AdamW(
            [{"params": embed, "weight_decay": 0.1},
             {"params": no_wd, "weight_decay": 0.0}],
            lr=lr_peak, betas=(0.9, 0.99), fused=fused)
        return [muon_opt, adamw_opt]
    adamw_opt = torch.optim.AdamW(
        [{"params": body + embed, "weight_decay": 0.1},
         {"params": no_wd, "weight_decay": 0.0}],
        lr=lr_peak, betas=(0.9, 0.99), fused=fused)
    return [adamw_opt]


def make_loss(z_loss, softcap):
    return LigerFusedLinearCrossEntropyLoss(
        lse_square_scale=1e-4 if z_loss else 0.0,
        softcap=30.0 if softcap else None,
    )


def estimate_loss(model, loss_fn, val_loader, device, eval_iters):
    model.eval()
    losses = []
    autocast_ctx = torch.amp.autocast(device, dtype=torch.bfloat16,
                                      enabled=(device == "cuda"))
    with torch.no_grad():
        for i, (xb, yb) in enumerate(val_loader):
            if i >= eval_iters:
                break
            torch.compiler.cudagraph_mark_step_begin()
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with autocast_ctx:
                hidden = model(xb)
                loss = loss_fn(model.lm_head.weight,
                               hidden.reshape(-1, hidden.size(-1)),
                               yb.reshape(-1))
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def main():
    parser = argparse.ArgumentParser(description="Factorial ablation run")
    # Factors
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--qk-norm", action="store_true")
    parser.add_argument("--qk-gain", action="store_true")
    parser.add_argument("--layerscale", action="store_true")
    parser.add_argument("--value-residual", action="store_true")
    parser.add_argument("--z-loss", action="store_true")
    parser.add_argument("--softcap", action="store_true")
    # Budget / hardware
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--lr-peak", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--final-eval-iters", type=int, default=100)
    parser.add_argument("--compile-mode", default="default",
                        choices=["none", "default", "reduce-overhead",
                                 "max-autotune", "max-autotune-no-cudagraphs"])
    parser.add_argument("--activation-checkpointing", action="store_true",
                        help="Recompute block activations in backward (memory "
                             "knob; ~25-30%% slower, results unchanged)")
    # Model size (held fixed across the matrix)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--n-heads", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "DataOutput" / "tokens"),
                        help="Directory of pretokenized .bin/meta files "
                             "written by prepare_data.py")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "DataOutput" / "runs"))
    parser.add_argument("--run-id", default=None,
                        help="Override the auto-generated run directory name")
    args = parser.parse_args()

    cfg = vars(args).copy()
    run_id = args.run_id or run_id_from_config(cfg)
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Pretokenized local data only (see prepare_data.py): no network, no
    # tokenization, and identical batches across every run in the matrix.
    data_dir = Path(args.data_dir)
    meta_path = data_dir / f"meta_{args.vocab_size}.json"
    if not meta_path.exists():
        sys.exit(f"Pretokenized data not found at {meta_path}.\n"
                 f"Run once first:  python DataScripts/prepare_data.py "
                 f"--vocab-size {args.vocab_size}")
    meta = json.loads(meta_path.read_text())

    train_dataset = BinTokenDataset(data_dir / meta["train_file"], args.block_size)
    val_dataset = BinTokenDataset(data_dir / meta["val_file"], args.block_size)

    # memmap reads are cheap: no worker processes needed (also sidesteps
    # multiprocessing teardown flakiness entirely).
    loader_kwargs = dict(num_workers=0, pin_memory=True, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, **loader_kwargs)

    padded_vocab_size = ((meta["vocab_size"] + 63) // 64) * 64

    model = AblationTransformerLM(
        padded_vocab_size,
        block_size=args.block_size, n_layers=args.n_layers,
        n_heads=args.n_heads, n_embd=args.n_embd,
        qk_norm=args.qk_norm, qk_gain=args.qk_gain,
        layerscale=args.layerscale, value_residual=args.value_residual,
        activation_checkpointing=args.activation_checkpointing,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{run_id}] parameters: {n_params / 1e6:.1f}M on {device}")

    if args.compile_mode != "none":
        from torch._inductor import config as inductor_config, select_algorithm
        select_algorithm.PRINT_AUTOTUNE = False
        inductor_config.max_autotune_report_choices_stats = False
        # checkpoint() introduces graph breaks, so fullgraph must be off then.
        model = torch.compile(model, mode=args.compile_mode,
                              fullgraph=not args.activation_checkpointing)

    optimizers = build_optimizers(model, args.optimizer, args.lr_peak)
    loss_fn = make_loss(args.z_loss, args.softcap)

    warmup_steps = min(99, max(0, args.max_steps - 1))

    def get_lr(step):
        if step < warmup_steps:
            return args.lr_peak * step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, args.max_steps - warmup_steps)
        return 0.1 * args.lr_peak + 0.45 * args.lr_peak * (1 + math.cos(math.pi * progress))

    (run_dir / "config.json").write_text(json.dumps(
        {**cfg, "run_id": run_id, "n_params": n_params,
         "dataset": meta["dataset"], "device": device,
         "tokens_per_step": args.batch_size * args.grad_accum * args.block_size},
        indent=2))
    log_path = run_dir / "log.jsonl"
    log_f = open(log_path, "w")

    autocast_ctx = torch.amp.autocast(device, dtype=torch.bfloat16,
                                      enabled=(device == "cuda"))
    train_iter = iter(train_loader)

    def next_batch():
        nonlocal train_iter
        try:
            return next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            return next(train_iter)

    best_val = float("inf")
    tokens_seen = 0
    t_start = time.time()
    diverged = False

    for step in range(args.max_steps):
        torch.compiler.cudagraph_mark_step_begin()
        lr = get_lr(step)
        for opt in optimizers:
            for g in opt.param_groups:
                g["lr"] = lr
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        loss_acc = 0.0
        for _ in range(args.grad_accum):
            xb, yb = next_batch()
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with autocast_ctx:
                hidden = model(xb)
                micro_loss = loss_fn(model.lm_head.weight,
                                     hidden.reshape(-1, hidden.size(-1)),
                                     yb.reshape(-1)) / args.grad_accum
            micro_loss.backward()
            loss_acc += micro_loss.detach()
            tokens_seen += xb.numel()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in optimizers:
            opt.step()

        train_loss = loss_acc.item()
        if not math.isfinite(train_loss):
            diverged = True
            print(f"[{run_id}] step {step}: non-finite train loss, stopping")

        if step % args.eval_interval == 0 or step == args.max_steps - 1 or diverged:
            val_loss = estimate_loss(model, loss_fn, val_loader, device,
                                     args.eval_iters)
            best_val = min(best_val, val_loss)
            rec = {"step": step, "lr": lr, "train_loss": train_loss,
                   "val_loss": val_loss, "grad_norm": float(grad_norm),
                   "tokens": tokens_seen, "wall_time_s": time.time() - t_start}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            print(f"[{run_id}] step {step:04d} | lr {lr:.2e} | "
                  f"train {train_loss:.4f} | val {val_loss:.4f}")

        if diverged:
            break

    final_val = (float("nan") if diverged else
                 estimate_loss(model, loss_fn, val_loader, device,
                               args.final_eval_iters))
    log_f.close()

    summary = {
        "run_id": run_id,
        **{f: cfg[f] for f in FACTORS},
        "seed": args.seed,
        "n_params": n_params,
        "max_steps": args.max_steps,
        "tokens_seen": tokens_seen,
        "final_val_loss": final_val,
        "best_val_loss": best_val,
        "diverged": diverged,
        "wall_time_s": time.time() - t_start,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{run_id}] done: final val {final_val:.4f} | best {best_val:.4f} | "
          f"{summary['wall_time_s'] / 60:.1f} min")

    # Persistent DataLoader workers + CUDA threads can abort during normal
    # interpreter teardown (PyGILState_Release fatal error), turning a
    # successful run into a nonzero exit code. All results are on disk at
    # this point, so shut the workers down explicitly and skip the rest of
    # teardown. (Without the explicit shutdown, os._exit kills the workers
    # mid-handshake and the resource_sharer thread prints harmless
    # ConnectionResetError tracebacks.)
    del train_iter
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
