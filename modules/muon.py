"""
Muon — Momentum + Newton-Schulz orthogonalization optimizer.

Based on: Kostrubiec et al. (2024), "Modular Adaptive Optimization"
with the Moonshot/Kimi `adjust_lr_fn="match_rms_adamw"` LR-scaling recipe.

Only intended for 2-D weight matrices (linear layers, attention projections).
Use AdamW for embeddings, norms, biases, and 1-D/scalar parameters.
"""

import torch
from torch.optim import Optimizer


def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Orthogonalize G via 5-step Newton-Schulz quintic iteration.

    Returns a matrix with the same shape as G whose columns (or rows, whichever
    is the smaller dimension) are approximately orthonormal. The iteration is
    numerically stable in BF16 and converges in ~5 steps for typical gradient
    matrices.
    """
    assert G.ndim == 2, "Newton-Schulz expects a 2-D tensor"
    a, b, c = 3.4445, -4.7750, 2.0315

    X = G.to(torch.bfloat16)
    norm = X.norm()
    X = X / (norm + 1e-7)

    # Work on the thin dimension: if G is tall (rows > cols) transpose so
    # the smaller dimension is the column count — cheaper matmuls.
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(G.dtype)


def _match_rms_adamw_scale(update: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """
    Scale `update` so its RMS matches an AdamW-style update at the same LR.

    AdamW normalises per-element (RMS ≈ 1 after the m/v ratio).  Muon's
    orthogonal update has RMS = 1/sqrt(min(rows, cols)).  Multiplying by
    sqrt(max(rows, cols)) brings the Muon update's RMS to ~1, matching AdamW.
    """
    rows, cols = update.shape
    return update * (max(rows, cols) ** 0.5)


class Muon(Optimizer):
    """
    Muon optimizer — SGD with Nesterov momentum and Newton-Schulz gradient
    orthogonalization.

    Args:
        params:          Iterable of 2-D parameter tensors (matrices only).
        lr:              Learning rate. Default: 1e-3.
        momentum:        SGD momentum coefficient. Default: 0.95.
        nesterov:        Use Nesterov momentum. Default: True.
        weight_decay:    L2 penalty applied to parameters (decoupled). Default: 0.
        ns_steps:        Newton-Schulz iteration count. Default: 5.
        adjust_lr_fn:    "match_rms_adamw" rescales the update RMS to match
                         AdamW's effective step size, enabling the same LR for
                         both optimizers. Pass None to disable. Default: None.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = None,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if adjust_lr_fn not in (None, "match_rms_adamw"):
            raise ValueError(f"Unknown adjust_lr_fn: {adjust_lr_fn!r}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr           = group["lr"]
            momentum     = group["momentum"]
            nesterov     = group["nesterov"]
            wd           = group["weight_decay"]
            ns_steps     = group["ns_steps"]
            adjust_lr_fn = group["adjust_lr_fn"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.ndim != 2:
                    raise RuntimeError(
                        f"Muon expects 2-D parameters, got shape {tuple(grad.shape)}. "
                        "Route 1-D / scalar parameters to AdamW."
                    )

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]

                # SGD momentum update
                buf.mul_(momentum).add_(grad)

                if nesterov:
                    effective_grad = grad + momentum * buf
                else:
                    effective_grad = buf

                # Orthogonalize via Newton-Schulz
                update = _zeropower_via_newtonschulz5(effective_grad, steps=ns_steps)

                # Optional RMS rescaling to match AdamW's effective step magnitude
                if adjust_lr_fn == "match_rms_adamw":
                    update = _match_rms_adamw_scale(update, effective_grad)

                # Decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.add_(update, alpha=-lr)

        return loss
