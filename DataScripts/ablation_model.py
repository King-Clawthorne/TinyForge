"""Configurable transformer for the factorial ablation study.

Same architecture family as SimpleTransformerLM in simple.py, but every
recipe component under study is an independent toggle:

  - qk_norm        : per-head RMSNorm on q/k before RoPE (Gemma 2 style)
  - qk_gain        : learnable per-head multiplicative gain on attention scores
  - layerscale     : learnable per-channel residual gates (CaiT)
  - value_residual : mix each layer's value with layer 0's value via a
                     learnable per-layer lambda (ResFormer, arXiv:2410.17897)

z-loss and logit softcap are loss-side toggles and live in train_ablation.py
(LigerFusedLinearCrossEntropyLoss arguments), not here. The optimizer toggle
(Muon vs AdamW) also lives in train_ablation.py.

Only the training/eval path is implemented (plain causal SDPA, returns
pre-projection hidden states for the fused loss). No KV cache / generation —
the study only measures loss curves.
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.layers import RMSNorm, RotaryEmbedding, QKNorm, apply_rope


class AblationBlock(nn.Module):
    def __init__(self, n_embd, n_heads, n_layers, layer_idx, *,
                 qk_norm, qk_gain, layerscale, value_residual, ls_init=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.head_dim = n_embd // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.layer_idx = layer_idx
        self.use_value_residual = value_residual

        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

        self.q_norm = QKNorm(n_heads, self.head_dim) if qk_norm else None
        self.k_norm = QKNorm(n_heads, self.head_dim) if qk_norm else None
        self.qk_gain = nn.Parameter(torch.ones(n_heads)) if qk_gain else None

        # ResFormer-style learnable value residual: v <- lam*v + (1-lam)*v1.
        # Layer 0 supplies v1 and never mixes.
        if value_residual and layer_idx > 0:
            self.v_lambda = nn.Parameter(torch.tensor(0.5))
        else:
            self.v_lambda = None

        hidden_dim = int(8 * n_embd / 3)
        hidden_dim = ((hidden_dim + 127) // 128) * 128
        self.w_up = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w_gate = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, n_embd, bias=False)

        if layerscale:
            self.ls_attn = nn.Parameter(torch.full((n_embd,), ls_init))
            self.ls_mlp = nn.Parameter(torch.full((n_embd,), ls_init))
        else:
            self.ls_attn = None
            self.ls_mlp = None

        for lin in (self.qkv, self.proj, self.w_up, self.w_gate, self.w_down):
            nn.init.normal_(lin.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
        nn.init.normal_(self.w_down.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def forward(self, x, cos, sin, v1):
        bsz, seq_len, _ = x.shape

        residual = x
        x_norm = self.ln1(x)
        q, k, v = self.qkv(x_norm).chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.use_value_residual:
            if v1 is None:
                v1 = v
            elif self.v_lambda is not None:
                lam = self.v_lambda.to(v.dtype)
                v = lam * v + (1.0 - lam) * v1

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = apply_rope(q, k, cos, sin)

        if self.qk_gain is not None:
            q = q * self.qk_gain.view(1, self.n_heads, 1, 1).to(q.dtype)

        # Plain causal attention so SDPA dispatches to its fused flash / cuDNN
        # backends (no document masking).
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=self.scale)
        y = y.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, self.n_embd)
        y = self.proj(y)
        x = residual + (self.ls_attn * y if self.ls_attn is not None else y)

        residual = x
        x_norm = self.ln2(x)
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction
        mlp_out = self.w_down(
            LigerSiLUMulFunction.apply(self.w_gate(x_norm), self.w_up(x_norm))
        )
        x = residual + (self.ls_mlp * mlp_out if self.ls_mlp is not None else mlp_out)

        return x, v1


class AblationTransformerLM(nn.Module):
    def __init__(self, vocab_size, *, block_size=2048, n_layers=12, n_heads=12,
                 n_embd=768, qk_norm=False, qk_gain=False, layerscale=False,
                 value_residual=False, activation_checkpointing=False):
        super().__init__()
        if n_embd % n_heads != 0:
            raise ValueError("n_embd must be divisible by n_heads")
        self.block_size = block_size
        self.head_dim = n_embd // n_heads
        # Recompute each block's activations in backward instead of storing
        # them — trades ~25-30% step time for a large activation-memory cut.
        # A memory knob only; it does not change the computed gradients, so it
        # is safe to flip per-machine without affecting the study's results.
        self.activation_checkpointing = activation_checkpointing

        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([
            AblationBlock(
                n_embd, n_heads, n_layers, i,
                qk_norm=qk_norm, qk_gain=qk_gain,
                layerscale=layerscale, value_residual=value_residual,
            )
            for i in range(n_layers)
        ])
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=block_size)
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        bsz, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError("Sequence length exceeds block_size")
        x = self.token_emb(idx)
        cos, sin = self.rope(seq_len)
        v1 = None
        checkpointing = (self.activation_checkpointing and self.training
                         and torch.is_grad_enabled())
        for block in self.blocks:
            if checkpointing:
                x, v1 = torch.utils.checkpoint.checkpoint(
                    block, x, cos, sin, v1, use_reentrant=False)
            else:
                x, v1 = block(x, cos, sin, v1)
        # Pre-projection hidden states; the caller applies the fused
        # lm_head + cross-entropy loss (see train_ablation.py).
        return self.ln_f(x)
