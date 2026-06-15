import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.tensor  # ensure DTensor is loaded before liger-kernel checks for it
from torch.nn.attention import sdpa_kernel, SDPBackend
from typing import Optional, Tuple

from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

from liger_kernel.ops.rope import LigerRopeFunction

class RMSNorm(LigerRMSNorm):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__(hidden_size=dim, eps=eps)

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) with dynamic NTK-aware scaling.

    Applies rotary positional encoding to queries and keys for relative position
    awareness. When a requested sequence length exceeds the length the model was
    trained on (`original_max_seq_len`), the RoPE base is increased via NTK-aware
    interpolation so high-frequency components are stretched smoothly rather than
    extrapolated off the end — extending usable context with no retraining.
    References: RoPE https://arxiv.org/abs/2104.09864 ;
    NTK-aware scaling https://arxiv.org/abs/2309.00071
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        # The context length the model is trained on; NTK scaling only kicks in
        # for requested lengths beyond this.
        self.original_max_seq_len = max_seq_len
        self.max_seq_len = max_seq_len

        self.register_buffer('inv_freq', self._compute_inv_freq(base), persistent=False)

        # Precompute cos/sin cache
        self._build_cache(max_seq_len)

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        return 1.0 / (base ** (torch.arange(0, self.dim, 2).float() / self.dim))

    def _build_cache(self, seq_len: int):
        """Build cos/sin cache for given sequence length, applying NTK-aware base
        scaling when seq_len exceeds the trained context window."""
        if seq_len > self.original_max_seq_len:
            # NTK-aware: rescale the base so the rotary frequencies are
            # interpolated to cover the longer context.
            scale = seq_len / self.original_max_seq_len
            base = self.base * (scale ** (self.dim / (self.dim - 2)))
            inv_freq = self._compute_inv_freq(base).to(self.inv_freq.device)
        else:
            inv_freq = self._compute_inv_freq(self.base).to(self.inv_freq.device)
        self.inv_freq = inv_freq

        t = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        # [seq_len, dim/2] -> [1, seq_len, dim]  (LigerRopeFunction expects [1, seq, dim])
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, :, :], persistent=False)
        self.max_seq_len = seq_len

    def forward(self, seq_len: int, offset: int = 0, position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cos, sin: [1, seq_len, head_dim] for LigerRopeFunction.
        """
        if offset + seq_len > self.max_seq_len:
            self._build_cache(offset + seq_len)

        if position_ids is not None:
            cos = self.cos_cached[0][position_ids]  # [batch, seq_len, dim]
            sin = self.sin_cached[0][position_ids]
            return cos, sin
        else:
            return self.cos_cached[:, offset : offset + seq_len, :], self.sin_cached[:, offset : offset + seq_len, :]


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Applies rotary position embeddings to q and k via fused Liger kernel."""
    return LigerRopeFunction.apply(q, k, cos, sin)


class QKNorm(nn.Module):
    """Per-head RMS normalization for query/key vectors.

    Normalizes over head_dim (the last axis of a [B, n_heads, seq, head_dim]
    tensor) and applies a learnable per-head gain of shape (n_heads, head_dim).
    Stabilizes attention logits (Gemma 2, Chameleon). We roll our own rather
    than reuse LigerRMSNorm because that shares one weight vector across heads;
    here each head gets its own.
    """

    def __init__(self, n_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_heads, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_heads, seq, head_dim]
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * self.weight.to(x.dtype)[None, :, None, :]
        return x.to(dtype)


class Block(nn.Module):
    """One pre-norm transformer layer: RMSNorm -> attention -> LayerScale
    residual, then RMSNorm -> SwiGLU MLP -> LayerScale residual.

    RoPE cos/sin are computed once by the parent and passed in so the rotary
    cache is shared across all blocks.
    """

    def __init__(self, n_embd: int, n_heads: int, n_layers: int, ls_init: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.head_dim = n_embd // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

        # QK-norm: per-head RMSNorm on q/k before RoPE.
        self.q_norm = QKNorm(n_heads, self.head_dim)
        self.k_norm = QKNorm(n_heads, self.head_dim)

        # QK-gain: learnable per-head multiplicative gain on the attention scale.
        # Init to 1.0 so initial behavior matches the fixed 1/sqrt(d_k) baseline.
        self.qk_gain = nn.Parameter(torch.ones(n_heads))

        hidden_dim = int(8 * n_embd / 3)
        hidden_dim = ((hidden_dim + 127) // 128) * 128
        self.w_up = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w_gate = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, n_embd, bias=False)

        # LayerScale (CaiT, Touvron et al. 2021) — learnable per-channel residual
        # gates. Init 0.1 for a ~100M-class model; use 1e-4 for stacks >24 layers.
        self.ls_attn = nn.Parameter(torch.full((n_embd,), ls_init))
        self.ls_mlp = nn.Parameter(torch.full((n_embd,), ls_init))

        # Base GPT-2 style init for all Linears, then a scaled init for the
        # residual-output projections (proj / w_down) so deep-stack residual
        # variance stays controlled (NanoGPT std = 0.02/sqrt(2*n_layers)).
        for lin in (self.qkv, self.proj, self.w_up, self.w_gate, self.w_down):
            nn.init.normal_(lin.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
        nn.init.normal_(self.w_down.weight, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def forward(self, x, cos, sin, past_kv=None, use_cache=False):
        bsz, seq_len, _ = x.shape

        residual = x
        x_norm = self.ln1(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # QK-norm before RoPE.
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rope(q, k, cos, sin)

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # QK-gain: fold the learnable per-head gain into q so it scales the
        # q·k score before softmax — equivalent to `score * gain[h]` but
        # expressed as a plain tensor op SDPA can fuse.
        q = q * self.qk_gain.view(1, self.n_heads, 1, 1).to(q.dtype)

        # Plain causal attention so SDPA can dispatch to its fused flash / cuDNN
        # backends. During cached generation each step extends a single sequence,
        # so an explicit incremental causal mask over the concatenated KV is
        # needed instead of the is_causal flag.
        attn_mask = None
        is_causal = True
        if past_len > 0:
            total_len = past_len + seq_len
            query_positions = past_len + torch.arange(seq_len, device=q.device)
            key_positions = torch.arange(total_len, device=q.device)
            attn_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            is_causal = False

        with sdpa_kernel(
            [SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION,
             SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
            set_priority=True,
        ):
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0,
                is_causal=is_causal, scale=self.scale,
            )

        y = y.permute(0, 2, 1, 3).contiguous().view(bsz, seq_len, self.n_embd)
        y = self.proj(y)
        x = residual + self.ls_attn * y

        residual = x
        x_norm = self.ln2(x)
        up = self.w_up(x_norm)
        gate = self.w_gate(x_norm)
        mlp_out = self.w_down(LigerSiLUMulFunction.apply(gate, up))
        x = residual + self.ls_mlp * mlp_out

        new_kv = (k, v) if use_cache else None
        return x, new_kv
