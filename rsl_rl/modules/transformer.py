# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pre-LN transformer blocks with an external KV cache and RoPE, shared by the episode-context model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from rsl_rl.modules.rope import apply_rope, rope_cos_sin
from rsl_rl.utils import resolve_nn_activation


class MultiHeadAttention(nn.Module):
    """Dense multi-head attention with an explicit boolean mask (``True`` == attend).

    Keys and values are passed in **already projected** so that the caller can keep a KV cache and can share the
    memory-token projections across timesteps.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def project_kv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project ``x`` ``[B, L, d]`` into keys and values, both ``[B, L, d]``."""
        return self.k_proj(x), self.v_proj(x)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, length, _ = x.shape
        return x.transpose(1, 2).reshape(batch, length, self.d_model)

    @staticmethod
    def _expand_mask(attn_mask: torch.Tensor) -> torch.Tensor:
        """Bring a ``[Lq, Lk]`` or ``[B, Lq, Lk]`` boolean mask to the ``[B|1, 1, Lq, Lk]`` shape SDPA wants."""
        if attn_mask.dim() == 2:
            return attn_mask.unsqueeze(0).unsqueeze(0)
        if attn_mask.dim() == 3:
            return attn_mask.unsqueeze(1)
        return attn_mask

    def forward(
        self,
        query_input: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        q_pos: torch.Tensor | None = None,
        k_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Attend ``query_input`` ``[B, Lq, d]`` (pre-projection) over projected ``keys``/``values`` ``[B, Lk, d]``.

        ``q_pos``/``k_pos`` ``[B, L]`` are RoPE positions; keys are rotated at READ time so a KV cache can keep
        storing un-rotated keys and hand their stored positions in here. ``None`` means no rotation.
        """
        query = self._split_heads(self.q_proj(query_input))
        key = self._split_heads(keys)
        value = self._split_heads(values)
        if q_pos is not None:
            query = apply_rope(query, *rope_cos_sin(q_pos, self.head_dim, dtype=query.dtype))
        if k_pos is not None:
            key = apply_rope(key, *rope_cos_sin(k_pos, self.head_dim, dtype=key.dtype))
        mask = None if attn_mask is None else self._expand_mask(attn_mask)

        if need_weights:
            scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            attended = torch.matmul(weights, value)
        else:
            weights = None
            attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)

        return self.out_proj(self._merge_heads(attended)), weights


class TrunkBlock(nn.Module):
    """One pre-LN transformer block, split so that cached keys/values can be assembled by the caller."""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, activation: str) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), resolve_nn_activation(activation), nn.Linear(ff_dim, d_model)
        )

    def token_kv(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize the token stream and project it into keys/values (also used to fill the KV cache)."""
        normed = self.norm_attn(hidden)
        keys, values = self.attn.project_kv(normed)
        return normed, keys, values

    def token_forward(
        self,
        hidden: torch.Tensor,
        normed: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attn_mask: torch.Tensor | None,
        q_pos: torch.Tensor | None = None,
        k_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Finish the block for the query tokens, given the assembled ``[memory | tokens]`` keys/values."""
        attended, _ = self.attn(normed, keys, values, attn_mask=attn_mask, q_pos=q_pos, k_pos=k_pos)
        hidden = hidden + attended
        return hidden + self.ff(self.norm_ff(hidden))
