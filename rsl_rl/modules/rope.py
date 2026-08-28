# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rotary position embedding (RoPE) with EXPLICIT positions.

Positions are always passed in as a tensor -- never assumed to be ``arange`` -- because the episode-context
policy rotates a KV cache whose slots hold arbitrary (ring-buffer) episode steps, and its memory rows all sit at
position 0. Half-split pairing: channel ``i`` pairs with ``i + head_dim / 2``, ``theta_i = base ** (-2 i / hd)``.
"""

from __future__ import annotations

import torch


def rope_cos_sin(
    pos: torch.Tensor, head_dim: int, base: float = 10000.0, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cos/sin tables for positions ``pos`` ``[..., L]``, shaped ``[..., 1, L, head_dim]``.

    The inserted head axis is what makes them broadcast against a ``[B, H, L, head_dim]`` query/key stream.
    """
    assert head_dim % 2 == 0, f"RoPE needs an even head_dim, got {head_dim}."
    half = head_dim // 2
    exponent = torch.arange(half, device=pos.device, dtype=dtype) * (2.0 / head_dim)
    inv_freq = torch.pow(torch.tensor(base, device=pos.device, dtype=dtype), -exponent)
    angles = pos.to(dtype).unsqueeze(-1) * inv_freq  # [..., L, half]
    angles = torch.cat([angles, angles], dim=-1)  # [..., L, head_dim]
    return angles.cos().unsqueeze(-3), angles.sin().unsqueeze(-3)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """``[x_1, x_2] -> [-x_2, x_1]`` on the last axis (halves, not interleaved pairs)."""
    first, second = x.chunk(2, dim=-1)
    return torch.cat([-second, first], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` ``[B, H, L, head_dim]`` by the tables :func:`rope_cos_sin` produced for its positions."""
    return x * cos + rotate_half(x) * sin
