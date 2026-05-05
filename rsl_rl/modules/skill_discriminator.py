# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.networks import MLP, EmpiricalNormalization


class SkillDiscriminator(nn.Module):
    """Discriminator q(z | s) for DIAYN-style diversity training.

    A simple MLP classifier over a fixed alphabet of ``num_skills`` skills. Optionally wraps the
    input with :class:`EmpiricalNormalization` so the discriminator sees standardized states.
    """

    def __init__(
        self,
        input_dim: int,
        num_skills: int,
        hidden_dims: list[int],
        activation: str = "elu",
        obs_normalization: bool = True,
    ) -> None:
        super().__init__()
        self.num_skills = num_skills
        self.input_dim = input_dim

        if obs_normalization:
            self.obs_normalizer: nn.Module = EmpiricalNormalization(input_dim)
        else:
            self.obs_normalizer = nn.Identity()

        self.net = MLP(
            input_dim=input_dim,
            output_dim=num_skills,
            hidden_dims=hidden_dims,
            activation=activation,
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Return logits over the skill alphabet."""
        return self.net(self.obs_normalizer(s))

    def update_normalization(self, s: torch.Tensor) -> None:
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            self.obs_normalizer.update(s)
