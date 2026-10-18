# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flow actor for PolicyFlow (arXiv 2602.01156), ported from the authors' policyflow_torch reference.

Conventions (reference): t=0 is noise, t=1 is the action; RK2-midpoint integration on a uniform grid; the executed
action is ``phi(x0) + n`` with ``n ~ N(0, std^2)`` from a state-independent learnable std.
"""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import List

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP, HiddenState
from rsl_rl.utils import unpad_trajectories


class FourierEmbedding(nn.Module):
    """``t [B] -> [B, dim]``: fixed random Fourier features -> Linear -> Mish -> Linear (reference)."""

    def __init__(self, dim: int, scale: float = 16.0) -> None:
        super().__init__()
        assert dim >= 8, "embed_dim must be >= 8 (dim // 8 Fourier frequencies)"
        # Non-trainable, but saved with the checkpoint (random at construction).
        self.register_buffer("freqs", torch.randn(dim // 8) * scale)
        self.mlp = nn.Sequential(nn.Linear(2 * (dim // 8), dim), nn.Mish(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = t.unsqueeze(-1) * (2.0 * math.pi * self.freqs.to(t.dtype))
        return self.mlp(torch.cat([emb.cos(), emb.sin()], dim=-1))


class PolicyFlowModel(MLPModel):
    """Velocity field ``v(cat(x, t_emb(t) + cond(latent)))`` plus a learnable per-dim noise std.

    ``forward(obs, stochastic_output=True)`` integrates from ``x0 ~ N(0, I)`` and adds the Gaussian noise; the
    act-time ``latent / x0 / phi / noise`` are cached for the algorithm. ``stochastic_output=False`` integrates from
    ``x0 = 0`` without noise (the deterministic action).
    """

    is_flow_policy: bool = True
    """Duck-type marker for exporters that must not treat this as a Gaussian ``MLPModel``."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        embed_dim: int = 64,
        sampling_steps: int = 10,
        fourier_scale: float = 16.0,
        init_std: float = 1.0,
        log_std_min: float = -20.0,
        log_std_max: float = 4.0,
    ) -> None:
        assert distribution_cfg is None, "PolicyFlowModel has its own noise std; leave distribution_cfg=None."
        assert sampling_steps >= 1, "sampling_steps must be >= 1"
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, None)
        self.num_actions = output_dim
        self.embed_dim = embed_dim
        self.sampling_steps = int(sampling_steps)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)

        self.cond = nn.Linear(self.obs_dim, embed_dim)
        self.time_embed = FourierEmbedding(embed_dim, fourier_scale)
        # Replaces the obs-only MLP the base class built.
        self.mlp = MLP(output_dim + embed_dim, output_dim, hidden_dims, activation)
        linears = [m for m in self.mlp if isinstance(m, nn.Linear)]
        for i, layer in enumerate(linears):
            nn.init.xavier_normal_(layer.weight, gain=0.01 if i == len(linears) - 1 else 1.0)
        self.log_std = nn.Parameter(math.log(init_std) * torch.ones(output_dim))

        # Uniform schedule t_i = i / steps; training times = every t_i, its midpoint, and 1.0 (reference).
        steps = self.sampling_steps
        grid = [i / steps for i in range(steps)]
        train_grid = []
        for t in grid:
            train_grid += [t, t + 0.5 / steps]
        train_grid.append(1.0)
        self.register_buffer("train_time_grid", torch.tensor(train_grid), persistent=False)

        self._output_mean: torch.Tensor | None = None
        self._batch_size = 1
        self.last_latent: torch.Tensor | None = None
        self.last_x0: torch.Tensor | None = None
        self.last_phi: torch.Tensor | None = None
        self.last_noise: torch.Tensor | None = None

    # ------------------------------------------------------------------------------------------------
    # Flow
    # ------------------------------------------------------------------------------------------------

    @property
    def std(self) -> torch.Tensor:
        return self.log_std.clamp(self.log_std_min, self.log_std_max).exp()

    def velocity(self, latent: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``latent [B, D], x [B, A], t [B] -> [B, A]``."""
        return self.mlp(torch.cat([x, self.time_embed(t) + self.cond(latent)], dim=-1))

    def integrate(self, latent: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """RK2-midpoint from ``t=0`` (``x0``) to ``t=1``; returns ``phi = x_1``."""
        steps = self.sampling_steps
        dt = 1.0 / steps
        x = x0
        batch = latent.shape[0]
        for i in range(steps):
            t = torch.full((batch,), i / steps, device=latent.device, dtype=latent.dtype)
            v = self.velocity(latent, x, t)
            x_mid = x + v * dt / 2
            v = self.velocity(latent, x_mid, t + dt / 2)
            x = x + dt * v
        return x

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """``phi(x0) + n`` (``stochastic_output``) or the zero-noise ``phi(0)``.

        The stochastic path integrates the ``x0 = 0`` rows in the same batched pass (``output_mean``, the eval pool's
        action).
        """
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        latent = self.get_latent(obs, masks, hidden_state)
        n = latent.shape[0]
        self._batch_size = n
        if not stochastic_output:
            action = self.integrate(latent, latent.new_zeros(n, self.num_actions))
            self._output_mean = action
            return action
        x0 = torch.randn(n, self.num_actions, device=latent.device, dtype=latent.dtype)
        phis = self.integrate(torch.cat([latent, latent]), torch.cat([x0, torch.zeros_like(x0)]))
        phi, mean = phis[:n], phis[n:]
        noise = torch.randn_like(phi) * self.std
        self.last_latent = latent.detach()
        self.last_x0 = x0
        self.last_phi = phi.detach()
        self.last_noise = noise.detach()
        self._output_mean = mean
        return phi + noise

    # ------------------------------------------------------------------------------------------------
    # Distribution-shaped API the runner / PPO base touch
    # ------------------------------------------------------------------------------------------------

    @property
    def output_mean(self) -> torch.Tensor:
        """Zero-noise action of the last forward pass."""
        assert self._output_mean is not None, "no forward pass yet"
        return self._output_mean

    @property
    def output_std(self) -> torch.Tensor:
        """Std of the injected Gaussian noise, ``[B, A]`` for the last batch size."""
        return self.std.detach().expand(self._batch_size, -1)

    @property
    def output_entropy(self) -> torch.Tensor:
        """Gaussian entropy of the injected noise per row ``[B]``."""
        ent = (0.5 + 0.5 * math.log(2 * math.pi) + self.log_std.clamp(self.log_std_min, self.log_std_max)).sum()
        return ent.expand(self._batch_size)

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError("PolicyFlowModel has no per-sample distribution parameters.")

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("PolicyFlowModel has no tractable action log-probability; use PolicyFlow.")

    def get_kl_divergence(self, old_params, new_params) -> torch.Tensor:
        raise NotImplementedError("PolicyFlowModel has no per-sample distribution parameters.")

    def as_jit(self) -> nn.Module:
        """TorchScript-able zero-noise sampler: ``forward(obs) -> action`` on the flat actor obs."""
        return _TorchPolicyFlowModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """ONNX-exportable zero-noise sampler (the RK2 loop unrolls at trace time)."""
        return _OnnxPolicyFlowModel(self, verbose)


class _TorchPolicyFlowModel(nn.Module):
    """Normalizer + fixed-step zero-noise RK2 integration."""

    def __init__(self, model: PolicyFlowModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.cond = copy.deepcopy(model.cond)
        self.time_embed = copy.deepcopy(model.time_embed)
        self.sampling_steps = int(model.sampling_steps)
        self.num_actions = int(model.num_actions)
        self.input_size = int(model.obs_dim)

    def _velocity(self, cond: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([x, self.time_embed(t) + cond], dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.obs_normalizer(x)
        cond = self.cond(latent)
        batch = latent.shape[0]
        dt = 1.0 / self.sampling_steps
        a = torch.zeros(batch, self.num_actions, device=latent.device, dtype=latent.dtype)
        for i in range(self.sampling_steps):
            t = torch.full((batch,), i / self.sampling_steps, device=latent.device, dtype=latent.dtype)
            v = self._velocity(cond, a, t)
            a_mid = a + v * dt / 2
            v = self._velocity(cond, a_mid, t + dt / 2)
            a = a + dt * v
        return a

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxPolicyFlowModel(_TorchPolicyFlowModel):
    is_recurrent: bool = False

    def __init__(self, model: PolicyFlowModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> List[str]:
        return ["obs"]

    @property
    def output_names(self) -> List[str]:
        return ["actions"]
