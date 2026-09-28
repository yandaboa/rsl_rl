# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flow-matching MLP actor for FPO++ (arXiv 2602.02481).

Conventions (from the reference implementation): t=1 is noise, t=0 is the action; x_t = t*eps + (1-t)*a; the MLP
predicts the velocity u = eps - a from cat(obs_latent, embed(t), x_t); sampling Euler-integrates t from 1 to 0.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import List

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP, HiddenState
from rsl_rl.utils import unpad_trajectories


def embed_timestep(t: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """``[..., 1] -> [..., embed_dim]``: ``[cos(t * 2^k), sin(t * 2^k)]`` for ``k < embed_dim / 2``."""
    freqs = 2 ** torch.arange(embed_dim // 2, device=t.device, dtype=t.dtype)
    scaled_t = t * freqs
    return torch.cat([torch.cos(scaled_t), torch.sin(scaled_t)], dim=-1)


class FlowMLPModel(MLPModel):
    """MLP velocity field over (normalized obs, timestep embedding, noisy action); no output distribution.

    ``forward(obs, stochastic_output=True)`` integrates from random noise (plus ``action_perturb_std`` Gaussian noise
    in train mode); ``stochastic_output=False`` integrates from zero noise (the deterministic action).
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
        timestep_embed_dim: int = 8,
        sampling_steps: int = 64,
        cfm_loss_t_inverse_cdf_beta: float = 1.0,
        cfm_loss_reduction: str = "sqrt",
        action_perturb_std: float = 0.02,
        actor_scale: float = 1.0,
        actor_mlp_output_scale: float = 1.0,
        actor_final_layer_weight_scale: float | None = None,
    ) -> None:
        assert distribution_cfg is None, "FlowMLPModel has no output distribution; leave distribution_cfg=None."
        assert cfm_loss_reduction in ("mean", "sum", "sqrt"), f"unknown cfm_loss_reduction {cfm_loss_reduction!r}"
        assert timestep_embed_dim % 2 == 0, "timestep_embed_dim must be even (cos/sin pairs)"
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, None)
        self.num_actions = output_dim
        self.timestep_embed_dim = timestep_embed_dim
        self.sampling_steps = sampling_steps
        self.cfm_loss_t_inverse_cdf_beta = cfm_loss_t_inverse_cdf_beta
        self.cfm_loss_reduction = cfm_loss_reduction
        self.action_perturb_std = action_perturb_std
        self.actor_scale = actor_scale
        self.mlp_output_scale = actor_mlp_output_scale

        # The MLP consumes cat(obs_latent, t_embed, x_t) (replaces the obs-only MLP the base class built).
        self.mlp = MLP(self.obs_dim + timestep_embed_dim + output_dim, output_dim, hidden_dims, activation)
        if actor_final_layer_weight_scale is not None and actor_final_layer_weight_scale != 1.0:
            final_layer = self.mlp[-1]
            assert isinstance(final_layer, nn.Linear), "expected the MLP to end in a Linear layer"
            with torch.no_grad():
                final_layer.weight.mul_(actor_final_layer_weight_scale)
                if final_layer.bias is not None:
                    final_layer.bias.mul_(actor_final_layer_weight_scale)

        # Diagnostics of the last stochastic forward (no distribution to read them from).
        self._output_mean: torch.Tensor | None = None
        self._output_std = torch.zeros(output_dim)

    # ------------------------------------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------------------------------------

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Sampled action (``stochastic_output``) or the zero-noise action.

        The stochastic path integrates the zero-noise rows in the same batched pass: they become ``output_mean``
        (the eval pool's action) and the per-dim RMS of (sample - zero-noise action) becomes ``output_std``.
        """
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        latent = self.get_latent(obs, masks, hidden_state)
        if not stochastic_output:
            action = self.integrate(latent, torch.zeros(latent.shape[0], self.num_actions, device=latent.device))
            self._output_mean = action
            return action
        n = latent.shape[0]
        x1 = torch.cat([torch.randn(n, self.num_actions, device=latent.device), latent.new_zeros(n, self.num_actions)])
        actions = self.integrate(torch.cat([latent, latent]), x1)
        sampled, mean = actions[:n], actions[n:]
        if self.training and self.action_perturb_std > 0:
            sampled = sampled + self.action_perturb_std * torch.randn_like(sampled)
        self._output_mean = mean
        self._output_std = (sampled - mean).pow(2).mean(dim=0).sqrt()
        return sampled

    def integrate(self, latent: torch.Tensor, x_t: torch.Tensor, sampling_steps: int | None = None) -> torch.Tensor:
        """Euler-integrate the velocity field from ``t=1`` (``x_t``) to ``t=0``; returns ``actor_scale * x_0``."""
        steps = self.sampling_steps if sampling_steps is None else sampling_steps
        t_path = torch.linspace(1.0, 0.0, steps + 1, device=latent.device, dtype=latent.dtype)
        dt = t_path[1:] - t_path[:-1]
        t_embed = embed_timestep(t_path[:-1, None], self.timestep_embed_dim)  # [steps, E]
        batch = latent.shape[0]
        for i in range(steps):
            u = self.mlp(torch.cat([latent, t_embed[i].expand(batch, -1), x_t], dim=-1)) * self.mlp_output_scale
            x_t = x_t + u * dt[i]
        return self.actor_scale * x_t

    def sample_cfm_t(self, shape: tuple[int, ...], device: torch.device | str) -> torch.Tensor:
        """``t = 0.005 + 0.99 * (1 - (1 - u)^(1/beta))``: inverse CDF of Beta(1, beta), kept off the endpoints."""
        u = torch.rand(shape, device=device)
        return 0.005 + 0.99 * (1.0 - (1.0 - u) ** (1.0 / self.cfm_loss_t_inverse_cdf_beta))

    # ------------------------------------------------------------------------------------------------
    # Conditional flow-matching loss
    # ------------------------------------------------------------------------------------------------

    def get_cfm_loss(
        self, obs: TensorDict, actions: torch.Tensor, eps: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sample CFM loss ``[B, S]`` and the implied ``x1_pred`` / ``x0_pred`` ``[B, S, A]``."""
        return self.get_cfm_loss_from_latent(self.get_latent(obs), actions, eps, t)

    def get_cfm_loss_from_latent(
        self, latent: torch.Tensor, actions: torch.Tensor, eps: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """:meth:`get_cfm_loss` on an already-normalized latent ``[B, D]``; ``eps [B, S, A]``, ``t [B, S, 1]``."""
        batch, num_samples, action_dim = eps.shape
        assert actions.shape == (batch, action_dim), f"actions {tuple(actions.shape)} vs eps {tuple(eps.shape)}"
        assert t.shape == (batch, num_samples, 1), f"t {tuple(t.shape)} vs eps {tuple(eps.shape)}"
        scaled_actions = actions[:, None, :] / self.actor_scale
        x_t = t * eps + (1.0 - t) * scaled_actions
        mlp_in = torch.cat(
            [latent[:, None, :].expand(batch, num_samples, -1), embed_timestep(t, self.timestep_embed_dim), x_t], dim=-1
        )
        velocity_pred = self.mlp(mlp_in) * self.mlp_output_scale
        x0_pred = x_t - t * velocity_pred
        x1_pred = x0_pred + velocity_pred
        squared_errors = (velocity_pred - (eps - scaled_actions)) ** 2
        if self.cfm_loss_reduction == "mean":
            loss = squared_errors.mean(dim=-1)
        elif self.cfm_loss_reduction == "sum":
            loss = squared_errors.sum(dim=-1)
        else:
            loss = squared_errors.sum(dim=-1) / (action_dim**0.5)
        return loss, x1_pred, x0_pred

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
        """Per-dim RMS of (sampled - zero-noise action) over the last stochastic batch (diagnostic only)."""
        return self._output_std

    @property
    def output_entropy(self) -> torch.Tensor:
        raise NotImplementedError("FlowMLPModel has no closed-form entropy.")

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError("FlowMLPModel has no distribution parameters.")

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("FlowMLPModel has no tractable log-probability; use FPO.")

    def get_kl_divergence(self, old_params, new_params) -> torch.Tensor:
        raise NotImplementedError("FlowMLPModel has no distribution parameters.")

    def as_jit(self) -> nn.Module:
        """TorchScript-able zero-noise sampler: ``forward(obs) -> action`` on the flat actor obs."""
        return _TorchFlowModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """ONNX-exportable zero-noise sampler (the Euler loop unrolls at trace time)."""
        return _OnnxFlowModel(self, verbose)


class _TorchFlowModel(nn.Module):
    """Normalizer + fixed-step zero-noise Euler integration."""

    def __init__(self, model: FlowMLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        steps = int(model.sampling_steps)
        t_path = torch.linspace(1.0, 0.0, steps + 1)
        self.register_buffer("dt", t_path[1:] - t_path[:-1])
        self.register_buffer("t_embed", embed_timestep(t_path[:-1, None], model.timestep_embed_dim))
        self.sampling_steps = steps
        self.num_actions = int(model.num_actions)
        self.actor_scale = float(model.actor_scale)
        self.mlp_output_scale = float(model.mlp_output_scale)
        self.input_size = int(model.obs_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.obs_normalizer(x)
        batch = latent.shape[0]
        x_t = torch.zeros(batch, self.num_actions, device=latent.device, dtype=latent.dtype)
        for i in range(self.sampling_steps):
            u = self.mlp(torch.cat([latent, self.t_embed[i].expand(batch, -1), x_t], dim=-1)) * self.mlp_output_scale
            x_t = x_t + u * self.dt[i]
        return self.actor_scale * x_t

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxFlowModel(_TorchFlowModel):
    is_recurrent: bool = False

    def __init__(self, model: FlowMLPModel, verbose: bool) -> None:
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
