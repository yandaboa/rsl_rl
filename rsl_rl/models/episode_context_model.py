# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-episode context transformer: a joint actor-critic over the frames of the current episode.

The trunk is a causal transformer over ``x_{t-L+1} .. x_t`` (clipped at the episode start) with RoPE positions
and a learned ``start_embed`` on step 0. Two forward paths must agree numerically: the incremental one
(:meth:`EpisodeContextModel.forward_step`, per-environment rolling KV cache) and the batched one
(:meth:`EpisodeContextModel.forward_window`, one causal pass over ``[prefix | window]``).

Optional features, each parameter-for-parameter absent when off: cross-episode memory tokens (``memory_tokens``),
a privileged encoder (``privileged_group``), an auxiliary next-state head (``next_state_head_hidden_dims``) and a
critic that is either a privileged single-step MLP, a value head on the shared trunk, or a separate ``critic_*``
trunk (``critic_design``).

Parameter names are the ones of the rsl-rl 3.1 ``ActorCriticEpisodeContext``, so its checkpoints load unchanged.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.modules import MLP, EmpiricalNormalization
from rsl_rl.modules.transformer import MultiHeadAttention, TrunkBlock
from rsl_rl.utils import resolve_nn_activation

# Which parameter holds the action noise, per ``noise_std_type``. A checkpoint written under one noise type
# cannot be loaded into another (the shapes differ: ``[A]`` vs ``[d_model, A]``).
_NOISE_PARAM_NAME = {"scalar": "std", "log": "log_std", "gsde": "log_std"}


def upcast_from_half(tensor: torch.Tensor) -> torch.Tensor:
    """fp16/bf16 -> fp32; fp32 and fp64 untouched (not ``.float()``, which would downcast fp64)."""
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.float()
    return tensor


class EpisodeContextDistribution:
    """Diagonal Gaussian over the action, parameterized by the owning model's noise parameter.

    Not an ``nn.Module``: the noise parameter (``std`` / ``log_std``) stays on the model under its 3.1 name.
    Under ``"gsde"`` the std is ``sqrt(h_t^2 @ exp(log_std)^2 + eps)`` and sampling is from the marginal
    Normal (the 3.1 / ``LegacyGsdeDistribution`` semantics: noise iid across steps and envs).
    """

    def __init__(self, model: EpisodeContextModel, epsilon: float = 1e-6) -> None:
        self._model = model
        self.epsilon = epsilon
        self._normal: Normal | None = None

    def update(self, mean: torch.Tensor, hidden: torch.Tensor) -> None:
        model = self._model
        if model.noise_std_type == "gsde":
            # Always fp32 even under autocast: ``h^2`` sits at the fp16 edge for d >= 256 and ``sigma^2``
            # underflows to zero below 2.4e-4, which makes every log-prob inf.
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                std = torch.exp(upcast_from_half(model.log_std))
                variance = torch.matmul(upcast_from_half(hidden) ** 2, std**2)
                self._normal = Normal(upcast_from_half(mean), torch.sqrt(variance + self.epsilon))
            return
        if model.noise_std_type == "scalar":
            std = model.std.expand_as(mean)
        else:
            std = torch.exp(model.log_std).expand_as(mean)
        self._normal = Normal(upcast_from_half(mean), upcast_from_half(std))

    def _dist(self) -> Normal:
        if self._normal is None:
            raise RuntimeError("Distribution not initialized: run the policy forward first.")
        return self._normal

    def sample(self) -> torch.Tensor:
        if self._model.noise_std_type == "gsde":
            with torch.no_grad():
                return self._dist().rsample()
        return self._dist().sample()

    @property
    def mean(self) -> torch.Tensor:
        return self._dist().mean

    @property
    def std(self) -> torch.Tensor:
        return self._dist().stddev

    @property
    def stddev(self) -> torch.Tensor:
        return self._dist().stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self._dist().entropy().sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return (self.mean, self.std)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self._dist().log_prob(outputs).sum(dim=-1)

    @staticmethod
    def kl_divergence(old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """KL(old || new) with the 3.1 PPO formula (the ``1e-5`` inside the log is part of the schedule)."""
        old_mu, old_sigma = old_params
        mu, sigma = new_params
        return torch.sum(
            torch.log(sigma / old_sigma + 1.0e-5)
            + (torch.square(old_sigma) + torch.square(old_mu - mu)) / (2.0 * torch.square(sigma))
            - 0.5,
            dim=-1,
        )


class MemoryTokenWriter(nn.Module):
    """The writer ``G``: ``Z_new = anchor + delta(H)``, one combined self+cross attention over the trunk's output.

    ``H`` ``[B, M + T, d]`` is the snapshot of one finished pass (memory rows first, then the episode's readouts);
    the queries are its memory rows. The FF's final linear is zero-initialized, so ``Z_new == anchor`` at init.
    """

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, activation: str) -> None:
        super().__init__()
        self.norm_query = nn.LayerNorm(d_model)
        self.norm_key = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), resolve_nn_activation(activation), nn.Linear(ff_dim, d_model)
        )
        self.zero_delta_init()

    def zero_delta_init(self) -> None:
        """Zero the FF's output layer. Re-applied by the owner AFTER its global init sweep."""
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(
        self,
        hidden: torch.Tensor,
        num_memory: int,
        anchor: torch.Tensor,
        mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        queries = hidden[:, :num_memory]
        keys, values = self.attn.project_kv(self.norm_key(hidden))

        if mask is not None:
            key_mask = mask.bool().clone()
            # The memory rows are always readable: a fully masked softmax row is NaN.
            key_mask[:, :num_memory] = True
            attn_mask = key_mask.unsqueeze(1).expand(-1, num_memory, -1)
        else:
            attn_mask = None

        attended, weights = self.attn(
            self.norm_query(queries), keys, values, attn_mask=attn_mask, need_weights=need_weights
        )
        delta = self.ff(self.norm_ff(queries + attended))
        return anchor.to(device=delta.device, dtype=delta.dtype) + delta, weights


@dataclass(frozen=True)
class EpisodeContextPrefix:
    """The history a minibatch of environments needs to re-infer its window in one causal pass.

    Rides in the ``hidden_states`` slot of :class:`rsl_rl.storage.RolloutStorage.Batch`.

    ``obs`` ``[P, B, obs_dim]`` are the frames PRECEDING the window, already normalized exactly as the acting
    path normalized them (a prefix frame usually predates the last normalizer commit). ``positions`` ``[P, B]``
    and ``window_positions`` ``[W, B]`` are episode steps. The memory fields (``memory_tokens > 0``) carry either
    a ready ``memory`` ``[B, n_seg, M, d]`` or the detached source episodes ``source_hidden`` ``[B, n_seg, M+T, d]``
    the update turns into ``Z`` in-graph. ``next_*`` are the auxiliary next-state targets.
    """

    obs: torch.Tensor
    positions: torch.Tensor
    window_positions: torch.Tensor
    memory: torch.Tensor | None = None
    memory_segments: torch.Tensor | None = None
    source_hidden: torch.Tensor | None = None
    source_valid: torch.Tensor | None = None
    segment_has_source: torch.Tensor | None = None
    next_delta: torch.Tensor | None = None
    next_valid: torch.Tensor | None = None
    next_target_delta: torch.Tensor | None = None


class EpisodeContextModel(nn.Module):
    """Causal transformer actor over the current episode + critic (see the module docstring)."""

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        context_length: int = 80,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        max_episode_length: int = 80,
        ff_mult: int = 4,
        embed_hidden_dims: tuple[int] | list[int] = (),
        actor_hidden_dims: tuple[int] | list[int] = [256],
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "gelu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        normalizer_until: int | None = None,
        critic_design: str = "privileged",
        detach_critic_trunk: bool = False,
        memory_tokens: int = 0,
        episodes_per_trial: int = 2,
        next_state_head_hidden_dims: tuple[int] | list[int] | None = None,
        next_state_pred_dims: tuple[int, int] | list[int] | None = None,
        next_state_target_group: str | None = None,
        next_state_target_dim: int = 0,
        privileged_group: str | None = None,
        privileged_encoder_hidden_dims: tuple[int] | list[int] = [256, 128],
        privileged_embed_dim: int = 32,
        **kwargs: dict[str, Any],
    ) -> None:
        """Args mirror the 3.1 ``ActorCriticEpisodeContext``. ``obs_groups`` uses the 5.2 set keys ``"actor"`` /
        ``"critic"``; the 3.1 ``"policy"`` key is accepted as an alias for ``"actor"``.

        ``context_length`` >= ``max_episode_length`` means the whole episode. ``noise_std_type`` is ``"scalar"``,
        ``"log"`` or ``"gsde"`` (``log_std`` keyed on the trunk readout, marginal sampling). ``critic_design`` is
        ``"privileged"`` (single-step MLP on the critic groups), ``"shared_trunk"`` (value head on the actor's
        ``h_t``; ``detach_critic_trunk`` cuts the value gradient into the trunk) or ``"separate_trunk"`` (own
        ``critic_*`` transformer over the actor observation). ``privileged_group`` must be the LAST actor group;
        it enters the token through a zero-initialized projection, so ``token_embed`` keeps the non-privileged
        width and a BC init loads with identical shapes.
        """
        if kwargs:
            print(f"EpisodeContextModel.__init__ got unexpected arguments, which will be ignored: {list(kwargs)}")
        super().__init__()

        if critic_design not in ("privileged", "shared_trunk", "separate_trunk"):
            raise ValueError(
                f"Unknown critic_design: {critic_design!r}. Should be 'privileged', 'shared_trunk' or"
                " 'separate_trunk'."
            )
        self.critic_design = critic_design
        self.detach_critic_trunk = bool(detach_critic_trunk)

        actor_groups = obs_groups.get("actor", obs_groups.get("policy"))
        assert actor_groups is not None, f"obs_groups needs an 'actor' (or legacy 'policy') set: {obs_groups}"
        self.obs_groups = {"actor": list(actor_groups), "critic": list(obs_groups.get("critic", actor_groups))}
        num_actor_obs = 0
        for obs_group in self.obs_groups["actor"]:
            assert len(obs[obs_group].shape) == 2, "The EpisodeContextModel only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in self.obs_groups["critic"]:
            # Neither trunk-based critic reads the critic groups, so a group this env does not publish is fine.
            if critic_design in ("shared_trunk", "separate_trunk") and obs_group not in obs:
                continue
            assert len(obs[obs_group].shape) == 2, "The EpisodeContextModel only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.privileged_group = privileged_group
        if privileged_group is None:
            self.privileged_dim = 0
        else:
            assert self.obs_groups["actor"][-1] == privileged_group, (
                f"privileged_group='{privileged_group}' must be the LAST entry of obs_groups['actor']"
                f" ({self.obs_groups['actor']}): it is read off the tail of the concatenated frame."
            )
            self.privileged_dim = int(obs[privileged_group].shape[-1])
            assert self.privileged_dim > 0, f"privileged_group='{privileged_group}' is empty."
        self.num_token_obs = num_actor_obs - self.privileged_dim

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_episode_length = int(max_episode_length)
        self.context_length = int(context_length)
        # Reachable history: an episode is at most T frames long. Sizes the KV cache, the storage ring and the mask.
        self.context_span = max(1, min(self.context_length, self.max_episode_length))

        if len(embed_hidden_dims) == 0:
            self.token_embed = nn.Linear(self.num_token_obs, d_model)
        else:
            self.token_embed = MLP(self.num_token_obs, d_model, list(embed_hidden_dims), activation)
        self.start_embed = nn.Parameter(torch.zeros(d_model))

        if self.privileged_dim > 0:
            assert len(privileged_encoder_hidden_dims) > 0 and privileged_embed_dim > 0, (
                "privileged_group needs privileged_encoder_hidden_dims and privileged_embed_dim > 0."
            )
            self.privileged_encoder = MLP(
                self.privileged_dim, privileged_embed_dim, list(privileged_encoder_hidden_dims), "elu"
            )
            self.privileged_proj = nn.Linear(privileged_embed_dim, d_model)

        ff_dim = ff_mult * d_model
        self.blocks = nn.ModuleList([TrunkBlock(d_model, num_heads, ff_dim, activation) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)

        self.num_memory_tokens = int(memory_tokens)
        self.episodes_per_trial = int(episodes_per_trial)
        self.hidden_history_span = self.num_memory_tokens + self.max_episode_length
        if self.num_memory_tokens > 0:
            self.z_init = nn.Parameter(torch.zeros(self.num_memory_tokens, d_model))
            self.memory_pos_embed = nn.Parameter(torch.zeros(self.num_memory_tokens, d_model))
            self.writer = MemoryTokenWriter(d_model, num_heads, ff_dim, activation)

        # The value head keeps the name ``critic`` in every design: downstream tooling keys on ``critic*``.
        self.actor = MLP(d_model, num_actions, list(actor_hidden_dims), activation)
        critic_input_dim = num_critic_obs if self.critic_design == "privileged" else d_model
        self.critic = MLP(critic_input_dim, 1, list(critic_hidden_dims), activation)

        self.next_state_target_group = next_state_target_group
        self.next_state_target_dim = int(next_state_target_dim)
        self.next_state_target_normalizer = None
        if next_state_head_hidden_dims is None:
            self.next_state_head = None
            self.next_state_pred_start, self.next_state_pred_end = 0, 0
            assert self.next_state_target_group is None, (
                "next_state_target_group needs next_state_head_hidden_dims: there is no head to predict it with."
            )
        else:
            start, end = (0, num_actor_obs) if next_state_pred_dims is None else next_state_pred_dims
            self.next_state_pred_start, self.next_state_pred_end = int(start), int(end)
            assert 0 <= self.next_state_pred_start < self.next_state_pred_end <= num_actor_obs, (
                f"next_state_pred_dims=({start}, {end}) is not a slice of the {num_actor_obs}-d actor observation."
            )
            if self.next_state_target_group is None:
                assert self.next_state_target_dim == 0, "next_state_target_dim > 0 needs next_state_target_group."
            else:
                assert self.next_state_target_dim > 0, (
                    f"next_state_target_group='{self.next_state_target_group}' needs next_state_target_dim > 0."
                )
                # mm-scale raw deltas: the stock eps (1e-2) would squash the target instead of scaling it.
                self.next_state_target_normalizer = EmpiricalNormalization(self.next_state_target_dim, eps=1e-6)
            self.next_state_head = MLP(
                d_model + num_actions,
                self.next_state_pred_end - self.next_state_pred_start + self.next_state_target_dim,
                list(next_state_head_hidden_dims),
                activation,
            )

        if self.critic_design == "separate_trunk":
            if len(embed_hidden_dims) == 0:
                self.critic_token_embed = nn.Linear(self.num_token_obs, d_model)
            else:
                self.critic_token_embed = MLP(self.num_token_obs, d_model, list(embed_hidden_dims), activation)
            self.critic_start_embed = nn.Parameter(torch.zeros(d_model))
            self.critic_blocks = nn.ModuleList([
                TrunkBlock(d_model, num_heads, ff_dim, activation) for _ in range(num_layers)
            ])
            self.critic_final_norm = nn.LayerNorm(d_model)
            if self.num_memory_tokens > 0:
                self.critic_memory_pos_embed = nn.Parameter(torch.zeros(self.num_memory_tokens, d_model))

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(self.num_token_obs, until=normalizer_until)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        if self.privileged_dim > 0 and actor_obs_normalization:
            self.privileged_normalizer = EmpiricalNormalization(self.privileged_dim, until=normalizer_until)
        else:
            self.privileged_normalizer = torch.nn.Identity()
        # A trunk critic never reads the critic observation, so it gets no normalizer.
        self.critic_obs_normalization = critic_obs_normalization and self.critic_design == "privileged"
        if self.critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs, until=normalizer_until)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        elif noise_std_type == "gsde":
            self.log_std = nn.Parameter(torch.ones(d_model, num_actions) * math.log(init_noise_std))
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar', 'log' or 'gsde'")

        self.distribution = EpisodeContextDistribution(self)
        Normal.set_default_validate_args(False)

        self._init_weights()
        self._reset_runtime_state()

        print(
            f"Episode-context trunk: L={num_layers} d={d_model} heads={num_heads}"
            f" context={self.context_length} (span {self.context_span}) T={self.max_episode_length}"
            + (
                f" M={self.num_memory_tokens} memory tokens (K={self.episodes_per_trial})"
                if self.num_memory_tokens > 0
                else " (no memory)"
            )
            + f", critic_design={self.critic_design}"
        )

    # --------------------------------------------------------------------------------------------------------
    # Initialization / runtime state
    # --------------------------------------------------------------------------------------------------------

    def _init_weights(self) -> None:
        """GPT-style init: small normal weights, residual output projections scaled by ``1 / sqrt(2 L)``."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        residual_scale = (2.0 * max(self.num_layers, 1)) ** -0.5
        for block in self.blocks:
            block.attn.out_proj.weight.data.mul_(residual_scale)
            block.ff[-1].weight.data.mul_(residual_scale)
        nn.init.normal_(self.start_embed, mean=0.0, std=0.02)
        if self.num_memory_tokens > 0:
            self.writer.attn.out_proj.weight.data.mul_(residual_scale)
            self.writer.zero_delta_init()
            nn.init.normal_(self.memory_pos_embed, mean=0.0, std=0.02)
            nn.init.normal_(self.z_init, mean=0.0, std=0.02)
        if self.critic_design == "separate_trunk":
            for block in self.critic_blocks:
                block.attn.out_proj.weight.data.mul_(residual_scale)
                block.ff[-1].weight.data.mul_(residual_scale)
            nn.init.normal_(self.critic_start_embed, mean=0.0, std=0.02)
            if self.num_memory_tokens > 0:
                nn.init.normal_(self.critic_memory_pos_embed, mean=0.0, std=0.02)
        if self.privileged_dim > 0:
            # Exactly zero: the token is the non-privileged one at init, so a BC init is reproduced bit for bit.
            nn.init.zeros_(self.privileged_proj.weight)
            nn.init.zeros_(self.privileged_proj.bias)
        if self.next_state_head is not None:
            nn.init.zeros_(self.next_state_head[-1].weight)
            nn.init.zeros_(self.next_state_head[-1].bias)

    def _reset_runtime_state(self) -> None:
        """Drop every acting-time buffer (lazily re-allocated on the first :meth:`forward_step`)."""
        self._num_envs: int | None = None
        self._key_cache: list[torch.Tensor] | None = None
        self._value_cache: list[torch.Tensor] | None = None
        self._critic_key_cache: list[torch.Tensor] | None = None
        self._critic_value_cache: list[torch.Tensor] | None = None
        # Episode step stored in every cache slot, ``-1`` for empty. Slot ``p % context_span`` holds position ``p``.
        self._cache_positions: torch.Tensor | None = None
        self._positions: torch.Tensor | None = None
        self._last_hidden: torch.Tensor | None = None
        self._last_critic_hidden: torch.Tensor | None = None
        # Update path: the ``h`` the last :meth:`act` computed for a minibatch, with the obs object it came from,
        # so :meth:`evaluate` can prove it is looking at the same minibatch before reusing it.
        self._window_hidden: torch.Tensor | None = None
        self._window_hidden_obs: Any = None
        self._critic_window_hidden: torch.Tensor | None = None
        self._critic_window_hidden_obs: Any = None
        self._memory: torch.Tensor | None = None
        self._memory_key_cache: list[torch.Tensor] | None = None
        self._memory_value_cache: list[torch.Tensor] | None = None
        self._critic_memory_key_cache: list[torch.Tensor] | None = None
        self._critic_memory_value_cache: list[torch.Tensor] | None = None
        self._hidden_history: torch.Tensor | None = None
        self._history_valid: torch.Tensor | None = None
        self._episode_index: torch.Tensor | None = None

    def initialize_state(self, num_envs: int, device: torch.device | str, dtype: torch.dtype | None = None) -> None:
        """Allocate the acting-time KV cache for ``num_envs`` environments (all episodes start at step 0)."""
        # Allocated as NORMAL tensors even when the rollout runs under ``torch.inference_mode()``: an inference
        # tensor cannot be updated in place outside that mode, which is where ``reset()`` may be called from.
        with torch.inference_mode(False):
            self._initialize_state(num_envs, device, dtype)

    def _initialize_state(self, num_envs: int, device: torch.device | str, dtype: torch.dtype | None) -> None:
        dtype = self.start_embed.dtype if dtype is None else dtype
        span = self.context_span
        self._num_envs = num_envs
        self._key_cache = [
            torch.zeros(num_envs, span, self.d_model, device=device, dtype=dtype) for _ in range(self.num_layers)
        ]
        self._value_cache = [
            torch.zeros(num_envs, span, self.d_model, device=device, dtype=dtype) for _ in range(self.num_layers)
        ]
        self._cache_positions = torch.full((num_envs, span), -1, device=device, dtype=torch.long)
        self._positions = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._last_hidden = None
        self._window_hidden = None
        self._window_hidden_obs = None
        self._last_critic_hidden = None
        self._critic_window_hidden = None
        self._critic_window_hidden_obs = None
        if self.critic_design == "separate_trunk":
            self._critic_key_cache = [
                torch.zeros(num_envs, span, self.d_model, device=device, dtype=dtype) for _ in range(self.num_layers)
            ]
            self._critic_value_cache = [
                torch.zeros(num_envs, span, self.d_model, device=device, dtype=dtype) for _ in range(self.num_layers)
            ]
        if self.num_memory_tokens > 0:
            self._memory = self.z_init.detach().to(device=device, dtype=dtype).unsqueeze(0).repeat(num_envs, 1, 1)
            self._memory_key_cache = [
                torch.zeros(num_envs, self.num_memory_tokens, self.d_model, device=device, dtype=dtype)
                for _ in range(self.num_layers)
            ]
            self._memory_value_cache = [
                torch.zeros(num_envs, self.num_memory_tokens, self.d_model, device=device, dtype=dtype)
                for _ in range(self.num_layers)
            ]
            self._hidden_history = torch.zeros(
                num_envs, self.hidden_history_span, self.d_model, device=device, dtype=dtype
            )
            self._history_valid = torch.zeros(num_envs, self.hidden_history_span, device=device, dtype=torch.bool)
            self._episode_index = torch.zeros(num_envs, device=device, dtype=torch.long)
            if self.critic_design == "separate_trunk":
                self._critic_memory_key_cache = [
                    torch.zeros(num_envs, self.num_memory_tokens, self.d_model, device=device, dtype=dtype)
                    for _ in range(self.num_layers)
                ]
                self._critic_memory_value_cache = [
                    torch.zeros(num_envs, self.num_memory_tokens, self.d_model, device=device, dtype=dtype)
                    for _ in range(self.num_layers)
                ]
            self._prefill_memory(torch.arange(num_envs, device=device))

    def _ensure_state(self, num_envs: int, device: torch.device, dtype: torch.dtype) -> None:
        if self._num_envs != num_envs or self._key_cache is None or self._key_cache[0].device != device:
            self.initialize_state(num_envs, device, dtype)

    @property
    def positions(self) -> torch.Tensor | None:
        """Episode step of every environment on the acting path (``None`` before the first step)."""
        return self._positions

    @property
    def context_prefix_length(self) -> int:
        """Frames the batched path needs in front of a window: ``num_layers * (span - 1)``, capped at ``T - 1``."""
        return min(self.max_episode_length - 1, self.num_layers * (self.context_span - 1))

    # --------------------------------------------------------------------------------------------------------
    # Cross-episode memory
    # --------------------------------------------------------------------------------------------------------

    @property
    def memory(self) -> torch.Tensor | None:
        """The acting path's ``Z`` ``[num_envs, M, d]`` (``None`` before the first step, or with no memory)."""
        return self._memory

    @property
    def episode_index_in_trial(self) -> torch.Tensor | None:
        """How many episodes of the current trial are already written into ``Z``, per environment."""
        return self._episode_index

    def initial_memory(self, batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
        """The learned "no memory yet" rows ``[B, M, d]``, differentiable."""
        self._assert_memory_enabled()
        memory = self.z_init.unsqueeze(0).expand(batch_size, -1, -1)
        return memory if device is None else memory.to(device)

    def write_memory(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the writer ``G``: ``Z_{e+1} = z_init + delta(H_e)``, in graph (the anchor ``z_init`` trains)."""
        self._assert_memory_enabled()
        return self.writer(hidden, self.num_memory_tokens, self.z_init, mask=mask, need_weights=need_weights)

    def memory_from_prefix(self, prefix: EpisodeContextPrefix) -> torch.Tensor | None:
        """Per-segment ``Z`` ``[B, n_seg, M, d]`` from a minibatch's cached source episodes, IN GRAPH.

        Trains the writer and ``z_init``, never the source episode's trunk (the cached ``H`` is detached). One
        write off ``z_init``: exact for ``episodes_per_trial = 2``, an approximation for later episodes.
        """
        if self.num_memory_tokens == 0 or prefix.source_hidden is None:
            return None
        has_source = prefix.segment_has_source
        batch_size, num_segments = has_source.shape
        source = prefix.source_hidden.reshape(batch_size * num_segments, -1, self.d_model)
        valid = prefix.source_valid.reshape(batch_size * num_segments, -1)
        # ``expand`` (not ``repeat``): differentiable and free, so an all-z_init minibatch still trains z_init.
        memory = self.z_init.view(1, 1, self.num_memory_tokens, self.d_model).expand(batch_size, num_segments, -1, -1)
        rows = has_source.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        if rows.numel() == 0:
            return memory
        written, _ = self.write_memory(source[rows].detach(), mask=valid[rows])
        flat = memory.reshape(batch_size * num_segments, self.num_memory_tokens, self.d_model)
        return flat.index_copy(0, rows, written.to(flat.dtype)).view(batch_size, num_segments, -1, self.d_model)

    def get_episode_hidden(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``H`` of the pass in progress plus its validity mask, ``[n, M + T, d]`` and ``[n, M + T]``, detached."""
        self._assert_memory_enabled()
        if self._hidden_history is None:
            raise RuntimeError("get_episode_hidden() called before the first forward_step(); no state allocated.")
        if env_ids is None:
            return self._hidden_history.detach(), self._history_valid.clone()
        return self._hidden_history[env_ids].detach(), self._history_valid[env_ids].clone()

    def _assert_memory_enabled(self) -> None:
        if self.num_memory_tokens == 0:
            raise RuntimeError("This policy was built with memory_tokens=0: there is no z_init and no writer.")

    def _prepare_memory(self, memory: torch.Tensor | None, batch_size: int, device: torch.device) -> torch.Tensor:
        """Bring the caller's memory to ``[B, n_seg, M, d]``, defaulting to ``z_init`` for every row."""
        if memory is None:
            return self.z_init.view(1, 1, self.num_memory_tokens, self.d_model).expand(batch_size, 1, -1, -1)
        if memory.dim() == 3:  # [B, M, d] -- one segment per row
            memory = memory.unsqueeze(1)
        assert memory.shape[0] == batch_size and memory.shape[2:] == (self.num_memory_tokens, self.d_model), (
            f"memory must be [B, n_seg, {self.num_memory_tokens}, {self.d_model}] with B={batch_size},"
            f" got {tuple(memory.shape)}"
        )
        return memory.to(device)

    @staticmethod
    def _segments_from_positions(positions: torch.Tensor) -> torch.Tensor:
        """Segment index of every row of ``[S, B]`` episode steps: a row at step 0 opens the next segment."""
        return ((positions == 0).long().cumsum(dim=0) - 1).clamp(min=0)

    def _memory_token_input(self, memory: torch.Tensor, critic: bool = False) -> torch.Tensor:
        """Trunk INPUT rows for a memory ``[B, n_seg, M, d]``: ``Z_i + memory_pos_embed[i]``, flattened."""
        memory_pos_embed = self._pathway(critic)[2]
        batch_size, num_segments = memory.shape[0], memory.shape[1]
        return (memory + memory_pos_embed).reshape(batch_size, num_segments * self.num_memory_tokens, -1)

    def _memory_prefix_mask(self, token_mask: torch.Tensor, num_segments: int, segments: torch.Tensor) -> torch.Tensor:
        """Grow a token-only mask ``[B, S, S]`` into the full ``[B, nM + S, nM + S]`` of ``[memory | tokens]``.

        Memory row ``i`` of block ``k`` attends to rows ``<= i`` of block ``k`` only; a token attends to the whole
        memory block of ITS segment plus its usual token reach. Tokens never feed back into the memory rows.
        """
        batch_size, num_steps = segments.shape
        device = token_mask.device
        total = num_segments * self.num_memory_tokens
        index = torch.arange(total, device=device)
        block, row = index // self.num_memory_tokens, index % self.num_memory_tokens
        memory_memory = (block.unsqueeze(1) == block.unsqueeze(0)) & (row.unsqueeze(1) >= row.unsqueeze(0))
        token_memory = segments.clamp(max=num_segments - 1).unsqueeze(-1) == block.view(1, 1, -1)  # [B, S, nM]
        top = torch.cat(
            [
                memory_memory.unsqueeze(0).expand(batch_size, -1, -1),
                torch.zeros(batch_size, total, num_steps, dtype=torch.bool, device=device),
            ],
            dim=2,
        )
        return torch.cat([top, torch.cat([token_memory, token_mask], dim=2)], dim=1)

    def _memory_trunk(
        self, rows: torch.Tensor, critic: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Push the ``M`` memory INPUT rows ``[B, M, d]`` through the trunk on their own (causal among themselves)."""
        _, _, _, blocks, final_norm = self._pathway(critic)
        index = torch.arange(self.num_memory_tokens, device=rows.device)
        attn_mask = (index.unsqueeze(1) >= index.unsqueeze(0)).unsqueeze(0)
        # Position 0 for every memory row, as in the batched pass.
        rope_pos = torch.zeros(rows.shape[0], self.num_memory_tokens, device=rows.device, dtype=torch.long)
        all_keys: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []
        hidden = rows
        for block in blocks:
            normed, keys, values = block.token_kv(hidden)
            all_keys.append(keys)
            all_values.append(values)
            hidden = block.token_forward(hidden, normed, keys, values, attn_mask, q_pos=rope_pos, k_pos=rope_pos)
        return final_norm(hidden), all_keys, all_values

    def memory_readout(self, memory: torch.Tensor) -> torch.Tensor:
        """Rows ``0 .. M - 1`` of the ``H`` a pass with this ``Z`` ``[B, M, d]`` produces (what the writer queries)."""
        self._assert_memory_enabled()
        hidden, _, _ = self._memory_trunk(self._memory_token_input(memory.unsqueeze(1)))
        return hidden

    def _prefill_memory(self, env_ids: torch.Tensor) -> None:
        """Cache the per-layer K/V of ``env_ids``' memory rows and lay them down as rows ``0 .. M-1`` of ``H``."""
        if env_ids.numel() == 0:
            return
        with torch.no_grad():
            hidden, keys, values = self._memory_trunk(self._memory_token_input(self._memory[env_ids].unsqueeze(1)))
            for layer in range(self.num_layers):
                self._memory_key_cache[layer][env_ids] = keys[layer].detach()
                self._memory_value_cache[layer][env_ids] = values[layer].detach()
        self._hidden_history[env_ids, : self.num_memory_tokens] = hidden.detach().to(self._hidden_history.dtype)
        self._history_valid[env_ids, : self.num_memory_tokens] = True
        if self.critic_design == "separate_trunk":
            # The critic pathway reads the same ``Z`` through its own weights; its readouts are NOT part of ``H``.
            with torch.no_grad():
                rows = self._memory_token_input(self._memory[env_ids].unsqueeze(1), critic=True)
                _, keys, values = self._memory_trunk(rows, critic=True)
                for layer in range(self.num_layers):
                    self._critic_memory_key_cache[layer][env_ids] = keys[layer].detach()
                    self._critic_memory_value_cache[layer][env_ids] = values[layer].detach()

    # --------------------------------------------------------------------------------------------------------
    # Distribution / observation helpers
    # --------------------------------------------------------------------------------------------------------

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    def forward(self) -> NoReturn:
        raise NotImplementedError("Use act() / evaluate() / forward_step() / forward_window().")

    def _update_distribution(self, hidden: torch.Tensor) -> None:
        """Set the action distribution from ``h_t``. Shape-agnostic: ``[N, d]`` or ``[S, B, d]``."""
        self.distribution.update(self.actor(hidden), hidden)

    def update_distribution_from_hidden(self, hidden: torch.Tensor) -> EpisodeContextDistribution:
        self._update_distribution(hidden)
        return self.distribution

    def distribution_from_hidden(self, hidden: torch.Tensor) -> EpisodeContextDistribution:
        return self.update_distribution_from_hidden(hidden)

    def action_mean_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Policy mean from a hidden state produced by either forward path (what a BC loss regresses)."""
        return self.actor(hidden)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    def get_actor_obs(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["actor"]], dim=-1)

    def get_critic_obs(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["critic"]], dim=-1)

    def split_privileged(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``(policy part, privileged part)`` of a frame. The privileged part is ``None`` without the group."""
        if self.privileged_dim == 0:
            return obs, None
        return obs[..., : self.num_token_obs], obs[..., self.num_token_obs :]

    def normalize_actor_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize a raw frame; the privileged tail goes through its own normalizer. Same width in and out."""
        if self.privileged_dim == 0:
            return self.actor_obs_normalizer(obs)
        policy_obs, privileged_obs = self.split_privileged(obs)
        return torch.cat([self.actor_obs_normalizer(policy_obs), self.privileged_normalizer(privileged_obs)], dim=-1)

    @property
    def frame_normalizer(self) -> Any:
        """What normalizes a STORED frame (the storage holds the raw concat, normalized at collection time)."""
        return self.actor_obs_normalizer if self.privileged_dim == 0 else self.normalize_actor_obs

    def update_actor_normalization(self, obs: TensorDict | torch.Tensor) -> None:
        if self.actor_obs_normalization:
            frames = self.get_actor_obs(obs).reshape(-1, self.num_actor_obs)
            policy_obs, privileged_obs = self.split_privileged(frames)
            self.actor_obs_normalizer.update(policy_obs)
            if privileged_obs is not None:
                self.privileged_normalizer.update(privileged_obs)

    def update_critic_normalization(self, obs: TensorDict | torch.Tensor) -> None:
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs).reshape(-1, self.num_critic_obs))

    def update_normalization(self, obs: TensorDict | torch.Tensor) -> None:
        self.update_actor_normalization(obs)
        self.update_critic_normalization(obs)

    # --------------------------------------------------------------------------------------------------------
    # Token embedding + batched (update) path
    # --------------------------------------------------------------------------------------------------------

    def _pathway(self, critic: bool = False) -> tuple[Any, ...]:
        """``(token_embed, start_embed, memory_pos_embed, blocks, final_norm)`` of the requested pathway."""
        if not critic:
            return (
                self.token_embed,
                self.start_embed,
                getattr(self, "memory_pos_embed", None),
                self.blocks,
                self.final_norm,
            )
        if self.critic_design != "separate_trunk":
            raise RuntimeError("The critic token pathway only exists with critic_design='separate_trunk'.")
        return (
            self.critic_token_embed,
            self.critic_start_embed,
            getattr(self, "critic_memory_pos_embed", None),
            self.critic_blocks,
            self.critic_final_norm,
        )

    def _embed_tokens(
        self, obs: torch.Tensor, positions: torch.Tensor, normalize_obs: bool = True, critic: bool = False
    ) -> torch.Tensor:
        """``x = Embed(o) + [step == 0] * start_embed``. Both pathways share the one observation normalizer."""
        token_embed, start_embed, _, _, _ = self._pathway(critic)
        if normalize_obs:
            obs = self.normalize_actor_obs(obs)
        policy_obs, privileged_obs = self.split_privileged(obs)
        tokens = token_embed(policy_obs)
        if privileged_obs is not None:
            tokens = tokens + self.privileged_proj(self.privileged_encoder(privileged_obs))
        is_start = (positions == 0).unsqueeze(-1).to(tokens.dtype)
        return tokens + is_start * start_embed

    def _window_attn_mask(
        self,
        positions: torch.Tensor,
        key_valid: torch.Tensor | None = None,
        segments: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Episode-segmented causal mask ``[B, S, S]`` (time-major inputs), ``True`` == attend.

        Row ``i`` attends to ``j`` iff ``j <= i`` and ``i - j <= min(step_i, span - 1)``. Padded rows are
        allowed to attend to themselves so no softmax row is empty (NaN).
        """
        num_steps = positions.shape[0]
        device = positions.device
        rows = torch.arange(num_steps, device=device)
        distance = rows.unsqueeze(1) - rows.unsqueeze(0)  # [S, S], >= 0 in the causal half
        reach = positions.transpose(0, 1).clamp(max=self.context_span - 1).unsqueeze(-1)  # [B, S, 1]
        mask = (distance >= 0).unsqueeze(0) & (distance.unsqueeze(0) <= reach)
        if segments is not None:
            ids = segments.transpose(0, 1)  # [B, S]
            mask = mask & (ids.unsqueeze(2) == ids.unsqueeze(1))
        if key_valid is not None:
            mask = mask & key_valid.transpose(0, 1).unsqueeze(1)
        if key_valid is not None or segments is not None:
            eye = torch.eye(num_steps, dtype=torch.bool, device=device).unsqueeze(0)
            mask = mask | eye
        return mask

    def _forward_tokens(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor,
        normalize_obs: bool = True,
        key_valid: torch.Tensor | None = None,
        segments: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
        memory_segments: torch.Tensor | None = None,
        critic: bool = False,
    ) -> torch.Tensor:
        """One causal pass over ``[S, B, obs_dim]`` with per-row episode steps ``[S, B]``. Returns ``h [S, B, d]``.

        Memory rows (``[B, n_seg, M, d]``) are REAL rows laid out in front of the tokens; only token rows return.
        """
        _, _, _, blocks, final_norm = self._pathway(critic)
        num_steps = obs.shape[0]
        tokens = self._embed_tokens(obs, positions, normalize_obs, critic)
        hidden = tokens.transpose(0, 1)  # [B, S, d]
        attn_mask = self._window_attn_mask(positions, key_valid, segments)
        rope_pos = positions.transpose(0, 1)  # [B, S]
        num_memory_rows = 0
        if self.num_memory_tokens > 0:
            memory = self._prepare_memory(memory, hidden.shape[0], hidden.device)
            if critic:
                # The writer, ``z_init`` and the actor trunk that produced ``Z`` must never see a value gradient.
                memory = memory.detach()
            if memory_segments is None:
                memory_segments = self._segments_from_positions(positions)
            memory_rows = self._memory_token_input(memory, critic).to(hidden.dtype)
            num_memory_rows = memory_rows.shape[1]
            attn_mask = self._memory_prefix_mask(attn_mask, memory.shape[1], memory_segments.transpose(0, 1))
            hidden = torch.cat([memory_rows, hidden], dim=1)
            # Every memory row sits at position 0 (identity rotation): memory_pos_embed tells the rows apart.
            memory_pos = torch.zeros(hidden.shape[0], num_memory_rows, device=rope_pos.device, dtype=rope_pos.dtype)
            rope_pos = torch.cat([memory_pos, rope_pos], dim=1)
        for block in blocks:
            normed, keys, values = block.token_kv(hidden)
            hidden = block.token_forward(hidden, normed, keys, values, attn_mask, q_pos=rope_pos, k_pos=rope_pos)
        hidden = final_norm(hidden)
        assert hidden.shape[1] == num_memory_rows + num_steps
        return hidden[:, num_memory_rows:].transpose(0, 1)

    @staticmethod
    def _positions_from_segments(segments: torch.Tensor) -> torch.Tensor:
        """Position inside its (contiguous) segment for every entry of ``[B, S]`` segment ids."""
        batch_size, num_steps = segments.shape
        index = torch.arange(num_steps, device=segments.device).unsqueeze(0).expand(batch_size, num_steps)
        changed = torch.ones_like(segments, dtype=torch.bool)
        changed[:, 1:] = segments[:, 1:] != segments[:, :-1]
        segment_start = torch.where(changed, index, torch.zeros_like(index)).cummax(dim=1).values
        return index - segment_start

    def forward_window(
        self,
        obs: TensorDict | torch.Tensor,
        seg_mask: torch.Tensor | EpisodeContextPrefix | None = None,
        prefix: EpisodeContextPrefix | None = None,
        normalize_obs: bool = True,
        positions: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
        memory_segments: torch.Tensor | None = None,
        critic: bool = False,
    ) -> torch.Tensor:
        """Run a window of frames through the trunk in ONE causal pass.

        Batch-first (BC): ``forward_window(obs [B, S, obs_dim], seg_mask [B, S]) -> h [B, S, d]``; ``seg_mask``
        holds an integer segment id per position (``None`` = one segment starting at step 0).

        Time-major with a prefix (PPO update): ``forward_window(obs [W, B, ...], prefix=prefix) -> h [W, B, d]``.
        The window is normalized here; the prefix is already normalized. With a memory and ``memory=None`` the
        prefix's explicit ``memory`` is used, else its cached source episodes are written in-graph.
        """
        if isinstance(seg_mask, EpisodeContextPrefix):  # tolerate a positional prefix
            prefix, seg_mask = seg_mask, None
        assert self.num_memory_tokens > 0 or (memory is None and memory_segments is None), (
            "forward_window(memory=...) needs a policy built with memory_tokens > 0."
        )
        if prefix is None:
            sequence = self.get_actor_obs(obs)  # [B, S, obs_dim]
            batch_size, num_steps = sequence.shape[0], sequence.shape[1]
            segments = (
                torch.zeros(batch_size, num_steps, dtype=torch.long, device=sequence.device)
                if seg_mask is None
                else seg_mask.long()
            )
            if positions is None:
                positions = self._positions_from_segments(segments)
            hidden = self._forward_tokens(
                sequence.transpose(0, 1),
                positions.transpose(0, 1),
                normalize_obs,
                segments=segments.transpose(0, 1),
                memory=memory,
                memory_segments=None if memory_segments is None else memory_segments.transpose(0, 1),
                critic=critic,
            )
            return hidden.transpose(0, 1)

        window_obs = self.get_actor_obs(obs)
        if normalize_obs:
            window_obs = self.normalize_actor_obs(window_obs)
        num_prefix = prefix.obs.shape[0]
        if num_prefix > 0:
            sequence = torch.cat([prefix.obs, window_obs], dim=0)
            positions = torch.cat([prefix.positions, prefix.window_positions], dim=0)
        else:
            sequence, positions = window_obs, prefix.window_positions
        if memory is None:
            memory = prefix.memory if prefix.memory is not None else self.memory_from_prefix(prefix)
        hidden = self._forward_tokens(
            sequence,
            positions,
            normalize_obs=False,
            memory=memory,
            memory_segments=prefix.memory_segments if memory_segments is None else memory_segments,
            critic=critic,
        )
        return hidden[num_prefix:]

    def forward_sequence(
        self,
        obs: TensorDict | torch.Tensor,
        mask: torch.Tensor | None = None,
        normalize_obs: bool = True,
        positions: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
        critic: bool = False,
    ) -> torch.Tensor:
        """One whole episode (starting at step 0), ``[S, B, obs_dim]`` -> ``h [S, B, d]``. ``mask`` ``[S, B]`` is
        row validity; ``memory`` ``[B, M, d]`` the episode's ``Z`` (``None`` = ``z_init``)."""
        assert self.num_memory_tokens > 0 or memory is None, (
            "forward_sequence(memory=...) needs a policy built with memory_tokens > 0."
        )
        sequence = self.get_actor_obs(obs)
        num_steps, batch_size = sequence.shape[0], sequence.shape[1]
        if positions is None:
            positions = torch.arange(num_steps, device=sequence.device).unsqueeze(1).expand(num_steps, batch_size)
        key_valid = None if mask is None else mask.bool()
        return self._forward_tokens(sequence, positions, normalize_obs, key_valid, memory=memory, critic=critic)

    # --------------------------------------------------------------------------------------------------------
    # Incremental (collection) path
    # --------------------------------------------------------------------------------------------------------

    def forward_step(
        self,
        obs: TensorDict | torch.Tensor,
        commit: bool = True,
        normalize_obs: bool = True,
        critic: bool = False,
    ) -> torch.Tensor:
        """Advance the acting path by one frame and return ``h_t`` ``[num_envs, d]``. No host sync on the hot path.

        ``commit=False`` peeks (nothing enters the KV cache). ``critic=True`` runs the ``critic_*`` mirror; the
        step schedule is shared and only the actor's commit advances it, so run the critic step FIRST.
        """
        _, _, _, blocks, final_norm = self._pathway(critic)
        obs = self.get_actor_obs(obs)
        num_envs = obs.shape[0]
        self._ensure_state(num_envs, obs.device, obs.dtype)
        key_cache = self._critic_key_cache if critic else self._key_cache
        value_cache = self._critic_value_cache if critic else self._value_cache
        memory_key_cache = self._critic_memory_key_cache if critic else self._memory_key_cache
        memory_value_cache = self._critic_memory_value_cache if critic else self._memory_value_cache

        positions = self._positions
        tokens = self._embed_tokens(obs.unsqueeze(1), positions.unsqueeze(1), normalize_obs, critic)  # [N, 1, d]

        # Keys visible to this query: cached frames of this episode inside the span, plus itself. The freshness
        # test is what makes L < T correct.
        cache_positions = self._cache_positions
        fresh = (cache_positions >= 0) & (cache_positions + self.context_span > positions.unsqueeze(1))
        ones = torch.ones(num_envs, 1, device=obs.device, dtype=torch.bool)
        if self.num_memory_tokens > 0:
            memory_ones = torch.ones(num_envs, self.num_memory_tokens, device=obs.device, dtype=torch.bool)
            attn_mask = torch.cat([memory_ones, fresh, ones], dim=-1).unsqueeze(1)  # [N, 1, M + span + 1]
        else:
            attn_mask = torch.cat([fresh, ones], dim=-1).unsqueeze(1)  # [N, 1, span + 1]

        slots = torch.remainder(positions, self.context_span)
        scatter_index = slots.view(num_envs, 1, 1).expand(num_envs, 1, self.d_model)
        # The cache stores UN-rotated K/V; a slot is rotated by the position it holds at read time.
        step_pos = positions.view(num_envs, 1)
        key_pos = torch.cat([cache_positions.clamp(min=0), step_pos], dim=1)
        if self.num_memory_tokens > 0:
            memory_pos = torch.zeros(num_envs, self.num_memory_tokens, device=obs.device, dtype=torch.long)
            key_pos = torch.cat([memory_pos, key_pos], dim=1)
        hidden = tokens
        for layer, block in enumerate(blocks):
            normed, keys, values = block.token_kv(hidden)
            if self.num_memory_tokens > 0:
                all_keys = torch.cat([memory_key_cache[layer], key_cache[layer], keys], dim=1)
                all_values = torch.cat([memory_value_cache[layer], value_cache[layer], values], dim=1)
            else:
                all_keys = torch.cat([key_cache[layer], keys], dim=1)
                all_values = torch.cat([value_cache[layer], values], dim=1)
            hidden = block.token_forward(hidden, normed, all_keys, all_values, attn_mask, step_pos, key_pos)
            if commit:
                key_cache[layer].scatter_(1, scatter_index, keys.detach())
                value_cache[layer].scatter_(1, scatter_index, values.detach())
        hidden = final_norm(hidden).squeeze(1)

        if commit and critic:
            self._last_critic_hidden = hidden
            return hidden
        if commit:
            self._cache_positions.scatter_(1, slots.view(num_envs, 1), positions.view(num_envs, 1))
            with torch.inference_mode(False):
                self._positions = positions + 1
            self._last_hidden = hidden
            if self.num_memory_tokens > 0:
                # Detached: the source episode's trunk is not shaped by the write objective.
                self._append_hidden(hidden.detach(), positions)
        return hidden

    def act(
        self,
        obs: TensorDict | torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: EpisodeContextPrefix | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Collection (``hidden_state is None``, returns a sample) or update (prefix given, returns the mean)."""
        if hidden_state is None:
            if self.critic_design == "separate_trunk":
                # BEFORE the actor's commit: the shared step counters advance on the actor's commit only.
                self.forward_step(obs, commit=True, critic=True)
            hidden = self.forward_step(obs, commit=True)
            self._window_hidden, self._window_hidden_obs = None, None
            self._critic_window_hidden, self._critic_window_hidden_obs = None, None
            self._update_distribution(hidden)
            return self.distribution.sample()
        hidden = self.forward_window(obs, prefix=hidden_state)
        # Handed to the value head instead of re-running the trunk: ``evaluate`` on the same minibatch object
        # MUST see exactly the states the surrogate loss saw.
        self._window_hidden, self._window_hidden_obs = hidden, obs
        self._critic_window_hidden, self._critic_window_hidden_obs = None, None
        self._update_distribution(hidden)
        return self.distribution.mean

    def act_inference(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        hidden = self.forward_step(obs, commit=True)
        return self.actor(hidden)

    def value_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Value from a trunk readout (``critic_design="shared_trunk"`` only)."""
        if self.critic_design != "shared_trunk":
            raise RuntimeError(
                f"value_from_hidden() is meaningless with critic_design={self.critic_design!r}; call evaluate(obs)."
            )
        return self.critic(hidden.detach() if self.detach_critic_trunk else hidden)

    @property
    def next_state_enabled(self) -> bool:
        return self.next_state_head is not None

    @property
    def next_state_obs_dim(self) -> int:
        """Width of the observation part of the head's output (the rest is the privileged target group)."""
        return self.next_state_pred_end - self.next_state_pred_start

    def next_state_from_hidden(self, hidden: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predicted normalized ``obs_{t+1} - obs_t`` (+ target-group delta) from ``h_t`` and ``a_t``."""
        assert self.next_state_head is not None, (
            "next_state_from_hidden() needs next_state_head_hidden_dims; this policy has no auxiliary head."
        )
        return self.next_state_head(torch.cat([hidden, actions.to(dtype=hidden.dtype)], dim=-1))

    def _hidden_for_value(
        self,
        obs: TensorDict | torch.Tensor,
        hidden_state: EpisodeContextPrefix | None,
        use_cached_hidden: bool,
    ) -> torch.Tensor:
        """The ``h`` a shared-trunk value is read off: the minibatch's ``act()`` readout (identity-checked), the
        collection step's committed ``h_t``, or -- for the terminal bootstrap -- a peek that does not commit."""
        if use_cached_hidden and self._window_hidden is not None and obs is self._window_hidden_obs:
            return self._window_hidden
        if hidden_state is not None:
            return self.forward_window(obs, prefix=hidden_state)
        actor_obs = self.get_actor_obs(obs)
        if actor_obs.dim() != 2:
            raise RuntimeError(
                "A shared-trunk value for a batched window needs the window's context: call act() on the"
                " minibatch first, or pass the EpisodeContextPrefix as hidden_state."
            )
        if use_cached_hidden and self._last_hidden is not None and self._last_hidden.shape[0] == actor_obs.shape[0]:
            return self._last_hidden
        return self.forward_step(actor_obs, commit=False)

    def _critic_hidden_for_value(
        self,
        obs: TensorDict | torch.Tensor,
        hidden_state: EpisodeContextPrefix | None,
        use_cached_hidden: bool,
    ) -> torch.Tensor:
        """The ``critic_*`` pathway's readout for the same three call sites as :meth:`_hidden_for_value`."""
        if use_cached_hidden and self._critic_window_hidden is not None and obs is self._critic_window_hidden_obs:
            return self._critic_window_hidden
        if hidden_state is not None:
            hidden = self.forward_window(obs, prefix=hidden_state, critic=True)
            self._critic_window_hidden, self._critic_window_hidden_obs = hidden, obs
            return hidden
        actor_obs = self.get_actor_obs(obs)
        if actor_obs.dim() != 2:
            raise RuntimeError(
                "A separate-trunk value for a batched window needs the window's context: pass the"
                " EpisodeContextPrefix as hidden_state."
            )
        cached = self._last_critic_hidden
        if use_cached_hidden and cached is not None and cached.shape[0] == actor_obs.shape[0]:
            return cached
        return self.forward_step(actor_obs, commit=False, critic=True)

    def evaluate(
        self,
        obs: TensorDict | torch.Tensor,
        hidden_state: EpisodeContextPrefix | None = None,
        use_cached_hidden: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Value ``[N, ...] -> [N, 1]`` (collection) or ``[W, B, ...] -> [W, B, 1]`` (update)."""
        if self.critic_design == "privileged":
            critic_obs = self.get_critic_obs(obs)
            return self.critic(self.critic_obs_normalizer(critic_obs))
        if self.critic_design == "separate_trunk":
            return self.critic(self._critic_hidden_for_value(obs, hidden_state, use_cached_hidden))
        return self.value_from_hidden(self._hidden_for_value(obs, hidden_state, use_cached_hidden))

    # --------------------------------------------------------------------------------------------------------
    # Reset / PPO plumbing
    # --------------------------------------------------------------------------------------------------------

    def _append_hidden(self, hidden: torch.Tensor, positions: torch.Tensor) -> None:
        """Record ``h_t`` ``[N, d]`` as row ``M + step_t`` of the writer's view of the pass in progress."""
        slots = (positions + self.num_memory_tokens).clamp(max=self.hidden_history_span - 1).view(-1, 1)
        self._hidden_history.scatter_(1, slots.unsqueeze(-1).expand(-1, 1, self.d_model), hidden.unsqueeze(1))
        self._history_valid.scatter_(1, slots, torch.ones_like(slots, dtype=torch.bool))

    def _write_memory(self, env_ids: torch.Tensor) -> None:
        """Close the passes of ``env_ids``: ``Z <- G(H)``, detached. ``H`` is left for the caller to clear."""
        if env_ids.numel() == 0:
            return
        with torch.no_grad():
            new_memory, _ = self.write_memory(self._hidden_history[env_ids], mask=self._history_valid[env_ids])
        with torch.inference_mode(False):
            self._memory = self._memory.clone()
        self._memory[env_ids] = new_memory.detach().to(self._memory.dtype)
        self._episode_index[env_ids] += 1

    def _reset_trial(self, env_ids: torch.Tensor) -> None:
        """Restore ``Z`` to the learned ``z_init`` for ``env_ids``."""
        if env_ids.numel() == 0:
            return
        with torch.inference_mode(False):
            self._memory = self._memory.clone()
        self._memory[env_ids] = self.z_init.detach().to(device=self._memory.device, dtype=self._memory.dtype)
        self._episode_index[env_ids] = 0

    def reset(self, dones: torch.Tensor | None = None, trial_dones: torch.Tensor | None = None, **kwargs: Any) -> None:
        """Clear the KV cache of the environments that just ended an episode.

        With a memory a done also writes ``H`` into ``Z`` and a ``trial_dones`` entry restores ``z_init``;
        ``trial_dones=None`` means every episode is its own trial. ``dones=None`` wipes everything.
        """
        self._window_hidden, self._window_hidden_obs = None, None
        self._critic_window_hidden, self._critic_window_hidden_obs = None, None
        if self._key_cache is None:
            return
        if self.num_memory_tokens > 0:
            # Resolving ``dones`` to indices syncs the host, but runs the writer on the done envs only.
            done_ids = (
                torch.arange(self._num_envs, device=self._positions.device)
                if dones is None
                else dones.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
            )
            if dones is not None:
                self._write_memory(done_ids)
            trial_ids = done_ids if trial_dones is None else trial_dones.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
            self._reset_trial(trial_ids)
            if dones is None:
                with torch.inference_mode(False):
                    self._memory = self.z_init.detach().to(self._memory).unsqueeze(0).expand_as(self._memory).clone()
                self._episode_index.zero_()
                refresh = torch.arange(self._num_envs, device=self._positions.device)
            else:
                refresh = torch.cat([done_ids, trial_ids]).unique()
            self._hidden_history[refresh] = 0.0
            self._history_valid[refresh] = False
            self._prefill_memory(refresh)
        if dones is None:
            self._cache_positions.fill_(-1)
            self._positions.zero_()
            self._last_hidden = None
            self._last_critic_hidden = None
            return
        keep = (~dones.reshape(-1, 1).bool()).to(self._cache_positions.dtype)
        # Branch-free: a done env's slots go back to "empty" (-1) and its step counter to 0.
        self._cache_positions.mul_(keep).add_(keep - 1)
        self._positions.mul_(keep.reshape(-1))
        self._last_hidden = None
        self._last_critic_hidden = None

    def get_hidden_states(self) -> tuple[None, None]:
        """The acting state is a KV cache re-derived from raw frames at update time; nothing is stored per step."""
        return None, None

    # --------------------------------------------------------------------------------------------------------
    # Checkpointing
    # --------------------------------------------------------------------------------------------------------

    def _check_noise_std_compatible(self, state_dict: dict) -> None:
        expected_name = _NOISE_PARAM_NAME[self.noise_std_type]
        present = [name for name in ("std", "log_std") if name in state_dict]
        if not present:
            return
        if expected_name not in state_dict:
            raise ValueError(
                f"Checkpoint stores the action noise as '{present[0]}' but this policy was built with"
                f" noise_std_type='{self.noise_std_type}', which expects '{expected_name}'."
            )
        expected_shape = tuple(getattr(self, expected_name).shape)
        actual_shape = tuple(state_dict[expected_name].shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Checkpoint '{expected_name}' has shape {actual_shape} but noise_std_type="
                f"'{self.noise_std_type}' expects {expected_shape}. A 'gsde' checkpoint stores a"
                " [d_model, num_actions] matrix and a 'scalar'/'log' one a [num_actions] vector."
            )

    @staticmethod
    def _drop_learned_position_table(state_dict: dict) -> dict:
        """Strip a pre-RoPE checkpoint's learned ``pos_embed`` table, loudly. ``memory_pos_embed`` is kept."""
        stale = [key for key in state_dict if key.endswith("pos_embed") and not key.endswith("memory_pos_embed")]
        if not stale:
            return state_dict
        print(
            "[EpisodeContextModel] WARNING: checkpoint was trained with a LEARNED position table"
            f" ({', '.join(stale)}); this trunk uses RoPE and will NOT reproduce it. Dropping the table."
        )
        return {key: value for key, value in state_dict.items() if key not in stale}

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters. Returns ``True`` (the 3.1 "resumed training" convention)."""
        self._check_noise_std_compatible(state_dict)
        state_dict = self._drop_learned_position_table(state_dict)
        super().load_state_dict(state_dict, strict=strict)
        self._reset_runtime_state()
        return True


# ------------------------------------------------------------------------------------------------------------
# Actor / critic adapter views (the 5.2 ``Model`` interface over the joint module)
# ------------------------------------------------------------------------------------------------------------


class _EpisodeContextView(nn.Module):
    """Thin adapter presenting one side of :class:`EpisodeContextModel` through the 5.2 ``Model`` interface.

    The joint model is a registered child (so ``.to()`` / ``.train()`` reach it), but ``named_parameters`` /
    ``named_buffers`` are partitioned by name: the critic view owns every ``critic*`` tensor, the actor view the
    rest. The two views therefore chain to the joint parameter set exactly once (optimizer, broadcast, reduce).
    """

    is_recurrent: bool = True

    def __init__(self, model: EpisodeContextModel, obs_set: str) -> None:
        super().__init__()
        self.model = model
        self.obs_set = obs_set

    def _owns(self, name: str) -> bool:
        raise NotImplementedError

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):  # type: ignore
        for name, parameter in self.model.named_parameters(prefix, recurse, remove_duplicate):
            if self._owns(name[len(prefix) + 1 :] if prefix else name):
                yield name, parameter

    def named_buffers(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):  # type: ignore
        for name, buffer in self.model.named_buffers(prefix, recurse, remove_duplicate):
            if self._owns(name[len(prefix) + 1 :] if prefix else name):
                yield name, buffer

    @property
    def obs_groups(self) -> list[str]:
        return self.model.obs_groups[self.obs_set]

    @property
    def distribution(self) -> EpisodeContextDistribution:
        return self.model.distribution

    def reset(self, dones: torch.Tensor | None = None, hidden_state: Any = None) -> None:
        pass

    def get_hidden_state(self) -> None:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        return self.model.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.model.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.model.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.model.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.model.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        return self.model.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        raise NotImplementedError(
            "EpisodeContextModel has no TorchScript export: the KV-cached transformer is exported by the"
            " application-side tooling, not by rsl_rl."
        )

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        raise NotImplementedError("EpisodeContextModel has no ONNX export.")


class EpisodeContextActorView(_EpisodeContextView):
    """``actor`` adapter: ``forward(obs, stochastic_output=True)`` samples through the KV cache (or re-infers a
    window when ``hidden_state`` is a prefix); deterministic ``forward(obs)`` is the model's ``act_inference``."""

    def __init__(self, model: EpisodeContextModel) -> None:
        super().__init__(model, "actor")

    def _owns(self, name: str) -> bool:
        return not name.startswith("critic")

    @property
    def obs_dim(self) -> int:
        return self.model.num_actor_obs

    @property
    def obs_normalizer(self) -> nn.Module:
        return self.model.actor_obs_normalizer

    def forward(
        self,
        obs: TensorDict | torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: EpisodeContextPrefix | None = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        if stochastic_output or hidden_state is not None:
            return self.model.act(obs, masks=masks, hidden_state=hidden_state)
        return self.model.act_inference(obs)

    def update_normalization(self, obs: TensorDict | torch.Tensor) -> None:
        self.model.update_actor_normalization(obs)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: Any = None) -> None:
        self.model.reset(dones)


class EpisodeContextCriticView(_EpisodeContextView):
    """``critic`` adapter: ``forward(obs[, hidden_state=prefix])`` is :meth:`EpisodeContextModel.evaluate`.
    ``reset`` is a no-op (the actor view's reset clears both KV caches)."""

    def __init__(self, model: EpisodeContextModel) -> None:
        super().__init__(model, "critic")

    def _owns(self, name: str) -> bool:
        return name.startswith("critic")

    @property
    def obs_dim(self) -> int:
        return self.model.num_critic_obs

    @property
    def obs_normalizer(self) -> nn.Module:
        return self.model.critic_obs_normalizer

    def forward(
        self,
        obs: TensorDict | torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: EpisodeContextPrefix | None = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        return self.model.evaluate(obs, hidden_state=hidden_state)

    def update_normalization(self, obs: TensorDict | torch.Tensor) -> None:
        self.model.update_critic_normalization(obs)
