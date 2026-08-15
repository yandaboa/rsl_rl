# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-episode context transformer actor-critic.

The actor is a causal transformer over the frames of the **current episode only**: at control step ``t`` it
attends over ``x_{t-L+1} .. x_t`` (clipped at the episode start), where ``L`` is :attr:`context_length` and

.. code::

    x_t = Embed(o_t) + pos_embed[episode_step_t] + [episode_step_t == 0] * start_embed

There is no memory, no writer, no trials and no reward/done channel: the single-step policy observation already
carries the previous action, and the context never crosses an episode boundary. That makes the whole model a
function of ``(frames since the episode started, capped at L)`` -- which is exactly what the storage can
reconstruct from a ring buffer of raw frames, with no ``x L`` duplication of observations.

The **critic** has two designs, selected by ``critic_design``:

* ``"privileged"`` (default, the original behavior) -- a plain single-step MLP on the (privileged) critic
  observation group. It sees no context and no transformer, so values are produced step-wise by :meth:`evaluate`
  on both the collection and the update path.
* ``"shared_trunk"`` -- the design validated by run 225077 on the 16-frame line: the critic conditions on the
  SAME episode context as the actor. The value head is an MLP on the trunk readout ``h_t`` (``d_model`` in, 1
  out), sharing the actor's trunk, and the critic observation group is unused. Values therefore come from the
  acting path's ``h_t`` during collection and from the batched path's ``h`` during the update -- see
  :meth:`evaluate` for how each of the three call sites (collection, terminal bootstrap, minibatch) is served.
  By default the value loss backpropagates into the trunk (faithful to 225077); ``detach_critic_trunk=True``
  cuts that gradient.

Two forward paths, required to agree numerically:

* incremental (collection) -- :meth:`forward_step` / :meth:`act` / :meth:`act_inference`, one token per call,
  backed by a per-environment rolling KV cache of ``min(L, T)`` slots that is cleared per environment on ``done``,
* batched (update) -- :meth:`forward_window`, ONE causal pass over ``[prefix | window]`` per minibatch, with an
  episode-segmented attention mask. :meth:`forward_sequence` is the same machinery for a whole episode starting
  at step 0 (what a BC script wants).

Both paths build the same token from the same frame and attend over the same key set, so the PPO reconstruction
canary (ratio == 1 at epoch 0 with unchanged parameters) holds up to floating-point reduction order.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.modules.actor_critic import GSDENoiseDistribution, upcast_from_half
from rsl_rl.modules.actor_critic_trial_memory import TrunkBlock
from rsl_rl.networks import MLP, EmpiricalNormalization

# Which parameter holds the action noise, per ``noise_std_type``. A checkpoint written under one noise type
# cannot be loaded into another (the shapes differ: ``[A]`` vs ``[d_model, A]``).
_NOISE_PARAM_NAME = {"scalar": "std", "log": "log_std", "gsde": "log_std"}


@dataclass(frozen=True)
class EpisodeContextPrefix:
    """The history a minibatch of environments needs in order to re-infer its window in one causal pass.

    Rides in PPO's ``hidden_states`` slot: ``ppo.py`` treats it as opaque and hands it straight back to
    ``policy.act(..., hidden_state=...)``, which is what lets the stock (unmodified) recurrent update path drive
    this policy.

    Attributes:
        obs: Actor observations of the ``P`` frames **preceding** the window, oldest first, ``[P, B, obs_dim]``,
            **already normalized exactly as the acting path normalized them**. They are not re-normalized at
            update time: a prefix frame usually comes from an earlier rollout, i.e. from before the last
            normalizer commit, and re-normalizing it with today's statistics would build a different token than
            the one the behavior policy actually saw (a ~1e-2 log-prob bias in the epoch-0 ratio).
            :class:`~rsl_rl.storage.EpisodeContextRolloutStorage` stores the frames in that form.
        positions: Episode-step index of every prefix frame, ``[P, B]``.
        window_positions: Episode-step index of every window row, ``[W, B]``.
    """

    obs: torch.Tensor
    positions: torch.Tensor
    window_positions: torch.Tensor


class ActorCriticEpisodeContext(nn.Module):
    """Causal transformer actor over the current episode + plain MLP critic (see the module docstring)."""

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
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the episode-context actor-critic.

        Args:
            obs: A sample observation ``TensorDict``, used only to read the observation dimensions.
            obs_groups: Mapping from ``"policy"``/``"critic"`` to the observation groups they consume. With
                ``critic_design="privileged"`` the critic groups are genuinely used (the critic is a separate MLP
                and may be privileged); with ``critic_design="shared_trunk"`` they are ignored, so
                ``{"critic": ["policy"]}`` (or a missing/absent group) is tolerated.
            num_actions: Action dimension.
            actor_obs_normalization: Empirically normalize the single-step observation feeding the tokens.
            critic_obs_normalization: Empirically normalize the critic observation.
            context_length: ``L``, the maximum number of frames a step attends over (itself included), always
                clipped to the current episode. Values ``>= max_episode_length`` mean "the whole episode".
            d_model: Trunk width.
            num_layers: Number of trunk layers.
            num_heads: Number of attention heads.
            max_episode_length: ``T``, the number of control steps in one episode. Sizes the positional
                embedding table; episode steps beyond it are clamped (on BOTH forward paths, so they stay
                consistent).
            ff_mult: Feed-forward expansion factor.
            embed_hidden_dims: Hidden dims of the token embedding. Empty means a single linear layer.
            actor_hidden_dims: Hidden dims of the policy head (consumes the trunk readout ``h_t``).
            critic_hidden_dims: Hidden dims of the value MLP. It consumes the critic observation directly under
                ``critic_design="privileged"`` and the trunk readout ``h_t`` under ``"shared_trunk"``.
            activation: Activation used in the trunk and the heads.
            init_noise_std: Initial action noise standard deviation. Under ``"gsde"`` this is not the realized
                action std: the realized std is ``init_noise_std * ||h_t||``.
            noise_std_type: ``"scalar"``, ``"log"`` or ``"gsde"`` (the validated BC -> RL recipe uses gSDE, with
                ``log_std`` keyed on the trunk readout, exactly as in the trial-memory policy).
            normalizer_until: ``until`` of the observation normalizers (number of samples after which the
                statistics freeze). ``None`` keeps updating forever.
            critic_design: ``"privileged"`` (a separate single-step MLP on the critic observation group) or
                ``"shared_trunk"`` (a value head on the actor trunk's ``h_t``, i.e. the critic conditions on the
                same episode context as the actor and is NOT privileged).
            detach_critic_trunk: Only meaningful with ``critic_design="shared_trunk"``. When ``True``, ``h_t`` is
                detached before the value head, so the value loss trains the head alone. Defaults to ``False``
                (the value loss shapes the trunk too, as in run 225077).
        """
        if kwargs:
            print(
                "ActorCriticEpisodeContext.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        if critic_design not in ("privileged", "shared_trunk"):
            raise ValueError(f"Unknown critic_design: {critic_design!r}. Should be 'privileged' or 'shared_trunk'.")
        self.critic_design = critic_design
        self.detach_critic_trunk = bool(detach_critic_trunk)

        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticEpisodeContext module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        num_critic_obs = 0
        for obs_group in obs_groups.get("critic", obs_groups["policy"]):
            # With a shared trunk the critic groups are never read, so a configuration that points the critic at
            # the policy group (or at a group this environment does not even publish) is tolerated.
            if critic_design == "shared_trunk" and obs_group not in obs:
                continue
            assert len(obs[obs_group].shape) == 2, "The ActorCriticEpisodeContext module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_episode_length = int(max_episode_length)
        self.context_length = int(context_length)
        # How many frames of history can actually be reached: an episode is at most T frames long, so a context
        # longer than that is the same as the full episode. Everything downstream (the KV cache size, the
        # storage's ring buffer and the attention mask) is sized on this number, so the two forward paths agree
        # by construction.
        self.context_span = max(1, min(self.context_length, self.max_episode_length))

        # Token embedding: Embed(o_t). No action/reward/done channels -- the policy observation already carries
        # the previous action, and nothing is meant to flow across the episode boundary.
        if len(embed_hidden_dims) == 0:
            self.token_embed = nn.Linear(num_actor_obs, d_model)
        else:
            self.token_embed = MLP(num_actor_obs, d_model, list(embed_hidden_dims), activation)
        self.pos_embed = nn.Parameter(torch.zeros(self.max_episode_length, d_model))
        self.start_embed = nn.Parameter(torch.zeros(d_model))

        # Trunk
        ff_dim = ff_mult * d_model
        self.blocks = nn.ModuleList([TrunkBlock(d_model, num_heads, ff_dim, activation) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)

        # Heads. The actor reads the trunk; the critic reads either the trunk (shared) or the privileged critic
        # observation. The value head keeps the name ``critic`` in BOTH designs: downstream tooling keys on it
        # (uwlab's critic-warmup freezes every parameter NOT named ``critic*``/``_crit*``, and the BC strict-load
        # tolerance drops mismatched ``critic.*`` keys).
        self.actor = MLP(d_model, num_actions, list(actor_hidden_dims), activation)
        critic_input_dim = d_model if self.critic_design == "shared_trunk" else num_critic_obs
        self.critic = MLP(critic_input_dim, 1, list(critic_hidden_dims), activation)

        # Observation normalization. Attribute names match ActorCritic so that train.py's --freeze_obs_norm
        # handling (which pokes ``actor_obs_normalizer`` / ``critic_obs_normalizer``) works unchanged.
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs, until=normalizer_until)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        # A shared-trunk critic never touches the critic observation, so its normalizer would only accumulate
        # statistics nothing reads (and would crash ``update_normalization`` when the group does not exist).
        self.critic_obs_normalization = critic_obs_normalization and self.critic_design == "privileged"
        if self.critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs, until=normalizer_until)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # Action noise
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        elif noise_std_type == "gsde":
            # gSDE keys the noise on the trunk readout h_t, so log_std is [d_model, num_actions].
            self.log_std = nn.Parameter(torch.ones(d_model, num_actions) * math.log(init_noise_std))
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar', 'log' or 'gsde'")

        self.distribution = self._make_gsde_distribution() if noise_std_type == "gsde" else None
        Normal.set_default_validate_args(False)

        self._init_weights()
        self._reset_runtime_state()

        print(
            f"Episode-context trunk: L={num_layers} d={d_model} heads={num_heads}"
            f" context={self.context_length} (span {self.context_span}) T={self.max_episode_length}"
        )
        print(f"Actor head: {self.actor}")
        if self.critic_design == "shared_trunk":
            print(
                f"Critic head (shared trunk, detach={self.detach_critic_trunk}, conditions on the same episode"
                f" context as the actor): {self.critic}"
            )
        else:
            print(f"Critic MLP (privileged, {num_critic_obs}-d observation): {self.critic}")

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
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.start_embed, mean=0.0, std=0.02)

    _RUNTIME_STATE_ATTRS: tuple[str, ...] = (
        "_num_envs",
        "_key_cache",
        "_value_cache",
        "_cache_positions",
        "_positions",
        "_last_hidden",
        "_window_hidden",
        "_window_hidden_obs",
    )

    def _reset_runtime_state(self) -> None:
        """Drop every acting-time buffer (they are lazily re-allocated on the first :meth:`forward_step`)."""
        self._num_envs: int | None = None
        self._key_cache: list[torch.Tensor] | None = None
        self._value_cache: list[torch.Tensor] | None = None
        # Episode step stored in every cache slot, ``-1`` for an empty slot. Slot ``p % context_span`` holds
        # position ``p``, so a slot that has fallen out of the window is simply overwritten.
        self._cache_positions: torch.Tensor | None = None
        self._positions: torch.Tensor | None = None
        self._last_hidden: torch.Tensor | None = None
        # Update path (shared-trunk critic only): the ``h`` the last :meth:`act` computed for a minibatch, kept
        # together with the observation object it came from so that :meth:`evaluate` can prove it is looking at
        # the same minibatch before reusing it (``ppo.py`` calls ``act`` then ``evaluate`` on the same object).
        self._window_hidden: torch.Tensor | None = None
        self._window_hidden_obs: Any = None

    def initialize_state(self, num_envs: int, device: torch.device | str, dtype: torch.dtype | None = None) -> None:
        """Allocate the acting-time KV cache for ``num_envs`` environments (all episodes start at step 0)."""
        dtype = self.pos_embed.dtype if dtype is None else dtype
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

    def _ensure_state(self, num_envs: int, device: torch.device, dtype: torch.dtype) -> None:
        if self._num_envs != num_envs or self._key_cache is None or self._key_cache[0].device != device:
            self.initialize_state(num_envs, device, dtype)

    @property
    def positions(self) -> torch.Tensor | None:
        """Episode step of every environment on the acting path (``None`` before the first step)."""
        return self._positions

    @property
    def context_prefix_length(self) -> int:
        """Frames the batched path needs IN FRONT of a window to reproduce the acting path exactly.

        Not ``L - 1``: each block attends at most ``span - 1`` rows back, so after ``num_layers`` blocks a row's
        receptive field reaches ``num_layers * (span - 1)`` frames back (the standard depth argument -- block 2
        reads block 1's states, which themselves read another ``span - 1`` rows). Capped at ``T - 1`` because the
        mask can never cross an episode start, so a full episode of history is always enough.

        For the target configuration (``L = T = 80``) this is ``79``, i.e. the whole episode, which is what makes
        the storage's ring buffer ``num_steps_per_env + min(L, T)`` slots.
        """
        return min(self.max_episode_length - 1, self.num_layers * (self.context_span - 1))

    # --------------------------------------------------------------------------------------------------------
    # Distribution / observation helpers
    # --------------------------------------------------------------------------------------------------------

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def _make_gsde_distribution(self) -> GSDENoiseDistribution:
        """Build the gSDE distribution and draw its exploration matrix (see ActorCriticTrialMemory for why this
        is safe to call at any time: the matrix feeds only ``get_noise``, which this policy never calls)."""
        distribution = GSDENoiseDistribution(action_dim=self.num_actions)
        distribution.sample_weights(self.log_std)
        return distribution

    def _action_std(self, mean: torch.Tensor) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            return self.std.expand_as(mean)
        return torch.exp(self.log_std).expand_as(mean)

    def _update_distribution(self, hidden: torch.Tensor) -> None:
        """Set the action distribution from ``h_t``. Shape-agnostic: ``[N, d]`` or ``[S, B, d]``."""
        mean = self.actor(hidden)
        if self.noise_std_type == "gsde":
            self.distribution.proba_distribution(mean, self.log_std, hidden)
        else:
            self.distribution = Normal(upcast_from_half(mean), upcast_from_half(self._action_std(mean)))

    def update_distribution_from_hidden(self, hidden: torch.Tensor) -> Normal | GSDENoiseDistribution:
        """Set (and return) the action distribution from a hidden state produced by either forward path."""
        self._update_distribution(hidden)
        return self.distribution

    # Alias: the design spec (and the BC script) call it ``distribution_from_hidden``; the trial-memory policy
    # calls the same thing ``update_distribution_from_hidden``. Both names, one implementation.
    def distribution_from_hidden(self, hidden: torch.Tensor) -> Normal | GSDENoiseDistribution:
        """See :meth:`update_distribution_from_hidden`."""
        return self.update_distribution_from_hidden(hidden)

    def action_mean_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Policy mean from a hidden state produced by either forward path (what a BC loss regresses)."""
        return self.actor(hidden)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_actor_obs(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs
        return torch.cat([obs[obs_group] for obs_group in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs
        obs_groups = self.obs_groups.get("critic", self.obs_groups["policy"])
        return torch.cat([obs[obs_group] for obs_group in obs_groups], dim=-1)

    def update_normalization(self, obs: TensorDict | torch.Tensor) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs).reshape(-1, self.num_actor_obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs).reshape(-1, self.num_critic_obs))

    # --------------------------------------------------------------------------------------------------------
    # Token embedding + batched (update) path
    # --------------------------------------------------------------------------------------------------------

    def _embed_tokens(self, obs: torch.Tensor, positions: torch.Tensor, normalize_obs: bool = True) -> torch.Tensor:
        """``x = Embed(o) + pos_embed[step] + [step == 0] * start_embed``. All inputs share their leading dims."""
        if normalize_obs:
            obs = self.actor_obs_normalizer(obs)
        tokens = self.token_embed(obs)
        # Steps past the table are clamped rather than rejected, identically on both forward paths: an
        # environment that overruns T must not make the two disagree (nor crash a 16k-env run).
        clamped = positions.clamp(max=self.max_episode_length - 1)
        tokens = tokens + self.pos_embed[clamped]
        is_start = (positions == 0).unsqueeze(-1).to(tokens.dtype)
        return tokens + is_start * self.start_embed

    def _window_attn_mask(
        self,
        positions: torch.Tensor,
        key_valid: torch.Tensor | None = None,
        segments: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The episode-segmented causal mask, ``True`` == attend, ``[B, S, S]`` (time-major inputs).

        Row ``i`` attends to column ``j`` iff ``j <= i`` and ``i - j <= min(step_i, span - 1)``: causal, never
        further back than the start of ``i``'s own episode (its episode step *is* the distance to that start)
        and never further back than the context span. Because the sequence is one environment's contiguous
        stretch of time, "same episode" and "not more than ``step_i`` rows back" are the same statement;
        ``segments`` (explicit per-position segment ids) is accepted as a redundant, equivalent statement of it
        and is what makes right-padding inert without a separate validity mask.
        """
        num_steps = positions.shape[0]
        device = positions.device
        rows = torch.arange(num_steps, device=device)
        distance = rows.unsqueeze(1) - rows.unsqueeze(0)  # [S, S], >= 0 in the causal half
        # [B, S, 1]: how far back row i is allowed to look
        reach = positions.transpose(0, 1).clamp(max=self.context_span - 1).unsqueeze(-1)
        mask = (distance >= 0).unsqueeze(0) & (distance.unsqueeze(0) <= reach)
        if segments is not None:
            ids = segments.transpose(0, 1)  # [B, S]
            mask = mask & (ids.unsqueeze(2) == ids.unsqueeze(1))
        if key_valid is not None:
            mask = mask & key_valid.transpose(0, 1).unsqueeze(1)
        if key_valid is not None or segments is not None:
            # A fully masked query row would make softmax produce NaN and poison the whole batch through the
            # residual stream. Padded rows are meaningless anyway, so they are allowed to attend to themselves.
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
    ) -> torch.Tensor:
        """One causal pass over ``[S, B, obs_dim]`` with per-row episode steps ``[S, B]``. Returns ``h [S, B, d]``."""
        num_steps = obs.shape[0]
        tokens = self._embed_tokens(obs, positions, normalize_obs)
        hidden = tokens.transpose(0, 1)  # [B, S, d]
        attn_mask = self._window_attn_mask(positions, key_valid, segments)
        for block in self.blocks:
            normed, keys, values = block.token_kv(hidden)
            hidden = block.token_forward(hidden, normed, keys, values, attn_mask)
        hidden = self.final_norm(hidden)
        assert hidden.shape[1] == num_steps
        return hidden.transpose(0, 1)

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
    ) -> torch.Tensor:
        """Run a window of frames through the trunk in ONE causal pass. Two calling conventions:

        **Batch-first, segment ids** (what a BC script wants)::

            h = policy.forward_window(obs, seg_mask)      # obs [B, S, obs_dim] -> h [B, S, d_model]

        ``seg_mask`` ``[B, S]`` holds an integer **segment id** per position: positions attend only inside their
        own segment (and only ``context_length`` back), so right padding is inert as long as it carries an id of
        its own. ``None`` means "one segment", i.e. a single episode starting at step 0. The episode step of a
        position is its index inside its segment, so a segment must be contiguous; pass ``positions``
        explicitly if a segment starts before the slice.

        **Time-major with a prefix** (what the PPO update uses)::

            h = policy.forward_window(obs, prefix=prefix)  # obs [W, B, ...] -> h [W, B, d_model]

        ``obs`` is the window (a ``TensorDict`` slice straight out of the storage is fine) and ``prefix`` an
        :class:`EpisodeContextPrefix` carrying the ``P`` preceding frames and the episode steps of both parts.
        Only the ``W`` window rows are returned; the prefix exists purely to make them exact.
        """
        if isinstance(seg_mask, EpisodeContextPrefix):  # tolerate a positional prefix
            prefix, seg_mask = seg_mask, None
        if prefix is None:
            # Batch-first: [B, S, ...] in and out; the trunk itself is time-major.
            sequence = self.get_actor_obs(obs)  # [B, S, obs_dim]
            batch_size, num_steps = sequence.shape[0], sequence.shape[1]
            segments = (
                torch.zeros(batch_size, num_steps, dtype=torch.long, device=sequence.device)
                if seg_mask is None
                else seg_mask.long()
            )
            # ``positions`` (also [B, S]) only has to be passed when a segment starts before this slice.
            if positions is None:
                positions = self._positions_from_segments(segments)
            hidden = self._forward_tokens(
                sequence.transpose(0, 1), positions.transpose(0, 1), normalize_obs, segments=segments.transpose(0, 1)
            )
            return hidden.transpose(0, 1)

        # The window is normalized here (its rows were collected under the statistics that are in effect now,
        # which is what ``defer_obs_normalization`` guarantees); the prefix is already normalized -- see
        # :class:`EpisodeContextPrefix`.
        window_obs = self.get_actor_obs(obs)
        if normalize_obs:
            window_obs = self.actor_obs_normalizer(window_obs)
        num_prefix = prefix.obs.shape[0]
        if num_prefix > 0:
            sequence = torch.cat([prefix.obs, window_obs], dim=0)
            positions = torch.cat([prefix.positions, prefix.window_positions], dim=0)
        else:
            sequence, positions = window_obs, prefix.window_positions
        hidden = self._forward_tokens(sequence, positions, normalize_obs=False)
        return hidden[num_prefix:]

    def forward_sequence(
        self,
        obs: TensorDict | torch.Tensor,
        mask: torch.Tensor | None = None,
        normalize_obs: bool = True,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One whole episode (starting at step 0) in a single batched forward -- what a BC script wants.

        Args:
            obs: ``[S, B, obs_dim]`` (or a ``TensorDict`` with that leading shape), left-aligned at the episode
                start.
            mask: Validity of every row, ``[S, B]``. Invalid rows cannot influence any valid one.
            normalize_obs: Whether to apply the observation normalizer.
            positions: Optional explicit episode steps ``[S, B]``. Defaults to ``0, 1, ... S - 1``.

        Returns:
            ``h`` ``[S, B, d]``.
        """
        sequence = self.get_actor_obs(obs)
        num_steps, batch_size = sequence.shape[0], sequence.shape[1]
        if positions is None:
            positions = torch.arange(num_steps, device=sequence.device).unsqueeze(1).expand(num_steps, batch_size)
        key_valid = None if mask is None else mask.bool()
        return self._forward_tokens(sequence, positions, normalize_obs, key_valid)

    # --------------------------------------------------------------------------------------------------------
    # Incremental (collection) path
    # --------------------------------------------------------------------------------------------------------

    def forward_step(
        self, obs: TensorDict | torch.Tensor, commit: bool = True, normalize_obs: bool = True
    ) -> torch.Tensor:
        """Advance the acting path by one frame and return ``h_t`` ``[num_envs, d]``.

        No Python branch on any tensor value (no ``.item()`` / ``.any()``), so the hot path never syncs the GPU.
        With ``commit=False`` the frame is not written to the KV cache.
        """
        obs = self.get_actor_obs(obs)
        num_envs = obs.shape[0]
        self._ensure_state(num_envs, obs.device, obs.dtype)

        positions = self._positions
        tokens = self._embed_tokens(obs.unsqueeze(1), positions.unsqueeze(1), normalize_obs)  # [N, 1, d]

        # Keys visible to this query: every cached frame of this episode still inside the context span, plus
        # itself. Slot p % span holds position p, so a slot that has fallen out of the span has already been
        # overwritten; the freshness test is kept explicit anyway (it is what makes L < T correct).
        cache_positions = self._cache_positions
        fresh = (cache_positions >= 0) & (cache_positions + self.context_span > positions.unsqueeze(1))
        ones = torch.ones(num_envs, 1, device=obs.device, dtype=torch.bool)
        attn_mask = torch.cat([fresh, ones], dim=-1).unsqueeze(1)  # [N, 1, span + 1]

        slots = torch.remainder(positions, self.context_span)
        scatter_index = slots.view(num_envs, 1, 1).expand(num_envs, 1, self.d_model)
        hidden = tokens
        for layer, block in enumerate(self.blocks):
            normed, keys, values = block.token_kv(hidden)
            all_keys = torch.cat([self._key_cache[layer], keys], dim=1)
            all_values = torch.cat([self._value_cache[layer], values], dim=1)
            hidden = block.token_forward(hidden, normed, all_keys, all_values, attn_mask)
            if commit:
                self._key_cache[layer].scatter_(1, scatter_index, keys.detach())
                self._value_cache[layer].scatter_(1, scatter_index, values.detach())
        hidden = self.final_norm(hidden).squeeze(1)

        if commit:
            self._cache_positions.scatter_(1, slots.view(num_envs, 1), positions.view(num_envs, 1))
            self._positions = positions + 1
            self._last_hidden = hidden
        return hidden

    def act(
        self,
        obs: TensorDict | torch.Tensor,
        masks: torch.Tensor | None = None,
        hidden_state: EpisodeContextPrefix | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Collection (``hidden_state is None``) or update (``hidden_state`` is the minibatch's prefix).

        On the update path PPO only consumes the distribution this leaves behind (``action_mean``,
        ``action_std``, ``entropy``, ``get_actions_log_prob``), never the return value, so the action mean is
        returned instead of a fresh sample: sampling ``[W, B, A]`` would burn RNG and memory for nothing.
        """
        if hidden_state is None:
            hidden = self.forward_step(obs, commit=True)
            self._window_hidden, self._window_hidden_obs = None, None
            self._update_distribution(hidden)
            return self.distribution.sample()
        hidden = self.forward_window(obs, prefix=hidden_state)
        # Hand this ``h`` to the value head instead of re-running the trunk: with a shared trunk, ``ppo.py``'s
        # ``evaluate(obs_batch, ...)`` two lines below MUST see exactly the states the surrogate loss saw.
        self._window_hidden, self._window_hidden_obs = hidden, obs
        self._update_distribution(hidden)
        return self.distribution.mean

    def act_inference(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        hidden = self.forward_step(obs, commit=True)
        return self.actor(hidden)

    def value_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Value from a trunk readout produced by either forward path (``critic_design="shared_trunk"`` only)."""
        if self.critic_design != "shared_trunk":
            raise RuntimeError(
                "value_from_hidden() is meaningless with critic_design='privileged': the value head consumes the"
                " critic observation, not the trunk readout. Call evaluate(obs) instead."
            )
        return self.critic(hidden.detach() if self.detach_critic_trunk else hidden)

    def _hidden_for_value(
        self,
        obs: TensorDict | torch.Tensor,
        hidden_state: EpisodeContextPrefix | None,
        use_cached_hidden: bool,
    ) -> torch.Tensor:
        """The ``h`` a shared-trunk value must be read off, for each of the three call sites of :meth:`evaluate`.

        1. **Update minibatch.** ``ppo.py`` calls ``act(obs_batch, hidden_state=prefix)`` and then
           ``evaluate(obs_batch, hidden_state=None)`` on the *same* observation object, so the ``h`` :meth:`act`
           just computed is reused (identity-checked). That keeps the value and the surrogate on one trunk pass:
           one forward instead of two, and one shared autograd graph.
        2. **Collection step.** ``ppo.act`` likewise calls ``policy.act(obs)`` and then ``policy.evaluate(obs)``,
           so the committed ``h_t`` sits in :attr:`_last_hidden` -- which :meth:`reset` clears on every
           environment step, i.e. it is non-``None`` only inside that act -> evaluate window.
        3. **Terminal bootstrap.** ``ppo.compute_returns(obs)`` evaluates an observation that never went through
           :meth:`act` (and runs after :meth:`reset`, so nothing is cached): the frame is pushed through the
           trunk with ``commit=False``, i.e. it is *peeked* -- it must not enter the KV cache, since the next
           rollout's first token is the same frame and would otherwise see a duplicate of itself.
        """
        if use_cached_hidden and self._window_hidden is not None and obs is self._window_hidden_obs:
            return self._window_hidden
        if hidden_state is not None:
            return self.forward_window(obs, prefix=hidden_state)
        actor_obs = self.get_actor_obs(obs)
        if actor_obs.dim() != 2:
            # Refuse rather than fall through to the step path: a ``[W, B, obs]`` window has no incremental
            # meaning, and reusing ``_last_hidden`` here could silently return a ``[N, 1]`` value for it.
            raise RuntimeError(
                "A shared-trunk value for a batched window needs the window's context: call act() on the"
                " minibatch first (what ppo.py does), or pass the EpisodeContextPrefix as hidden_state."
            )
        if use_cached_hidden and self._last_hidden is not None and self._last_hidden.shape[0] == actor_obs.shape[0]:
            return self._last_hidden
        return self.forward_step(actor_obs, commit=False)

    def evaluate(
        self,
        obs: TensorDict | torch.Tensor,
        hidden_state: EpisodeContextPrefix | None = None,
        use_cached_hidden: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Value of the given observation, ``[N, ...] -> [N, 1]`` (collection) or ``[W, B, ...] -> [W, B, 1]``
        (update).

        With ``critic_design="privileged"`` this is a context-free MLP on the (privileged) critic observation.
        With ``"shared_trunk"`` the value is read off the SAME episode-context readout ``h`` the actor uses; see
        :meth:`_hidden_for_value` for where that ``h`` comes from on each call site. ``use_cached_hidden=False``
        forces a fresh trunk pass instead of reusing the one :meth:`act` left behind.
        """
        if self.critic_design == "privileged":
            critic_obs = self.get_critic_obs(obs)
            return self.critic(self.critic_obs_normalizer(critic_obs))
        return self.value_from_hidden(self._hidden_for_value(obs, hidden_state, use_cached_hidden))

    # --------------------------------------------------------------------------------------------------------
    # Reset / PPO plumbing
    # --------------------------------------------------------------------------------------------------------

    def reset(self, dones: torch.Tensor | None = None, **kwargs: Any) -> None:
        """Clear the KV cache of the environments that just ended an episode (the context never crosses a done)."""
        self._window_hidden, self._window_hidden_obs = None, None
        if self._key_cache is None:
            return
        if dones is None:
            self._cache_positions.fill_(-1)
            self._positions.zero_()
            self._last_hidden = None
            return
        keep = (~dones.reshape(-1, 1).bool()).to(self._cache_positions.dtype)
        # Branch-free (no host sync): a done environment's slots go back to "empty" and its step counter to 0.
        # The cached keys/values themselves are left alone -- they are unreachable while _cache_positions is -1
        # and are overwritten as the new episode fills the ring.
        self._cache_positions.mul_(keep).add_(keep - 1)
        self._positions.mul_(keep.reshape(-1))
        self._last_hidden = None

    def get_hidden_states(self) -> tuple[None, None]:
        """The acting state is a KV cache, not a hidden state: it is never stored per step (it is re-derived
        from the raw frames at update time), so PPO is handed nothing to save."""
        return None, None

    # --------------------------------------------------------------------------------------------------------
    # Checkpointing
    # --------------------------------------------------------------------------------------------------------

    def _check_noise_std_compatible(self, state_dict: dict) -> None:
        """Fail loudly when a checkpoint's noise parameterization does not match ``noise_std_type``."""
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

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model. Returns whether this resumes a previous training."""
        self._check_noise_std_compatible(state_dict)
        super().load_state_dict(state_dict, strict=strict)
        self._reset_runtime_state()
        if self.noise_std_type == "gsde":
            self.distribution = self._make_gsde_distribution()
        return True
