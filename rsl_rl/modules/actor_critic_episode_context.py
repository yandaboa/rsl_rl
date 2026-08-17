# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-episode context transformer actor-critic.

The actor is a causal transformer over the frames of the **current episode only**: at control step ``t`` it
attends over ``x_{t-L+1} .. x_t`` (clipped at the episode start), where ``L`` is :attr:`context_length` and

.. code::

    x_t = Embed(o_t) + pos_embed[episode_step_t] + [episode_step_t == 0] * start_embed

There is no reward/done channel: the single-step policy observation already carries the previous action, and the
context never crosses an episode boundary. That makes the whole model a function of ``(frames since the episode
started, capped at L)`` -- which is exactly what the storage can reconstruct from a ring buffer of raw frames,
with no ``x L`` duplication of observations.

``memory_tokens > 0`` adds an OPTIONAL cross-episode memory on top of that, and nothing else about the model
changes (with the default ``memory_tokens=0`` not even a submodule is built):

* ``M`` persistent rows ``Z`` per environment, initialized from a learned ``z_init`` at every **trial** start,
* the memory enters the model as ``M`` **prefix tokens of the trunk's own sequence**: the pass a step runs is
  ``[Z_0 + memory_pos_embed[0], ..., Z_{M-1} + memory_pos_embed[M-1], x_0, ..., x_t]``, memory row ``i``
  attending causally to memory rows ``<= i`` and every environment token attending to all ``M`` memory rows on
  top of its usual (episode-clipped, span-clipped) reach. There is no separate read layer and no gate: the
  memory is read by every block, exactly like any other token,
* a :class:`MemoryTokenWriter` ``G`` run at every **episode** end, ``Z <- z_init + delta(H)``, where ``H`` is the
  (detached) snapshot of the trunk readouts of the pass that just finished -- the ``M`` memory rows FIRST, then
  the episode's ``h_t``. The writer queries the memory rows and attends them over the whole of ``H`` (itself
  included), i.e. one combined self+cross attention, so the recurrence ``Z_{e+1} = G(trunk(Z_e), episode_e)``
  is carried entirely by that snapshot. The write is anchored on ``z_init`` and ``delta`` is exactly zero at
  init, so a freshly built policy runs episode 2 of a trial exactly as it runs episode 1.

The episode context itself still never crosses a done; the memory is the only thing that does. Note that the
memory is NOT an identity at init *for the trunk*: prepending ``M`` rows to the sequence perturbs a BC'd trunk
from the very first step (``z_init`` and ``memory_pos_embed`` are initialized small, ``std = 0.02``, so the
perturbation is small -- how small is measured by the closed-loop gate, not asserted here). What IS exact at
init is the write itself: episode 2 of a trial sees the same ``Z`` episode 1 did.

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
from rsl_rl.modules.actor_critic_trial_memory import MultiHeadAttention, TrunkBlock
from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation

# Which parameter holds the action noise, per ``noise_std_type``. A checkpoint written under one noise type
# cannot be loaded into another (the shapes differ: ``[A]`` vs ``[d_model, A]``).
_NOISE_PARAM_NAME = {"scalar": "std", "log": "log_std", "gsde": "log_std"}


class MemoryTokenWriter(nn.Module):
    """The writer ``G``: ``Z_new = anchor + delta(H)``, one combined self+cross attention over the trunk's output.

    ``H`` ``[B, M + T, d]`` is the snapshot of one finished pass: the ``M`` memory rows the pass was prefixed
    with, followed by the episode's per-step readouts. The queries are the memory rows of ``H`` -- so the writer
    attends the memory *to itself and to the episode that just ran*, in one attention::

        A     = MHA(LN_q(H[:, :M]), proj(LN_kv(H)))   # ALL of H as keys/values, masked by validity
        U     = H[:, :M] + A
        delta = FF(LN_ff(U))                          # FF's final linear (weight AND bias) is zero-initialized
        Z     = anchor + delta                        # anchor = the policy's learned z_init

    That single ``M x (M + T)`` attention is the whole recurrence: ``Z_{e+1}`` depends on ``Z_e`` through the
    trunk outputs of ``Z_e``'s rows, which is exactly what the (detached, cached) ``H`` carries at update time.

    The write is anchored on ``z_init``, not on ``H[:, :M]``: ``Z`` is consumed as an INPUT token of the next
    pass, and a post-``final_norm`` trunk row has norm ~``sqrt(d)`` -- ~58x the scale of ``z_init`` and of every
    input token a BC'd trunk has ever seen. Because ``delta`` is exactly zero at initialization,
    ``Z_new == anchor`` bit-for-bit and episode 2 of a trial starts out identical to episode 1; the optimizer
    grows ``delta`` as the memory earns its influence. The anchor stays differentiable, so a written segment
    trains ``z_init`` as well.
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
        """Zero the FF's output layer, i.e. make the write an exact no-op relative to its anchor.

        The owning policy re-applies this AFTER its own global initialization sweep, which would otherwise
        overwrite these zeros with the GPT-style normal init.
        """
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
        """Write one finished pass into the memory.

        Args:
            hidden: ``H`` ``[B, M + S, d]``: the memory rows first, then the episode's readouts.
            num_memory: ``M``, how many leading rows of ``hidden`` are the memory.
            anchor: The learned "no memory yet" rows the write is relative to, ``[M, d]`` or ``[B, M, d]`` (the
                policy's ``z_init``). Passed in rather than owned here: the writer stays stateless.
            mask: Validity of every row of ``H``, ``[B, M + S]``. Rows that are ``False`` cannot influence ``Z``.
                The memory rows themselves are always valid (they are written by the prefill).
            need_weights: Whether to also return the ``[B, num_heads, M, M + S]`` attention weights.

        Returns:
            ``Z_new`` ``[B, M, d]`` and, if requested, the attention weights.
        """
        queries = hidden[:, :num_memory]
        keys, values = self.attn.project_kv(self.norm_key(hidden))

        if mask is not None:
            key_mask = mask.bool().clone()
            # The memory rows are always readable: it is what keeps the softmax rows non-empty (a fully masked
            # row is NaN) and it is what the acting path does anyway.
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
        memory: Optional per-segment memory ``[B, n_seg, M, d]`` (``memory_tokens > 0`` only), READY TO USE. It
            is what a caller that already holds a ``Z`` passes; the storage does not use it (it ships the source
            episodes instead, below, so that the writer runs in-graph). ``None`` means ``z_init`` for every row,
            which is what the first episode of a trial gets anyway.
        memory_segments: Optional index into the segment axis of ``memory`` / :attr:`source_hidden` for every row
            of ``[prefix | window]``, ``[P + W, B]``. ``None`` derives it from the episode steps (a row with step
            0 opens a segment) -- which the storage never relies on: it computes the segments explicitly from the
            episode boundaries it recorded and passes them here.
        source_hidden: The cached, DETACHED ``H`` of the episode that precedes every segment,
            ``[B, n_seg, M + T, d]`` (memory rows first, then the episode's readouts), zero-padded for segments
            that have none. The update path turns this into ``memory`` in-graph (``Z = G(H)``) -- see
            :meth:`ActorCriticEpisodeContext.memory_from_prefix`. Ignored when :attr:`memory` is given.
        source_valid: Validity of every row of :attr:`source_hidden`, ``[B, n_seg, M + T]``.
        segment_has_source: Whether a segment has a source episode at all, ``[B, n_seg]``. ``False`` means the
            segment is episode 0 of its trial and reads ``z_init``.
    """

    obs: torch.Tensor
    positions: torch.Tensor
    window_positions: torch.Tensor
    memory: torch.Tensor | None = None
    memory_segments: torch.Tensor | None = None
    source_hidden: torch.Tensor | None = None
    source_valid: torch.Tensor | None = None
    segment_has_source: torch.Tensor | None = None


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
        memory_tokens: int = 0,
        episodes_per_trial: int = 2,
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
            memory_tokens: ``M``, the number of cross-episode memory rows PREPENDED to the trunk's sequence.
                ``0`` (the default) builds no memory submodule at all and leaves every code path exactly as it
                is without this feature.
            episodes_per_trial: ``K``, bookkeeping only: it is what :attr:`episode_index_in_trial` counts up to,
                and what a caller that does not track trials itself can use to decide when to pass
                ``trial_dones``. The writer recurrence itself is general in ``K``.
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

        # Cross-episode memory (optional). Nothing below exists with ``memory_tokens=0``, so the module is then
        # parameter-for-parameter and bit-for-bit the memory-free policy.
        self.num_memory_tokens = int(memory_tokens)
        self.episodes_per_trial = int(episodes_per_trial)
        # One row per memory token plus one per step of a full episode: what the writer keys on, and exactly the
        # sequence a step of that episode ran over.
        self.hidden_history_span = self.num_memory_tokens + self.max_episode_length
        if self.num_memory_tokens > 0:
            self.z_init = nn.Parameter(torch.zeros(self.num_memory_tokens, d_model))
            self.memory_pos_embed = nn.Parameter(torch.zeros(self.num_memory_tokens, d_model))
            self.writer = MemoryTokenWriter(d_model, num_heads, ff_dim, activation)

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
            + (
                f" M={self.num_memory_tokens} memory tokens prepended to the sequence (K={self.episodes_per_trial})"
                if self.num_memory_tokens > 0
                else " (no memory)"
            )
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
        if self.num_memory_tokens > 0:
            self.writer.attn.out_proj.weight.data.mul_(residual_scale)
            # ... but the writer's FF output goes back to exactly zero: ``Z_new = z_init + delta`` with
            # ``delta == 0`` at init, so a written memory starts out indistinguishable from ``z_init``.
            self.writer.zero_delta_init()
            # Small, like every other embedding: the memory rows are ordinary tokens of the trunk's sequence, so
            # this is what keeps their step-0 perturbation of a BC'd trunk small (it is NOT zero -- there is no
            # identity-at-init in this design).
            nn.init.normal_(self.memory_pos_embed, mean=0.0, std=0.02)
            nn.init.normal_(self.z_init, mean=0.0, std=0.02)

    _RUNTIME_STATE_ATTRS: tuple[str, ...] = (
        "_num_envs",
        "_key_cache",
        "_value_cache",
        "_cache_positions",
        "_positions",
        "_last_hidden",
        "_window_hidden",
        "_window_hidden_obs",
        "_memory",
        "_memory_key_cache",
        "_memory_value_cache",
        "_hidden_history",
        "_history_valid",
        "_episode_index",
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
        # Memory state (all ``None`` with ``memory_tokens=0``): the per-environment ``Z``, the per-layer K/V of
        # its ``M`` prefix rows (refreshed by the prefill at every write, never evicted by the episode ring),
        # the readouts of the pass in progress + their validity, and how many episodes of the current trial are
        # already written into ``Z``.
        self._memory: torch.Tensor | None = None
        self._memory_key_cache: list[torch.Tensor] | None = None
        self._memory_value_cache: list[torch.Tensor] | None = None
        self._hidden_history: torch.Tensor | None = None
        self._history_valid: torch.Tensor | None = None
        self._episode_index: torch.Tensor | None = None

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
            # Every episode opens with the memory rows already in the sequence, so their K/V (and their rows of
            # ``H``) have to exist before the first environment token is stepped.
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
    # Cross-episode memory
    # --------------------------------------------------------------------------------------------------------

    @property
    def memory(self) -> torch.Tensor | None:
        """The acting path's ``Z`` ``[num_envs, M, d]`` (``None`` before the first step, or with no memory)."""
        return self._memory

    @property
    def episode_index_in_trial(self) -> torch.Tensor | None:
        """How many episodes of the current trial are already written into ``Z``, per environment ``[num_envs]``."""
        return self._episode_index

    def initial_memory(self, batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
        """The learned "no memory yet" rows ``[B, M, d]``. Differentiable, so an episode-0 loss reaches ``z_init``."""
        self._assert_memory_enabled()
        memory = self.z_init.unsqueeze(0).expand(batch_size, -1, -1)
        return memory if device is None else memory.to(device)

    def write_memory(
        self,
        hidden: torch.Tensor,
        mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the writer ``G``: ``Z_{e+1} = z_init + delta(H_e)``. See :class:`MemoryTokenWriter` for the shapes.

        ``H_e`` ``[B, M + S, d]`` is one finished pass -- the ``M`` memory rows first, then the episode's
        readouts. This is what the update path calls **in graph** on the cached (detached) ``H`` of the source
        episode, so that the writer gets a gradient while the source episode's trunk does not. The anchor is the
        DIFFERENTIABLE ``z_init``, so a written segment trains it too.
        """
        self._assert_memory_enabled()
        return self.writer(hidden, self.num_memory_tokens, self.z_init, mask=mask, need_weights=need_weights)

    def memory_from_prefix(self, prefix: EpisodeContextPrefix) -> torch.Tensor | None:
        """Turn a minibatch's cached source episodes into the per-segment ``Z`` ``[B, n_seg, M, d]``, IN GRAPH.

        This is the update-path counterpart of the acting path's :meth:`_write_memory`, and the only place the
        writer runs with a gradient::

            Z_seg = G(H_seg)   for a segment that has a source episode
            Z_seg = z_init     for episode 0 of a trial

        ``H_seg`` is the detached snapshot the storage took when that episode ended -- and because the memory
        rows of that pass are its first ``M`` rows, the whole recurrence ``Z_{e+1} = G(trunk(Z_e), episode_e)``
        rides in it. The gradient therefore trains the writer, and ``z_init`` through BOTH kinds of segment (the
        source-free ones read it directly, the written ones through the writer's anchor), but never the source
        episode's trunk. The writer is batched over the ``B x n_seg`` segments that actually have a source and
        the result is scattered back, so the padded ones cost nothing.

        Note: the chain is one write off ``z_init``, i.e. it reproduces the acting path exactly for
        ``episodes_per_trial = 2`` (the validated configuration). With ``K > 2`` a segment of episode ``e > 1``
        would need ``G`` applied ``e`` times down the trial, which the cached-``H`` design deliberately does not
        keep; the memory is then an approximation of the acting one.

        Returns:
            ``[B, n_seg, M, d]``, or ``None`` when the prefix carries no source data.
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
        """``H`` of the pass in progress plus its validity mask, ``[n, M + T, d]`` and ``[n, M + T]``, detached.

        Rows ``0 .. M - 1`` are the memory tokens the episode opened with (written by :meth:`_prefill_memory`),
        rows ``M + t`` the readout of environment step ``t``. Snapshotting this at an episode boundary (before
        :meth:`reset` clears it) is how the storage keeps the source episode of the next one without holding on
        to its raw frames.
        """
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

    def _memory_token_input(self, memory: torch.Tensor) -> torch.Tensor:
        """The trunk's INPUT rows for a memory ``[B, n_seg, M, d]``: ``Z_i + memory_pos_embed[i]``, flattened.

        No ``start_embed`` and no episode positional embedding -- a memory row is not a frame of the episode, it
        is its own kind of token and :attr:`memory_pos_embed` is the only thing that tells the rows apart.
        """
        batch_size, num_segments = memory.shape[0], memory.shape[1]
        return (memory + self.memory_pos_embed).reshape(batch_size, num_segments * self.num_memory_tokens, -1)

    def _memory_prefix_mask(self, token_mask: torch.Tensor, num_segments: int, segments: torch.Tensor) -> torch.Tensor:
        """Grow a token-only mask ``[B, S, S]`` into the full ``[B, nM + S, nM + S]`` of ``[memory | tokens]``.

        The ``n_seg`` memories of a pass are laid out FIRST, block by block, and

        * memory row ``i`` of block ``k`` attends to rows ``<= i`` of block ``k`` and to nothing else (so a block
          is exactly the prefill the acting path runs, and blocks cannot see each other),
        * an environment token attends to the whole memory block of ITS segment, plus its usual token reach
          (``segments`` ``[B, S]`` says which block that is; it is clamped so that a token can never end up with
          an empty key set, which would make softmax produce NaN).

        Environment tokens never feed back into the memory rows, which is what makes the ``M`` prefix rows of one
        batched pass reproducible by a small ``M``-row prefill on the acting path.
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

    def _memory_trunk(self, rows: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Push the ``M`` memory INPUT rows ``[B, M, d]`` through the trunk on their own.

        The memory rows attend causally to each other and to nothing else, so this is exactly what a full pass
        would do to them -- which is what lets the acting path replace the memory prefix of the sequence by a
        cached K/V per layer.

        Returns:
            Their readouts ``[B, M, d]`` (rows ``0 .. M - 1`` of ``H``) and the per-layer keys and values.
        """
        index = torch.arange(self.num_memory_tokens, device=rows.device)
        attn_mask = (index.unsqueeze(1) >= index.unsqueeze(0)).unsqueeze(0)
        all_keys: list[torch.Tensor] = []
        all_values: list[torch.Tensor] = []
        hidden = rows
        for block in self.blocks:
            normed, keys, values = block.token_kv(hidden)
            all_keys.append(keys)
            all_values.append(values)
            hidden = block.token_forward(hidden, normed, keys, values, attn_mask)
        return self.final_norm(hidden), all_keys, all_values

    def memory_readout(self, memory: torch.Tensor) -> torch.Tensor:
        """Rows ``0 .. M - 1`` of the ``H`` a pass with this ``Z`` ``[B, M, d]`` produces -- what the writer queries.

        Differentiable, and the exact counterpart of what :meth:`_prefill_memory` records on the acting path: a
        caller that reconstructs a pass from scratch (a BC script, a test) builds its ``H`` as
        ``cat([memory_readout(Z), h_0, ..., h_{T-1}])``.
        """
        self._assert_memory_enabled()
        hidden, _, _ = self._memory_trunk(self._memory_token_input(memory.unsqueeze(1)))
        return hidden

    def _prefill_memory(self, env_ids: torch.Tensor) -> None:
        """Run ``env_ids``' ``M`` memory rows through the trunk and cache their per-layer K/V + their ``H`` rows.

        The acting path's counterpart of the memory block a batched pass carries in front of its window: because
        the memory rows only ever attend to each other, one ``M``-row pass at the start of an episode produces
        exactly the keys and values every later step of that episode has to attend over, and they are kept in
        their own (never-evicted) cache instead of in the episode ring.
        """
        if env_ids.numel() == 0:
            return
        with torch.no_grad():
            hidden, keys, values = self._memory_trunk(self._memory_token_input(self._memory[env_ids].unsqueeze(1)))
            for layer in range(self.num_layers):
                self._memory_key_cache[layer][env_ids] = keys[layer].detach()
                self._memory_value_cache[layer][env_ids] = values[layer].detach()
        # The memory rows are the first M rows of H, exactly as the writer (and the storage) expect them.
        self._hidden_history[env_ids, : self.num_memory_tokens] = hidden.detach().to(self._hidden_history.dtype)
        self._history_valid[env_ids, : self.num_memory_tokens] = True

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
        memory: torch.Tensor | None = None,
        memory_segments: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One causal pass over ``[S, B, obs_dim]`` with per-row episode steps ``[S, B]``. Returns ``h [S, B, d]``.

        ``memory`` ``[B, n_seg, M, d]`` and ``memory_segments`` ``[S, B]`` (both optional, memory only) say which
        ``M`` rows each token attends over. The memory rows are REAL rows of this pass: they are laid out in
        front of the tokens (``[mem_seg_0 | ... | mem_seg_{n-1} | tokens]``), run through every block, and only
        the token rows are returned. The trunk therefore processes -- and is trained through -- the memory.
        """
        num_steps = obs.shape[0]
        tokens = self._embed_tokens(obs, positions, normalize_obs)
        hidden = tokens.transpose(0, 1)  # [B, S, d]
        attn_mask = self._window_attn_mask(positions, key_valid, segments)
        num_memory_rows = 0
        if self.num_memory_tokens > 0:
            memory = self._prepare_memory(memory, hidden.shape[0], hidden.device)
            if memory_segments is None:
                memory_segments = self._segments_from_positions(positions)
            memory_rows = self._memory_token_input(memory).to(hidden.dtype)
            num_memory_rows = memory_rows.shape[1]
            attn_mask = self._memory_prefix_mask(attn_mask, memory.shape[1], memory_segments.transpose(0, 1))
            hidden = torch.cat([memory_rows, hidden], dim=1)
        for block in self.blocks:
            normed, keys, values = block.token_kv(hidden)
            hidden = block.token_forward(hidden, normed, keys, values, attn_mask)
        hidden = self.final_norm(hidden)
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

        **Memory** (``memory_tokens > 0`` only). ``memory`` ``[B, n_seg, M, d]`` holds one ``Z`` per episode
        segment of the pass and ``memory_segments`` says which of them every token reads: ``[B, S]`` batch-first,
        ``[P + W, B]`` (time-major, aligned with the concatenated positions) with a prefix. Every segment's ``M``
        rows are prepended to the pass as real tokens and a token attends ONLY to its own segment's block, which
        is what lets one minibatch mix rows from before and after an episode boundary.
        ``memory=None`` means ``z_init`` everywhere; ``memory_segments=None``
        derives the segments from the episode steps. With a prefix, ``memory=None`` first looks at the prefix: an
        explicit ``prefix.memory``, else the cached source episodes it carries, which are written into a ``Z``
        in-graph by :meth:`memory_from_prefix` -- the PPO update path.
        """
        if isinstance(seg_mask, EpisodeContextPrefix):  # tolerate a positional prefix
            prefix, seg_mask = seg_mask, None
        assert self.num_memory_tokens > 0 or (memory is None and memory_segments is None), (
            "forward_window(memory=...) needs a policy built with memory_tokens > 0."
        )
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
                sequence.transpose(0, 1),
                positions.transpose(0, 1),
                normalize_obs,
                segments=segments.transpose(0, 1),
                memory=memory,
                memory_segments=None if memory_segments is None else memory_segments.transpose(0, 1),
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
        if memory is None:
            # A prefix straight out of the storage carries the SOURCE episodes, not a ready ``Z``: the writer runs
            # here, in this minibatch's graph. ``prefix.memory`` (an explicit ``Z``) takes precedence.
            memory = prefix.memory if prefix.memory is not None else self.memory_from_prefix(prefix)
        hidden = self._forward_tokens(
            sequence,
            positions,
            normalize_obs=False,
            memory=memory,
            memory_segments=prefix.memory_segments if memory_segments is None else memory_segments,
        )
        return hidden[num_prefix:]

    def forward_sequence(
        self,
        obs: TensorDict | torch.Tensor,
        mask: torch.Tensor | None = None,
        normalize_obs: bool = True,
        positions: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One whole episode (starting at step 0) in a single batched forward -- what a BC script wants.

        Args:
            obs: ``[S, B, obs_dim]`` (or a ``TensorDict`` with that leading shape), left-aligned at the episode
                start.
            mask: Validity of every row, ``[S, B]``. Invalid rows cannot influence any valid one.
            normalize_obs: Whether to apply the observation normalizer.
            positions: Optional explicit episode steps ``[S, B]``. Defaults to ``0, 1, ... S - 1``.
            memory: Optional ``Z`` ``[B, M, d]`` for the episode (``memory_tokens > 0`` only). ``None`` means
                ``z_init``, i.e. the first episode of a trial.

        Returns:
            ``h`` ``[S, B, d]``.
        """
        assert self.num_memory_tokens > 0 or memory is None, (
            "forward_sequence(memory=...) needs a policy built with memory_tokens > 0."
        )
        sequence = self.get_actor_obs(obs)
        num_steps, batch_size = sequence.shape[0], sequence.shape[1]
        if positions is None:
            positions = torch.arange(num_steps, device=sequence.device).unsqueeze(1).expand(num_steps, batch_size)
        key_valid = None if mask is None else mask.bool()
        return self._forward_tokens(sequence, positions, normalize_obs, key_valid, memory=memory)

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
        if self.num_memory_tokens > 0:
            # The ``M`` memory rows sit in front of the episode's own keys and are ALWAYS attendable -- they are
            # the prefix of the very same sequence the batched path builds, prefilled once per write.
            memory_ones = torch.ones(num_envs, self.num_memory_tokens, device=obs.device, dtype=torch.bool)
            attn_mask = torch.cat([memory_ones, fresh, ones], dim=-1).unsqueeze(1)  # [N, 1, M + span + 1]
        else:
            attn_mask = torch.cat([fresh, ones], dim=-1).unsqueeze(1)  # [N, 1, span + 1]

        slots = torch.remainder(positions, self.context_span)
        scatter_index = slots.view(num_envs, 1, 1).expand(num_envs, 1, self.d_model)
        hidden = tokens
        for layer, block in enumerate(self.blocks):
            normed, keys, values = block.token_kv(hidden)
            if self.num_memory_tokens > 0:
                all_keys = torch.cat([self._memory_key_cache[layer], self._key_cache[layer], keys], dim=1)
                all_values = torch.cat([self._memory_value_cache[layer], self._value_cache[layer], values], dim=1)
            else:
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
            if self.num_memory_tokens > 0:
                # The writer's view of this episode. Detached: the source episode's trunk is not shaped by the
                # write objective (it trains from its own PPO rows), which is what makes the cached-H design work.
                self._append_hidden(hidden.detach(), positions)
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

    def _append_hidden(self, hidden: torch.Tensor, positions: torch.Tensor) -> None:
        """Record ``h_t`` ``[N, d]`` as row ``M + step_t`` of the writer's view of the pass in progress.

        Rows ``0 .. M - 1`` belong to the memory tokens and are written by :meth:`_prefill_memory`, so the
        episode's own readouts start at ``M`` -- which is exactly the order of the batched pass.
        """
        slots = (positions + self.num_memory_tokens).clamp(max=self.hidden_history_span - 1).view(-1, 1)
        self._hidden_history.scatter_(1, slots.unsqueeze(-1).expand(-1, 1, self.d_model), hidden.unsqueeze(1))
        self._history_valid.scatter_(1, slots, torch.ones_like(slots, dtype=torch.bool))

    def _write_memory(self, env_ids: torch.Tensor) -> None:
        """Close the passes of ``env_ids``: ``Z <- G(H)``, detached. ``H`` is left for the caller to clear."""
        if env_ids.numel() == 0:
            return
        with torch.no_grad():
            new_memory, _ = self.write_memory(self._hidden_history[env_ids], mask=self._history_valid[env_ids])
        self._memory = self._memory.clone()
        self._memory[env_ids] = new_memory.detach().to(self._memory.dtype)
        self._episode_index[env_ids] += 1

    def _reset_trial(self, env_ids: torch.Tensor) -> None:
        """Restore ``Z`` to the learned ``z_init`` for ``env_ids`` -- the only thing a trial boundary does."""
        if env_ids.numel() == 0:
            return
        self._memory = self._memory.clone()
        self._memory[env_ids] = self.z_init.detach().to(device=self._memory.device, dtype=self._memory.dtype)
        self._episode_index[env_ids] = 0

    def reset(self, dones: torch.Tensor | None = None, trial_dones: torch.Tensor | None = None, **kwargs: Any) -> None:
        """Clear the KV cache of the environments that just ended an episode (the context never crosses a done).

        With a memory (``memory_tokens > 0``) a done additionally closes that environment's ``H`` and writes it
        into ``Z``, and a ``trial_dones`` entry restores ``Z`` to ``z_init``. Either way the affected
        environments' ``H`` is cleared and the new ``Z`` is PREFILLED: its per-layer K/V (which the next episode
        attends over from its very first step) and its rows ``0 .. M - 1`` of the fresh ``H``.
        ``trial_dones=None`` means "every episode is its own trial", i.e. the memory never survives a done -- the
        safe fallback for an environment that does not publish a trial signal. ``dones=None`` wipes everything,
        memory included.

        A trial boundary is expected to coincide with an episode boundary (a trial ends because its last episode
        did). A ``trial_dones`` entry WITHOUT the matching ``dones`` entry is still handled -- the environment
        gets ``z_init`` and a fresh prefill -- but it changes the memory in the middle of an episode, so that
        episode's ``H`` no longer describes one consistent pass and the two forward paths disagree on its
        earlier rows.
        """
        self._window_hidden, self._window_hidden_obs = None, None
        if self._key_cache is None:
            return
        if self.num_memory_tokens > 0:
            # Unlike the rest of this method, resolving ``dones`` to indices syncs the host -- but it makes the
            # writer run on the handful of environments that are actually done instead of on all of them.
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
                self._memory = self.z_init.detach().to(self._memory).unsqueeze(0).expand_as(self._memory).clone()
                self._episode_index.zero_()
                refresh = torch.arange(self._num_envs, device=self._positions.device)
            else:
                refresh = torch.cat([done_ids, trial_ids]).unique()
            # A new pass starts for these environments: drop the old readouts, then lay the (new) memory rows
            # down again as rows 0 .. M - 1 of the next ``H``.
            self._hidden_history[refresh] = 0.0
            self._history_valid[refresh] = False
            self._prefill_memory(refresh)
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
