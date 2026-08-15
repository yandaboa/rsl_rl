# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hierarchical trial-memory transformer actor-critic.

A *trial* is ``K`` consecutive episodes that share a hidden latent ``z``. Within an episode the policy attends
densely (causally) over that episode's tokens; across episodes information flows **only** through a small set of
``M`` persistent memory tokens ``Z``, which are written once per episode boundary by a cross-attention writer
``G`` and are otherwise frozen for the whole episode.

How ``Z`` enters the trunk is selected by ``memory_interface``:

* ``"residual_read"`` (default): the blocks are a plain causal transformer over the environment tokens
  ``x_1 .. x_t`` and the memory is read exactly **once**, by a gated residual cross-attention layer
  (:class:`MemoryReadLayer`) sitting after block ``memory_read_layer``. Its two gates start at zero, so at
  initialization the trunk is *bit-for-bit* the memory-free one whatever ``Z`` contains -- a BC'd trunk keeps
  its function on the first PPO update, and later drift is confined to one layer instead of perturbing all
  ``L`` of them at once,
* ``"kv_prepend"`` (legacy): ``Z`` is prepended to the keys/values of every self-attention layer, i.e. the
  trunk input is

  .. code::

      [ Z_1 .. Z_M | x_1 .. x_t ]        x_t = Embed(o_t, a_{t-1}, r_{t-1}, d_{t-1})

The rest of this docstring describes the legacy layout; under ``residual_read`` the memory-token statements
apply to the read layer's cross-attention instead (and its keys/values are cached per write, not per layer).

* memory tokens attend to memory tokens only (bidirectionally, never causally masked among themselves), so their
  per-layer states depend on ``Z`` alone -- which is what makes them cacheable for the whole episode,
* environment tokens attend to all memory tokens plus environment tokens ``<= t`` (optionally restricted to the
  last ``W`` of them; ``W >= T`` is the default and means "full episode"),
* the policy head and the value head both consume the same ``h_t``. The critic is memory-conditioned, **not**
  privileged -- see ``META_MEMORY_DESIGN.md`` section 4.

The critic has two possible trunks, selected by ``critic_trunk``:

* ``"shared"`` (default, the original architecture): the value head sits on the actor's ``h_t``,
* ``"separate"``: a full duplicate token pathway (``critic_token_embed`` / ``critic_blocks`` / ...), consuming the
  same tokens and the same memory ``Z`` (entering **detached**), so that the value loss can fit a real value
  function without ever touching the actor's representation. Every module of it is named ``critic_*`` so that
  ``train.py``'s critic-warmup freeze filter (``n.startswith("critic")``) trains all of it. This exists because
  the shared trunk made the BC -> RL recipe collapse at the warmup unfreeze: a value head on frozen features
  plateaus at a ~20-magnitude loss, and the coherent trial-level advantages that plateau implies destroy the
  policy on the first actor update.

There are two forward paths and they are required to agree numerically:

* incremental (acting) -- :meth:`forward_step` / :meth:`act` / :meth:`act_inference`, one token per call, backed
  by a per-environment KV cache,
* batched (training) -- :meth:`forward_sequence`, a padded episode ``[S, B, ...]`` in a single forward, no Python
  loop over timesteps. This is what the PPO reconstruction pass uses.

Reset semantics need two distinct signals, a single ``dones`` flag cannot express them:

* :meth:`reset_episode` clears the short-term tokens and the KV cache but **retains** ``Z``,
* :meth:`reset_trial` resets ``Z`` back to the learned ``Z_init`` ("NO_MEMORY") tokens.
"""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import warnings
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.modules.actor_critic import GSDENoiseDistribution, upcast_from_half
from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.utils import resolve_nn_activation

# Which parameter holds the action noise, per ``noise_std_type``, and its shape (``A`` = num_actions,
# ``d`` = d_model). A checkpoint written under one noise type cannot be loaded into another.
_NOISE_PARAM_NAME = {"scalar": "std", "log": "log_std", "gsde": "log_std"}


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
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Attend ``query_input`` ``[B, Lq, d]`` (pre-projection) over projected ``keys``/``values`` ``[B, Lk, d]``.

        Args:
            query_input: Query stream *before* the query projection.
            keys: Already projected keys.
            values: Already projected values.
            attn_mask: Boolean mask, ``True`` where the (query, key) pair may attend. ``[Lq, Lk]`` or ``[B, Lq, Lk]``.
            need_weights: Whether to also return the attention weights ``[B, num_heads, Lq, Lk]``.

        Returns:
            The attention output ``[B, Lq, d]`` and, if requested, the attention weights.
        """
        query = self._split_heads(self.q_proj(query_input))
        key = self._split_heads(keys)
        value = self._split_heads(values)
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
    """One pre-LN transformer block, split so that memory tokens and environment tokens can be run separately."""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, activation: str) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), resolve_nn_activation(activation), nn.Linear(ff_dim, d_model)
        )

    def memory_forward(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the ``M`` memory tokens through this block (they attend to each other only).

        Returns:
            The updated memory states and the keys/values the environment tokens must attend over.
        """
        normed = self.norm_attn(memory)
        keys, values = self.attn.project_kv(normed)
        attended, _ = self.attn(normed, keys, values, attn_mask=None)
        memory = memory + attended
        memory = memory + self.ff(self.norm_ff(memory))
        return memory, keys, values

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
    ) -> torch.Tensor:
        """Finish the block for the environment tokens, given the assembled ``[memory | tokens]`` keys/values."""
        attended, _ = self.attn(normed, keys, values, attn_mask=attn_mask)
        hidden = hidden + attended
        return hidden + self.ff(self.norm_ff(hidden))


class MemoryReadLayer(nn.Module):
    """A single gated residual cross-attention read of the memory ``Z`` (Flamingo-style), exact identity at init.

    The alternative to prepending ``Z`` to the K/V of *every* self-attention layer: the trunk stays a plain causal
    transformer over environment tokens and the memory enters exactly once, through this layer::

        q       = LN_q(h)                              # env tokens [B, S, d]
        k, v    = proj(LN_kv(Z + memory_pos_embed))    # memory     [B, M, d]
        h       = h + tanh(gate_attn) * MHA(q, k, v)
        h       = h + tanh(gate_ff)   * FF(LN_ff(h))

    Both gates are scalars initialized to zero, so ``tanh(gate) == 0`` and the layer is a bit-exact identity at
    init *whatever* ``Z`` contains. That is what lets a BC'd trunk keep its function on the first PPO update: the
    memory can only start influencing the policy once the optimizer has moved a gate off zero, and the drift is
    confined to this one layer instead of perturbing all ``L`` of them at once.
    """

    def __init__(self, d_model: int, num_heads: int, ff_dim: int, activation: str) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), resolve_nn_activation(activation), nn.Linear(ff_dim, d_model)
        )
        self.gate_attn = nn.Parameter(torch.zeros(()))
        self.gate_ff = nn.Parameter(torch.zeros(()))

    def memory_kv(self, memory: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the (position-embedded) memory ``[B, M, d]`` into this layer's keys and values.

        Split out of :meth:`forward` so that the acting path can cache it: ``Z`` only changes when the writer
        runs, i.e. once per episode, while ``forward`` runs once per environment step.
        """
        return self.attn.project_kv(self.norm_kv(memory))

    def forward(self, hidden: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Read ``keys``/``values`` (already projected by :meth:`memory_kv`) into the token stream ``[B, S, d]``."""
        attended, _ = self.attn(self.norm_q(hidden), keys, values, attn_mask=None)
        hidden = hidden + torch.tanh(self.gate_attn) * attended
        return hidden + torch.tanh(self.gate_ff) * self.ff(self.norm_ff(hidden))


class MemoryWriter(nn.Module):
    """The writer ``G``: ``Z_{e+1} = G(Z_e, H_e)`` as a residual cross-attention block plus a residual MLP.

    ``Q = Z_e`` (``M`` rows), ``K = V = H_e`` (``T + 1`` rows, the extra row is the terminal token), so the
    attention is ``M x (T + 1)`` -- never ``(T + 1) x (T + 1)``.
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

    def forward(
        self,
        memory: torch.Tensor,
        hidden: torch.Tensor,
        mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Write one episode's hidden states into the persistent memory.

        Args:
            memory: ``Z_e``, ``[B, M, d]``.
            hidden: ``H_e``, ``[B, S, d]`` with ``S = T + 1`` for a full episode.
            mask: Validity of every row of ``H_e``, ``[B, S]``. Rows that are ``False`` cannot influence ``Z``.
            need_weights: Whether to also return the ``[B, num_heads, M, S]`` attention weights.

        Returns:
            ``Z_{e+1}`` ``[B, M, d]`` and, if requested, the attention weights.
        """
        num_memory = memory.shape[1]
        query = self.norm_query(memory)
        keys, values = self.attn.project_kv(self.norm_key(hidden))

        empty = None
        if mask is not None:
            key_mask = mask.bool()
            empty = ~key_mask.any(dim=-1)
            if bool(empty.any()):
                # Avoid a fully masked softmax row (NaN); the output of those rows is zeroed out below anyway.
                key_mask = key_mask.clone()
                key_mask[empty, 0] = True
            attn_mask = key_mask.unsqueeze(1).expand(-1, num_memory, -1)
        else:
            attn_mask = None

        attended, weights = self.attn(query, keys, values, attn_mask=attn_mask, need_weights=need_weights)
        if empty is not None and bool(empty.any()):
            attended = attended * (~empty).to(attended.dtype).view(-1, 1, 1)

        memory = memory + attended
        memory = memory + self.ff(self.norm_ff(memory))
        return memory, weights


class ActorCriticTrialMemory(nn.Module):
    """Trial-memory transformer actor-critic (see the module docstring for the architecture)."""

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        d_model: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        num_memory_tokens: int = 8,
        attention_window: int | None = None,
        max_episode_length: int = 80,
        ff_mult: int = 4,
        embed_hidden_dims: tuple[int] | list[int] = (),
        actor_hidden_dims: tuple[int] | list[int] = [256],
        critic_hidden_dims: tuple[int] | list[int] = [256],
        activation: str = "gelu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        detach_critic_trunk: bool = False,
        critic_trunk: str = "shared",
        memory_interface: str = "residual_read",
        memory_read_layer: int = -1,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize the trial-memory actor-critic.

        Args:
            obs: A sample observation ``TensorDict``, used only to read the observation dimensions.
            obs_groups: Mapping from ``"policy"``/``"critic"`` to the observation groups they consume.
            num_actions: Action dimension.
            actor_obs_normalization: Whether to empirically normalize the observations feeding the tokens.
            critic_obs_normalization: Accepted for API parity. The critic consumes the same ``h_t`` as the actor,
                so there is no separate critic observation stream to normalize.
            d_model: Trunk width ``d``.
            num_layers: Number of trunk layers ``L``.
            num_heads: Number of attention heads.
            num_memory_tokens: Number of persistent memory tokens ``M``.
            attention_window: ``W``. ``None`` (default) or ``>= max_episode_length + 1`` means full-episode causal
                attention; windowing only engages when ``W`` is smaller.
            max_episode_length: ``T``, the number of acting steps in one episode. One extra token slot is reserved
                for the terminal token, so the trunk supports ``T + 1`` environment tokens.
            ff_mult: Feed-forward expansion factor.
            embed_hidden_dims: Hidden dims of the token embedding. Empty means a single linear layer.
            actor_hidden_dims: Hidden dims of the policy head (consumes ``h_t``).
            critic_hidden_dims: Hidden dims of the value head (consumes ``h_t``).
            activation: Activation used in the trunk, the writer and the heads.
            init_noise_std: Initial action noise standard deviation. Under ``"gsde"`` this is *not* the realized
                action std -- see :meth:`_update_distribution` and ``calibrate_gsde_init.py``: the realized std is
                ``init_noise_std * ||h_t||``, and ``||h_t|| = sqrt(d_model)`` for a fresh ``final_norm``.
            detach_critic_trunk: Detach the trunk readout before the value head, so the value loss trains the
                value head only. Only meaningful with ``critic_trunk="shared"``; ``"separate"`` supersedes it
                (there is nothing shared left to protect).
            memory_interface: How the memory ``Z`` enters the trunk. ``"residual_read"`` (default) runs a plain
                causal trunk over the environment tokens and reads ``Z`` exactly once, through a gated residual
                cross-attention :class:`MemoryReadLayer` that is an identity at init. ``"kv_prepend"`` is the
                legacy interface, where ``Z`` is prepended to the keys/values of *every* self-attention layer.
            memory_read_layer: Which block the read layer runs after (``residual_read`` only). ``-1`` (default)
                resolves to ``num_layers // 2``.
            critic_trunk: ``"shared"`` (the value head sits on the actor's ``h_t``) or ``"separate"`` (a full
                duplicate ``critic_*`` token pathway; see the module docstring). ``"separate"`` has **no**
                incremental/acting path -- :meth:`evaluate` and :meth:`value_from_hidden` raise, and values are
                produced in batch by :meth:`forward_sequence_critic`.
            noise_std_type: ``"scalar"``, ``"log"`` or ``"gsde"``. The first two are a state-independent diagonal
                Gaussian; ``"gsde"`` (generalized State-Dependent Exploration, https://arxiv.org/abs/2005.05719)
                makes the std a function of the trunk readout ``h_t`` -- this is what the repo's validated BC -> RL
                recipe uses (design doc section 11).
        """
        if kwargs:
            print(
                "ActorCriticTrialMemory.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        # Observation dimensions. The trunk is shared by the actor and the critic, so it is built on the policy
        # groups; see META_MEMORY_DESIGN.md section 4 for why the critic must not be privileged.
        self.obs_groups = obs_groups
        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCriticTrialMemory module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]
        if obs_groups.get("critic", obs_groups["policy"]) != obs_groups["policy"]:
            warnings.warn(
                "ActorCriticTrialMemory conditions the value head on the same memory-conditioned hidden state as the"
                " policy, so the 'critic' observation groups are ignored. Remove the asymmetry from the env cfg to"
                " make this explicit.",
                stacklevel=2,
            )
        if critic_obs_normalization and not actor_obs_normalization:
            warnings.warn(
                "critic_obs_normalization is ignored by ActorCriticTrialMemory (there is a single token stream);"
                " set actor_obs_normalization instead.",
                stacklevel=2,
            )

        self.num_actor_obs = num_actor_obs
        self.num_actions = num_actions
        # Detach the trunk readout before the value head: the value loss then trains ONLY the
        # critic head and can never rewrite the shared trunk. Added after run 225872, where the
        # unfreeze handed a ~20-magnitude residual value loss (vs ~0.05 surrogate) the trunk and
        # bulldozed the 85% BC policy to 14% in two updates.
        self.detach_critic_trunk = bool(detach_critic_trunk)
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_memory_tokens = num_memory_tokens
        self.max_episode_length = max_episode_length
        # +1 for the terminal token (design doc section 8): it carries the final reward/observation into the writer
        # but emits no action and receives no PPO loss.
        self.max_tokens = max_episode_length + 1
        self.attention_window = self.max_tokens if attention_window is None else int(attention_window)

        # Token embedding: Embed(o_t, a_{t-1}, r_{t-1}, d_{t-1})
        token_input_dim = num_actor_obs + num_actions + 2
        if len(embed_hidden_dims) == 0:
            self.token_embed = nn.Linear(token_input_dim, d_model)
        else:
            self.token_embed = MLP(token_input_dim, d_model, list(embed_hidden_dims), activation)
        self.pos_embed = nn.Parameter(torch.zeros(self.max_tokens, d_model))
        self.start_embed = nn.Parameter(torch.zeros(d_model))
        self.memory_pos_embed = nn.Parameter(torch.zeros(num_memory_tokens, d_model))

        # Persistent memory: M "NO_MEMORY" tokens
        self.z_init = nn.Parameter(torch.zeros(num_memory_tokens, d_model))

        # Trunk
        ff_dim = ff_mult * d_model
        self.blocks = nn.ModuleList([TrunkBlock(d_model, num_heads, ff_dim, activation) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)

        # Memory interface. "residual_read" keeps the blocks above a plain causal transformer and reads Z through
        # a single gated cross-attention layer; "kv_prepend" is the legacy per-layer memory K/V.
        self.memory_interface = str(memory_interface)
        if self.memory_interface not in ("residual_read", "kv_prepend"):
            raise ValueError(
                f"Unknown memory_interface '{memory_interface}'. Expected 'residual_read' or 'kv_prepend'."
            )
        self.residual_memory_read = self.memory_interface == "residual_read"
        resolved_read_layer = num_layers // 2 if int(memory_read_layer) < 0 else int(memory_read_layer)
        if self.residual_memory_read and not 0 <= resolved_read_layer < num_layers:
            raise ValueError(
                f"memory_read_layer={memory_read_layer} resolves to block {resolved_read_layer}, which is outside"
                f" [0, {num_layers - 1}]."
            )
        self.memory_read_layer = resolved_read_layer
        if self.residual_memory_read:
            self.memory_read = MemoryReadLayer(d_model, num_heads, ff_dim, activation)

        # Memory writer G
        self.writer = MemoryWriter(d_model, num_heads, ff_dim, activation)

        # Heads (both consume h_t)
        self.actor = MLP(d_model, num_actions, list(actor_hidden_dims), activation)
        self.critic = MLP(d_model, 1, list(critic_hidden_dims), activation)

        # Optional fully separate critic pathway. Same shapes as the actor pathway, every module prefixed
        # ``critic_`` so that the warm-start freeze filter (``n.startswith("critic")``) trains all of it, and no
        # actor parameter is reachable from the value loss (the memory Z enters detached, see
        # :meth:`forward_sequence_critic`). Batched-only: there is no KV cache and no acting state for it.
        self.critic_trunk = str(critic_trunk)
        if self.critic_trunk not in ("shared", "separate"):
            raise ValueError(f"Unknown critic_trunk '{critic_trunk}'. Expected 'shared' or 'separate'.")
        if self.critic_trunk == "separate":
            if len(embed_hidden_dims) == 0:
                self.critic_token_embed = nn.Linear(token_input_dim, d_model)
            else:
                self.critic_token_embed = MLP(token_input_dim, d_model, list(embed_hidden_dims), activation)
            self.critic_pos_embed = nn.Parameter(torch.zeros(self.max_tokens, d_model))
            self.critic_start_embed = nn.Parameter(torch.zeros(d_model))
            self.critic_memory_pos_embed = nn.Parameter(torch.zeros(num_memory_tokens, d_model))
            self.critic_blocks = nn.ModuleList([
                TrunkBlock(d_model, num_heads, ff_dim, activation) for _ in range(num_layers)
            ])
            self.critic_final_norm = nn.LayerNorm(d_model)
            if self.residual_memory_read:
                # The critic's own read layer, with its own gates: mirrors the actor's placement but is trained
                # (and drifts) independently. Memory still enters detached, see forward_sequence_critic.
                self.critic_memory_read = MemoryReadLayer(d_model, num_heads, ff_dim, activation)

        # Observation normalization (single stream, shared by both heads)
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()
        self.critic_obs_normalization = False

        # Action noise
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        elif noise_std_type == "gsde":
            # gSDE keys the noise on the trunk readout h_t, so log_std is [d_model, num_actions] -- one column of
            # per-feature scales per action, exactly as in the state-history transformer recipe.
            self.log_std = nn.Parameter(torch.ones(d_model, num_actions) * math.log(init_noise_std))
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar', 'log' or 'gsde'")

        # Action distribution (populated in _update_distribution). gSDE keeps a single distribution object that is
        # re-parameterized in place; the other noise types build a fresh Normal per call.
        self.distribution = self._make_gsde_distribution() if noise_std_type == "gsde" else None
        Normal.set_default_validate_args(False)

        self._init_weights()
        self._reset_runtime_state()

        print(
            f"Trial-memory trunk: L={num_layers} d={d_model} heads={num_heads} M={num_memory_tokens}"
            f" critic_trunk={self.critic_trunk} memory_interface={self.memory_interface}"
            + (f" (read after block {self.memory_read_layer})" if self.residual_memory_read else "")
        )
        print(f"Actor head: {self.actor}")
        print(f"Critic head: {self.critic}")

    # ------------------------------------------------------------------------------------------------------------
    # Initialization / runtime state
    # ------------------------------------------------------------------------------------------------------------

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
        self.writer.attn.out_proj.weight.data.mul_(residual_scale)
        self.writer.ff[-1].weight.data.mul_(residual_scale)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.memory_pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.start_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.z_init, mean=0.0, std=0.02)
        if getattr(self, "critic_trunk", "shared") == "separate":
            for block in self.critic_blocks:
                block.attn.out_proj.weight.data.mul_(residual_scale)
                block.ff[-1].weight.data.mul_(residual_scale)
            nn.init.normal_(self.critic_pos_embed, mean=0.0, std=0.02)
            nn.init.normal_(self.critic_memory_pos_embed, mean=0.0, std=0.02)
            nn.init.normal_(self.critic_start_embed, mean=0.0, std=0.02)

    # Every attribute :meth:`_reset_runtime_state` owns. Kept as a list so that a frozen copy of the policy
    # (:meth:`frozen_copy`) can be taken without duplicating the multi-GB acting-time buffers.
    _RUNTIME_STATE_ATTRS: tuple[str, ...] = (
        "_num_envs",
        "_memory",
        "_memory_kv",
        "_read_kv",
        "_key_cache",
        "_value_cache",
        "_hidden_history",
        "_token_valid",
        "_positions",
        "_prev_actions",
        "_prev_rewards",
        "_prev_dones",
        "_is_start",
        "_last_hidden",
    )

    def _reset_runtime_state(self) -> None:
        """Drop every acting-time buffer (they are lazily re-allocated on the first :meth:`forward_step`)."""
        self._num_envs: int | None = None
        self._memory: torch.Tensor | None = None
        self._memory_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        # Cached keys/values of Z for the residual read layer, recomputed only when Z changes (a write or a
        # trial reset), never per step.
        self._read_kv: tuple[torch.Tensor, torch.Tensor] | None = None
        self._key_cache: list[torch.Tensor] | None = None
        self._value_cache: list[torch.Tensor] | None = None
        self._hidden_history: torch.Tensor | None = None
        self._token_valid: torch.Tensor | None = None
        self._positions: torch.Tensor | None = None
        self._prev_actions: torch.Tensor | None = None
        self._prev_rewards: torch.Tensor | None = None
        self._prev_dones: torch.Tensor | None = None
        self._is_start: torch.Tensor | None = None
        self._last_hidden: torch.Tensor | None = None

    def frozen_copy(self) -> ActorCriticTrialMemory:
        """A detached deep copy of this policy in eval mode, **without** its acting-time buffers.

        Used as the reference ("policy at load time") for the drift diagnostics. The acting state is swapped out
        before the copy is taken and restored afterwards, because at 16k environments the KV cache alone is tens
        of gigabytes -- copying it would be the single largest allocation in the run.
        """
        # The live action distribution holds non-leaf tensors (the last h_t and its base Normal), which
        # deepcopy refuses; it is pure scratch space, so it is swapped out with the rest of the runtime state
        # and re-armed on the copy.
        saved = {name: getattr(self, name, None) for name in self._RUNTIME_STATE_ATTRS}
        saved["distribution"] = self.distribution
        for name in saved:
            setattr(self, name, None)
        try:
            reference = copy.deepcopy(self)
        finally:
            for name, value in saved.items():
                setattr(self, name, value)
        reference._reset_runtime_state()
        # A fresh gSDE distribution *without* re-drawing the exploration matrix: the matrix feeds only
        # ``get_noise``, which this policy never calls, and re-drawing it would consume global RNG draws and
        # therefore change the minibatch shuffle of the run this copy is only supposed to observe.
        reference.distribution = (
            GSDENoiseDistribution(action_dim=self.num_actions) if self.noise_std_type == "gsde" else None
        )
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        return reference

    def initialize_state(self, num_envs: int, device: torch.device | str, dtype: torch.dtype | None = None) -> None:
        """Allocate the acting-time state for ``num_envs`` environments and set ``Z = Z_init``."""
        dtype = self.z_init.dtype if dtype is None else dtype
        self._num_envs = num_envs
        self._memory = self.z_init.detach().to(device=device, dtype=dtype).unsqueeze(0).repeat(num_envs, 1, 1)
        self._memory_kv = None
        self._read_kv = None
        self._key_cache = [
            torch.zeros(num_envs, self.max_tokens, self.d_model, device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]
        self._value_cache = [
            torch.zeros(num_envs, self.max_tokens, self.d_model, device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]
        self._hidden_history = torch.zeros(num_envs, self.max_tokens, self.d_model, device=device, dtype=dtype)
        self._token_valid = torch.zeros(num_envs, self.max_tokens, device=device, dtype=torch.bool)
        self._positions = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._prev_actions = torch.zeros(num_envs, self.num_actions, device=device, dtype=dtype)
        self._prev_rewards = torch.zeros(num_envs, 1, device=device, dtype=dtype)
        self._prev_dones = torch.zeros(num_envs, 1, device=device, dtype=dtype)
        self._is_start = torch.ones(num_envs, device=device, dtype=torch.bool)
        self._last_hidden = None

    def _ensure_state(self, num_envs: int, device: torch.device, dtype: torch.dtype) -> None:
        if self._num_envs != num_envs or self._memory is None or self._memory.device != device:
            self.initialize_state(num_envs, device, dtype)

    # ------------------------------------------------------------------------------------------------------------
    # Properties / distribution
    # ------------------------------------------------------------------------------------------------------------

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    @property
    def memory(self) -> torch.Tensor | None:
        """The persistent memory ``Z`` of the acting path, ``[num_envs, M, d]`` (``None`` before the first step)."""
        return self._memory

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def _make_gsde_distribution(self) -> GSDENoiseDistribution:
        """Build the gSDE distribution and draw its exploration matrix.

        **When is ``sample_weights`` called, and does it matter for the PPO ratio?** Once, here (construction and
        after :meth:`load_state_dict`), which is the convention of the reference implementation. It is safe to call
        at any time: the exploration matrix feeds only :meth:`GSDENoiseDistribution.get_noise`, which this policy
        never calls. Both the density (``log_prob``/``entropy``) and the sample come from the base Normal that
        :meth:`GSDENoiseDistribution.proba_distribution` builds from ``sqrt(h_t^2 @ sigma^2)``, which depends on
        ``h_t`` and ``log_std`` only. So resampling the weights cannot desynchronize the acting path from the PPO
        reconstruction pass, and cannot move a stored behavior log-prob.
        """
        distribution = GSDENoiseDistribution(action_dim=self.num_actions)
        distribution.sample_weights(self.log_std)
        return distribution

    def _action_std(self, mean: torch.Tensor) -> torch.Tensor:
        if self.noise_std_type == "scalar":
            return self.std.expand_as(mean)
        return torch.exp(self.log_std).expand_as(mean)

    def _update_distribution(self, hidden: torch.Tensor) -> None:
        """Set the action distribution from ``h_t``.

        Shape-agnostic: ``hidden`` may be ``[N, d]`` (acting path) or ``[S, B, d]`` (batched reconstruction path).
        Both routes run the identical arithmetic, which is what keeps the PPO reconstruction canary exact.
        """
        mean = self.actor(hidden)
        if self.noise_std_type == "gsde":
            # std = sqrt(h_t^2 @ exp(log_std)^2), i.e. state-dependent noise keyed on the shared trunk readout.
            # (``proba_distribution`` forces that computation to fp32; see the note there about autocast.)
            self.distribution.proba_distribution(mean, self.log_std, hidden)
        else:
            # The upcast is a no-op outside autocast; under it, it keeps the density in fp32 and matches the
            # dtype of the std parameter (``Normal`` does not promote a half mean against an fp32 std).
            self.distribution = Normal(upcast_from_half(mean), upcast_from_half(self._action_std(mean)))

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    # ------------------------------------------------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------------------------------------------------

    def get_actor_obs(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            return obs
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict | torch.Tensor) -> torch.Tensor:
        """The critic consumes ``h_t``; this exists for API parity and returns the policy observation."""
        return self.get_actor_obs(obs)

    def update_normalization(self, obs: TensorDict | torch.Tensor) -> None:
        if self.actor_obs_normalization:
            actor_obs = self.get_actor_obs(obs)
            self.actor_obs_normalizer.update(actor_obs.reshape(-1, self.num_actor_obs))

    # ------------------------------------------------------------------------------------------------------------
    # Token embedding + trunk
    # ------------------------------------------------------------------------------------------------------------

    def _pathway(self, critic: bool = False) -> tuple[Any, ...]:
        """The token pathway to run: the actor's, or (``critic=True``) the separate ``critic_*`` duplicate.

        Returns ``(token_embed, pos_embed, start_embed, memory_pos_embed, blocks, final_norm)``.
        """
        if not critic:
            return (self.token_embed, self.pos_embed, self.start_embed, self.memory_pos_embed, self.blocks,
                    self.final_norm)
        if self.critic_trunk != "separate":
            raise RuntimeError("The critic token pathway only exists with critic_trunk='separate'.")
        return (self.critic_token_embed, self.critic_pos_embed, self.critic_start_embed,
                self.critic_memory_pos_embed, self.critic_blocks, self.critic_final_norm)

    def _embed_tokens(
        self,
        obs: torch.Tensor,
        prev_actions: torch.Tensor,
        prev_rewards: torch.Tensor,
        prev_dones: torch.Tensor,
        positions: torch.Tensor,
        is_start: torch.Tensor,
        normalize_obs: bool = True,
        critic: bool = False,
    ) -> torch.Tensor:
        """Build ``x = Embed(o, a_prev, r_prev, d_prev) + pos + start_marker``. All inputs share leading dims.

        Note: both pathways normalize with the SAME ``actor_obs_normalizer`` -- there is one observation stream,
        and the normalizer holds buffers only, so nothing differentiable is shared by this.
        """
        token_embed, pos_embed, start_embed, _, _, _ = self._pathway(critic)
        if normalize_obs:
            obs = self.actor_obs_normalizer(obs)
        features = torch.cat([obs, prev_actions, prev_rewards, prev_dones], dim=-1)
        tokens = token_embed(features)
        tokens = tokens + pos_embed[positions]
        return tokens + is_start.unsqueeze(-1).to(tokens.dtype) * start_embed

    def _read_layer(self, critic: bool = False) -> MemoryReadLayer:
        """The gated residual read layer of the requested pathway (``residual_read`` only)."""
        if not self.residual_memory_read:
            raise RuntimeError("There is no memory read layer with memory_interface='kv_prepend'.")
        if critic:
            if self.critic_trunk != "separate":
                raise RuntimeError("The critic read layer only exists with critic_trunk='separate'.")
            return self.critic_memory_read
        return self.memory_read

    def _memory_read_kv(self, memory: torch.Tensor, critic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys/values of ``Z + memory_pos_embed`` for the read layer of the requested pathway."""
        _, _, _, memory_pos_embed, _, _ = self._pathway(critic)
        return self._read_layer(critic).memory_kv(memory + memory_pos_embed)

    def memory_gate_values(self) -> dict[str, float]:
        """The realized gate values ``tanh(g)`` of every read layer, for logging. Empty in legacy mode."""
        if not self.residual_memory_read:
            return {}
        gates = {
            "memory_gate_attn": float(torch.tanh(self.memory_read.gate_attn.detach()).item()),
            "memory_gate_ff": float(torch.tanh(self.memory_read.gate_ff.detach()).item()),
        }
        if self.critic_trunk == "separate":
            gates["critic_memory_gate_attn"] = float(torch.tanh(self.critic_memory_read.gate_attn.detach()).item())
            gates["critic_memory_gate_ff"] = float(torch.tanh(self.critic_memory_read.gate_ff.detach()).item())
        return gates

    def _memory_kv_states(
        self, memory: torch.Tensor, critic: bool = False
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Run the ``M`` memory tokens through the trunk and collect the per-layer keys/values they expose."""
        _, _, _, memory_pos_embed, blocks, _ = self._pathway(critic)
        states = memory + memory_pos_embed
        memory_kv = []
        for block in blocks:
            states, keys, values = block.memory_forward(states)
            memory_kv.append((keys, values))
        return memory_kv

    def initial_memory(self, batch_size: int, device: torch.device | str | None = None) -> torch.Tensor:
        """The learned "NO_MEMORY" memory ``[B, M, d]``. Differentiable, so a trial-1 loss reaches ``Z_init``."""
        memory = self.z_init.unsqueeze(0).expand(batch_size, -1, -1)
        return memory if device is None else memory.to(device)

    # ------------------------------------------------------------------------------------------------------------
    # Batched (training) path
    # ------------------------------------------------------------------------------------------------------------

    def _sequence_attn_mask(
        self, num_tokens: int, mask: torch.Tensor | None, device: torch.device
    ) -> torch.Tensor:
        """Assemble the ``[memory | tokens]`` attention mask, ``[S, M + S]`` or ``[B, S, M + S]``, ``True`` = attend.

        With ``memory_interface="residual_read"`` there is no memory stripe: the blocks only ever see environment
        tokens, so the mask is the pure causal/windowed one, ``[S, S]`` or ``[B, S, S]``.
        """
        positions = torch.arange(num_tokens, device=device)
        causal = positions.unsqueeze(1) >= positions.unsqueeze(0)
        if self.attention_window < num_tokens:
            causal = causal & ((positions.unsqueeze(1) - positions.unsqueeze(0)) < self.attention_window)
        if self.residual_memory_read:
            return causal if mask is None else (causal.unsqueeze(0) & mask.bool().unsqueeze(1))
        if mask is None:
            memory_mask = torch.ones(num_tokens, self.num_memory_tokens, device=device, dtype=torch.bool)
            return torch.cat([memory_mask, causal], dim=-1)
        key_valid = mask.bool()  # [B, S]
        token_mask = causal.unsqueeze(0) & key_valid.unsqueeze(1)
        memory_mask = torch.ones(
            key_valid.shape[0], num_tokens, self.num_memory_tokens, device=device, dtype=torch.bool
        )
        return torch.cat([memory_mask, token_mask], dim=-1)

    def forward_sequence(
        self,
        obs: TensorDict | torch.Tensor,
        prev_actions: torch.Tensor,
        prev_rewards: torch.Tensor,
        prev_dones: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor | None = None,
        episode_start: torch.Tensor | None = None,
        normalize_obs: bool = True,
    ) -> torch.Tensor:
        """Run a whole padded episode in **one** batched forward (no Python loop over timesteps).

        Args:
            obs: ``[S, B, obs_dim]`` (or a ``TensorDict`` with that leading shape). ``S`` is the padded episode
                length; pass ``S = T + 1`` to include the terminal token.
            prev_actions: ``a_{t-1}``, ``[S, B, num_actions]``.
            prev_rewards: ``r_{t-1}``, ``[S, B]`` or ``[S, B, 1]``.
            prev_dones: ``d_{t-1}``, ``[S, B]`` or ``[S, B, 1]``.
            memory: ``Z_in`` for these episodes, ``[B, M, d]``. Use :meth:`initial_memory` for the first episode.
            mask: Validity of every timestep, ``[S, B]``. Episodes are assumed left-aligned (padding on the right).
            episode_start: ``[S, B]`` marker for the learned "episode start" token. Defaults to row 0.
            normalize_obs: Whether to apply the observation normalizer (disable if the caller pre-normalized).

        Returns:
            ``h`` ``[S, B, d]``. Rows where ``mask`` is ``False`` are meaningless and must be ignored by the caller.
        """
        return self._forward_sequence(
            obs, prev_actions, prev_rewards, prev_dones, memory, mask, episode_start, normalize_obs, critic=False
        )

    def _forward_sequence(
        self,
        obs: TensorDict | torch.Tensor,
        prev_actions: torch.Tensor,
        prev_rewards: torch.Tensor,
        prev_dones: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor | None = None,
        episode_start: torch.Tensor | None = None,
        normalize_obs: bool = True,
        critic: bool = False,
    ) -> torch.Tensor:
        """The batched trunk forward, shared by the actor pathway and the separate critic pathway.

        The two differ ONLY in which set of modules :meth:`_pathway` hands back; the arithmetic (and hence the
        PPO reconstruction canary on the actor path) is identical.
        """
        _, _, _, _, blocks, final_norm = self._pathway(critic)
        obs = self.get_actor_obs(obs)
        num_steps, batch_size = obs.shape[0], obs.shape[1]
        if num_steps > self.max_tokens:
            raise ValueError(f"Sequence length {num_steps} exceeds max_tokens {self.max_tokens} (T + 1).")
        device = obs.device

        prev_rewards = prev_rewards.reshape(num_steps, batch_size, 1)
        prev_dones = prev_dones.reshape(num_steps, batch_size, 1).to(obs.dtype)
        if episode_start is None:
            episode_start = torch.zeros(num_steps, batch_size, device=device, dtype=torch.bool)
            episode_start[0] = True

        positions = torch.arange(num_steps, device=device).unsqueeze(1).expand(num_steps, batch_size)
        tokens = self._embed_tokens(
            obs, prev_actions, prev_rewards, prev_dones, positions, episode_start, normalize_obs, critic=critic
        )
        # [S, B, d] -> [B, S, d]
        hidden = tokens.transpose(0, 1)

        attn_mask = self._sequence_attn_mask(num_steps, None if mask is None else mask.transpose(0, 1), device)
        if self.residual_memory_read:
            # Plain causal trunk over the environment tokens; the memory enters once, after block
            # ``memory_read_layer``, through the gated read layer (an identity while its gates are zero).
            read = self._read_layer(critic)
            read_keys, read_values = self._memory_read_kv(memory, critic=critic)
            for layer, block in enumerate(blocks):
                normed, keys, values = block.token_kv(hidden)
                hidden = block.token_forward(hidden, normed, keys, values, attn_mask)
                if layer == self.memory_read_layer:
                    hidden = read(hidden, read_keys, read_values)
        else:
            memory_kv = self._memory_kv_states(memory, critic=critic)
            for block, (memory_keys, memory_values) in zip(blocks, memory_kv):
                normed, keys, values = block.token_kv(hidden)
                hidden = block.token_forward(
                    hidden,
                    normed,
                    torch.cat([memory_keys, keys], dim=1),
                    torch.cat([memory_values, values], dim=1),
                    attn_mask,
                )
        hidden = final_norm(hidden)
        return hidden.transpose(0, 1)

    def forward_sequence_critic(
        self,
        obs: TensorDict | torch.Tensor,
        prev_actions: torch.Tensor,
        prev_rewards: torch.Tensor,
        prev_dones: torch.Tensor,
        memory: torch.Tensor,
        mask: torch.Tensor | None = None,
        episode_start: torch.Tensor | None = None,
        normalize_obs: bool = True,
    ) -> torch.Tensor:
        """Values ``[S, B, 1]`` of a padded episode, through the **separate** critic pathway.

        Mirrors :meth:`forward_sequence` (same tokens, same attention mask, same memory ``Z``) but runs the
        ``critic_*`` modules and ends in the value head. ``memory`` enters **detached**: the writer, ``z_init``
        and the actor trunk that produced ``Z`` must never see a value gradient -- that separation is the entire
        point of the separate trunk.
        """
        if self.critic_trunk != "separate":
            raise RuntimeError(
                "forward_sequence_critic() requires critic_trunk='separate'; with the shared trunk use"
                " evaluate_sequence()/value_from_hidden()."
            )
        hidden = self._forward_sequence(
            obs,
            prev_actions,
            prev_rewards,
            prev_dones,
            memory.detach(),
            mask,
            episode_start,
            normalize_obs,
            critic=True,
        )
        return self.critic(hidden)

    def act_sequence(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """:meth:`forward_sequence` followed by the policy head. Returns ``(hidden, sampled_actions)``."""
        hidden = self.forward_sequence(*args, **kwargs)
        self._update_distribution(hidden)
        return hidden, self.distribution.sample()

    def _critic_input(self, hidden: torch.Tensor) -> torch.Tensor:
        """The value head's view of the trunk readout (detached when ``detach_critic_trunk``)."""
        return hidden.detach() if self.detach_critic_trunk else hidden

    def evaluate_sequence(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """:meth:`forward_sequence` followed by the value head. Returns values ``[S, B, 1]``.

        With ``critic_trunk="separate"`` this forwards to :meth:`forward_sequence_critic` (identical signature),
        so it never silently returns an actor-trunk value.
        """
        if self.critic_trunk == "separate":
            return self.forward_sequence_critic(*args, **kwargs)
        return self.critic(self._critic_input(self.forward_sequence(*args, **kwargs)))

    def action_mean_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Policy mean from a hidden state produced by either forward path."""
        return self.actor(hidden)

    def update_distribution_from_hidden(self, hidden: torch.Tensor) -> Normal | GSDENoiseDistribution:
        """Set (and return) the action distribution from a hidden state, e.g. during the PPO recompute pass."""
        self._update_distribution(hidden)
        return self.distribution

    def value_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Value from a hidden state produced by either forward path (``critic_trunk="shared"`` only)."""
        if self.critic_trunk == "separate":
            raise RuntimeError(
                "value_from_hidden() is meaningless with critic_trunk='separate': the value head consumes the"
                " CRITIC trunk's readout, not the actor's. Use forward_sequence_critic() on the same tokens."
            )
        return self.critic(self._critic_input(hidden))

    def write_memory(
        self,
        memory: torch.Tensor,
        hidden: torch.Tensor,
        mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the writer ``G``: ``Z_{e+1} = G(Z_e, H_e)``. See :class:`MemoryWriter` for the shapes."""
        return self.writer(memory, hidden, mask=mask, need_weights=need_weights)

    # ------------------------------------------------------------------------------------------------------------
    # Incremental (acting) path
    # ------------------------------------------------------------------------------------------------------------

    def forward_step(
        self,
        obs: TensorDict | torch.Tensor,
        prev_actions: torch.Tensor | None = None,
        prev_rewards: torch.Tensor | None = None,
        prev_dones: torch.Tensor | None = None,
        commit: bool = True,
        normalize_obs: bool = True,
    ) -> torch.Tensor:
        """Advance the acting path by one token and return ``h_t`` ``[num_envs, d]``.

        ``prev_*`` default to the module's own buffers (the previously emitted action and the last
        :meth:`record_transition`). With ``commit=False`` the token is *not* written to the KV cache -- used for the
        value bootstrap at the end of a rollout.
        """
        obs = self.get_actor_obs(obs)
        num_envs = obs.shape[0]
        self._ensure_state(num_envs, obs.device, obs.dtype)

        prev_actions = self._prev_actions if prev_actions is None else prev_actions
        prev_rewards = self._prev_rewards if prev_rewards is None else prev_rewards.reshape(num_envs, 1)
        prev_dones = self._prev_dones if prev_dones is None else prev_dones.reshape(num_envs, 1).to(obs.dtype)

        positions = self._positions
        if bool((positions >= self.max_tokens).any()):
            raise RuntimeError(
                f"Episode exceeded max_tokens={self.max_tokens}; call reset_episode() at the episode boundary."
            )

        tokens = self._embed_tokens(
            obs.unsqueeze(1),
            prev_actions.unsqueeze(1),
            prev_rewards.unsqueeze(1),
            prev_dones.unsqueeze(1),
            positions.unsqueeze(1),
            self._is_start.unsqueeze(1),
            normalize_obs,
        )  # [N, 1, d]

        if self.residual_memory_read:
            # One K/V projection of Z per *write*, not per step: ``_read_kv`` is invalidated exactly where
            # ``_memory`` changes (initialize_state / update_memory / reset_trial).
            if self._read_kv is None:
                read_keys, read_values = self._memory_read_kv(self._memory)
                self._read_kv = (read_keys.detach(), read_values.detach())
        elif self._memory_kv is None:
            self._memory_kv = [(k.detach(), v.detach()) for k, v in self._memory_kv_states(self._memory)]

        # Keys visible to this query: memory tokens (legacy interface only) + committed tokens inside the
        # window + itself (last).
        key_positions = torch.arange(self.max_tokens, device=obs.device)
        window_ok = key_positions.unsqueeze(0) > (positions.unsqueeze(1) - self.attention_window)
        cache_mask = self._token_valid & window_ok  # [N, max_tokens]
        num_always = 1 if self.residual_memory_read else self.num_memory_tokens + 1
        ones = torch.ones(num_envs, num_always, device=obs.device, dtype=torch.bool)
        if self.residual_memory_read:
            attn_mask = torch.cat([cache_mask, ones], dim=-1).unsqueeze(1)
        else:
            attn_mask = torch.cat([ones[:, : self.num_memory_tokens], cache_mask, ones[:, -1:]], dim=-1).unsqueeze(1)

        hidden = tokens
        scatter_index = positions.view(num_envs, 1, 1).expand(num_envs, 1, self.d_model)
        for layer, block in enumerate(self.blocks):
            normed, keys, values = block.token_kv(hidden)
            if self.residual_memory_read:
                all_keys = torch.cat([self._key_cache[layer], keys], dim=1)
                all_values = torch.cat([self._value_cache[layer], values], dim=1)
            else:
                memory_keys, memory_values = self._memory_kv[layer]
                all_keys = torch.cat([memory_keys, self._key_cache[layer], keys], dim=1)
                all_values = torch.cat([memory_values, self._value_cache[layer], values], dim=1)
            hidden = block.token_forward(hidden, normed, all_keys, all_values, attn_mask)
            if commit:
                self._key_cache[layer].scatter_(1, scatter_index, keys.detach())
                self._value_cache[layer].scatter_(1, scatter_index, values.detach())
            if self.residual_memory_read and layer == self.memory_read_layer:
                hidden = self.memory_read(hidden, self._read_kv[0], self._read_kv[1])
        hidden = self.final_norm(hidden)

        if commit:
            self._hidden_history.scatter_(1, scatter_index, hidden.detach())
            self._token_valid.scatter_(1, positions.view(num_envs, 1), True)
            self._positions = positions + 1
            self._is_start = torch.zeros_like(self._is_start)

        hidden = hidden.squeeze(1)
        if commit:
            self._last_hidden = hidden
        return hidden

    def act(
        self,
        obs: TensorDict | torch.Tensor,
        prev_actions: torch.Tensor | None = None,
        prev_rewards: torch.Tensor | None = None,
        prev_dones: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden = self.forward_step(obs, prev_actions, prev_rewards, prev_dones, commit=True)
        self._update_distribution(hidden)
        actions = self.distribution.sample()
        self._prev_actions = actions.detach()
        return actions

    def act_inference(
        self,
        obs: TensorDict | torch.Tensor,
        prev_actions: torch.Tensor | None = None,
        prev_rewards: torch.Tensor | None = None,
        prev_dones: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.forward_step(obs, prev_actions, prev_rewards, prev_dones, commit=True)
        actions = self.actor(hidden)
        self._prev_actions = actions.detach()
        return actions

    def evaluate(
        self,
        obs: TensorDict | torch.Tensor,
        prev_actions: torch.Tensor | None = None,
        prev_rewards: torch.Tensor | None = None,
        prev_dones: torch.Tensor | None = None,
        use_cached_hidden: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Value of the current step.

        Right after :meth:`act` this reuses the cached ``h_t`` (no extra token). Otherwise -- e.g. the bootstrap
        value at the end of a rollout, after :meth:`record_transition` -- it peeks a token without committing it.

        Only available with ``critic_trunk="shared"``: the separate critic pathway is batched-only (no KV cache),
        so PPO fills the values with a :meth:`forward_sequence_critic` sweep after collection instead.
        """
        if self.critic_trunk == "separate":
            raise RuntimeError(
                "evaluate() is not available with critic_trunk='separate' (the critic pathway has no incremental"
                " path). PPO must fill the values with a batched forward_sequence_critic sweep."
            )
        if use_cached_hidden and self._last_hidden is not None:
            return self.critic(self._critic_input(self._last_hidden))
        hidden = self.forward_step(obs, prev_actions, prev_rewards, prev_dones, commit=False)
        return self.critic(self._critic_input(hidden))

    def record_transition(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        """Stash ``r_{t}`` / ``d_{t}`` so the *next* token can carry them. Call once per environment step."""
        if self._prev_rewards is None:
            raise RuntimeError("record_transition() called before the first act(); no acting state allocated.")
        num_envs = self._num_envs
        self._prev_rewards = rewards.detach().reshape(num_envs, 1).to(self._prev_rewards.dtype)
        self._prev_dones = dones.detach().reshape(num_envs, 1).to(self._prev_dones.dtype)
        self._last_hidden = None

    def append_terminal_token(
        self,
        obs: TensorDict | torch.Tensor,
        prev_actions: torch.Tensor | None = None,
        prev_rewards: torch.Tensor | None = None,
        prev_dones: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Commit the terminal token ``x_{T+1} = Embed(o_{T+1}, a_T, r_T, d_T)`` (design doc section 8).

        It emits no action and receives no PPO loss; it exists so the writer can see the episode outcome. Call it
        *before* :meth:`update_memory` and :meth:`reset_episode`. ``env_ids`` restricts the update to the
        environments that just finished an episode (the others keep their cache untouched); all tensor arguments
        stay full-size ``[num_envs, ...]`` and are indexed internally.
        """
        obs = self.get_actor_obs(obs)
        if env_ids is None:
            hidden = self.forward_step(obs, prev_actions, prev_rewards, prev_dones, commit=True)
            self._last_hidden = None
            return hidden
        obs = obs[env_ids]
        prev_actions = None if prev_actions is None else prev_actions[env_ids]
        prev_rewards = None if prev_rewards is None else prev_rewards.reshape(-1, 1)[env_ids]
        prev_dones = None if prev_dones is None else prev_dones.reshape(-1, 1)[env_ids]
        # Sub-select the acting state so that only ``env_ids`` advance, then scatter the result back.
        saved = self._checkout_state(env_ids)
        hidden = self.forward_step(obs, prev_actions, prev_rewards, prev_dones, commit=True)
        self._commit_state(env_ids, saved)
        self._last_hidden = None
        return hidden

    def update_memory(self, env_ids: torch.Tensor | None = None, need_weights: bool = False) -> torch.Tensor | None:
        """Run the writer over the episode collected so far and store ``Z_{e+1}``. Call at an episode boundary.

        Returns the writer attention weights ``[n, num_heads, M, T + 1]`` when ``need_weights`` is set.
        """
        if self._memory is None:
            raise RuntimeError("update_memory() called before the first act(); no acting state allocated.")
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._memory.device)
        if env_ids.numel() == 0:
            return None
        new_memory, weights = self.writer(
            self._memory[env_ids],
            self._hidden_history[env_ids],
            mask=self._token_valid[env_ids],
            need_weights=need_weights,
        )
        self._memory = self._memory.clone()
        self._memory[env_ids] = new_memory.detach()
        self._memory_kv = None
        self._read_kv = None
        return weights

    def _checkout_state(self, env_ids: torch.Tensor) -> dict[str, Any]:
        """Temporarily narrow the acting state to ``env_ids`` (used by :meth:`append_terminal_token`)."""
        saved = {
            "num_envs": self._num_envs,
            "memory": self._memory,
            "memory_kv": self._memory_kv,
            "read_kv": self._read_kv,
            "key_cache": self._key_cache,
            "value_cache": self._value_cache,
            "hidden_history": self._hidden_history,
            "token_valid": self._token_valid,
            "positions": self._positions,
            "prev_actions": self._prev_actions,
            "prev_rewards": self._prev_rewards,
            "prev_dones": self._prev_dones,
            "is_start": self._is_start,
        }
        self._num_envs = int(env_ids.numel())
        self._memory = self._memory[env_ids]
        if saved["memory_kv"] is None:
            self._memory_kv = None
        else:
            self._memory_kv = [(keys[env_ids], values[env_ids]) for keys, values in saved["memory_kv"]]
        if saved["read_kv"] is None:
            self._read_kv = None
        else:
            self._read_kv = (saved["read_kv"][0][env_ids], saved["read_kv"][1][env_ids])
        self._key_cache = [cache[env_ids] for cache in saved["key_cache"]]
        self._value_cache = [cache[env_ids] for cache in saved["value_cache"]]
        self._hidden_history = saved["hidden_history"][env_ids]
        self._token_valid = saved["token_valid"][env_ids]
        self._positions = saved["positions"][env_ids]
        self._prev_actions = saved["prev_actions"][env_ids]
        self._prev_rewards = saved["prev_rewards"][env_ids]
        self._prev_dones = saved["prev_dones"][env_ids]
        self._is_start = saved["is_start"][env_ids]
        return saved

    def _commit_state(self, env_ids: torch.Tensor, saved: dict[str, Any]) -> None:
        """Scatter the narrowed acting state back into the full-size buffers."""
        for layer in range(self.num_layers):
            saved["key_cache"][layer][env_ids] = self._key_cache[layer]
            saved["value_cache"][layer][env_ids] = self._value_cache[layer]
        saved["hidden_history"][env_ids] = self._hidden_history
        saved["token_valid"][env_ids] = self._token_valid
        saved["positions"][env_ids] = self._positions
        saved["prev_actions"][env_ids] = self._prev_actions
        saved["prev_rewards"][env_ids] = self._prev_rewards
        saved["prev_dones"][env_ids] = self._prev_dones
        saved["is_start"][env_ids] = self._is_start

        self._num_envs = saved["num_envs"]
        self._memory = saved["memory"]
        self._memory_kv = saved["memory_kv"]
        self._read_kv = saved["read_kv"]
        self._key_cache = saved["key_cache"]
        self._value_cache = saved["value_cache"]
        self._hidden_history = saved["hidden_history"]
        self._token_valid = saved["token_valid"]
        self._positions = saved["positions"]
        self._prev_actions = saved["prev_actions"]
        self._prev_rewards = saved["prev_rewards"]
        self._prev_dones = saved["prev_dones"]
        self._is_start = saved["is_start"]

    # ------------------------------------------------------------------------------------------------------------
    # Reset semantics
    # ------------------------------------------------------------------------------------------------------------

    def reset_episode(self, dones: torch.Tensor | None = None) -> None:
        """Clear the short-term tokens and the KV cache. ``Z`` is **retained** (it is per trial, not per episode)."""
        if self._memory is None:
            return
        if dones is None:
            env_ids = torch.arange(self._num_envs, device=self._memory.device)
        else:
            env_ids = dones.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return
        for layer in range(self.num_layers):
            self._key_cache[layer][env_ids] = 0.0
            self._value_cache[layer][env_ids] = 0.0
        self._hidden_history[env_ids] = 0.0
        self._token_valid[env_ids] = False
        self._positions[env_ids] = 0
        self._prev_actions[env_ids] = 0.0
        self._prev_rewards[env_ids] = 0.0
        self._prev_dones[env_ids] = 0.0
        self._is_start[env_ids] = True
        self._last_hidden = None

    def reset_trial(self, trial_dones: torch.Tensor | None = None) -> None:
        """Reset the persistent memory ``Z`` back to the learned ``Z_init`` tokens for the given environments."""
        if self._memory is None:
            return
        if trial_dones is None:
            env_ids = torch.arange(self._num_envs, device=self._memory.device)
        else:
            env_ids = trial_dones.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return
        self._memory = self._memory.clone()
        self._memory[env_ids] = self.z_init.detach().to(device=self._memory.device, dtype=self._memory.dtype)
        self._memory_kv = None
        self._read_kv = None

    def reset(self, dones: torch.Tensor | None = None, trial_dones: torch.Tensor | None = None, **kwargs: Any) -> None:
        """Convenience wrapper: clear short-term state on ``dones``, clear ``Z`` on ``trial_dones``."""
        self.reset_episode(dones)
        if trial_dones is not None:
            self.reset_trial(trial_dones)

    def get_hidden_states(self) -> tuple[torch.Tensor | None, None]:
        """The persistent memory is the only state carried across episodes."""
        return self._memory, None

    def get_episode_hidden(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``H_e`` collected by the acting path so far, plus its validity mask (``[n, T + 1, d]``, ``[n, T + 1]``)."""
        if self._hidden_history is None:
            raise RuntimeError("get_episode_hidden() called before the first act(); no acting state allocated.")
        if env_ids is None:
            return self._hidden_history, self._token_valid
        return self._hidden_history[env_ids], self._token_valid[env_ids]

    # ------------------------------------------------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------------------------------------------------

    def _check_noise_std_compatible(self, state_dict: dict) -> None:
        """Fail loudly when a checkpoint's noise parameterization does not match ``noise_std_type``.

        The noise parameter differs in *name* (``std`` vs ``log_std``) and in *shape* (``[A]`` vs ``[d, A]``) across
        the three types, so a mismatched checkpoint would otherwise silently keep this module's freshly initialized
        noise (with ``strict=False``) instead of the trained one.
        """
        expected_name = _NOISE_PARAM_NAME[self.noise_std_type]
        present = [name for name in ("std", "log_std") if name in state_dict]
        if not present:
            return  # no noise parameter saved at all (e.g. a partial BC init); nothing to check
        if expected_name not in state_dict:
            raise ValueError(
                f"Checkpoint stores the action noise as '{present[0]}' but this policy was built with"
                f" noise_std_type='{self.noise_std_type}', which expects '{expected_name}'. Rebuild the policy with"
                " the checkpoint's noise_std_type."
            )
        expected_shape = tuple(getattr(self, expected_name).shape)
        actual_shape = tuple(state_dict[expected_name].shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"Checkpoint '{expected_name}' has shape {actual_shape} but noise_std_type="
                f"'{self.noise_std_type}' expects {expected_shape}. A 'gsde' checkpoint stores a"
                " [d_model, num_actions] matrix and a 'scalar'/'log' one a [num_actions] vector; they are not"
                " interchangeable."
            )

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Load the parameters of the actor-critic model.

        Args:
            state_dict: State dictionary of the model.
            strict: Whether to strictly enforce that the keys in `state_dict` match the keys returned by this module's
                :meth:`state_dict` function.

        Returns:
            Whether this training resumes a previous training. This flag is used by the :func:`load` function of
                :class:`OnPolicyRunner` to determine how to load further parameters (relevant for, e.g., distillation).
        """
        self._check_noise_std_compatible(state_dict)
        super().load_state_dict(state_dict, strict=strict)
        self._reset_runtime_state()
        if self.noise_std_type == "gsde":
            # Re-draw the exploration matrix from the loaded log_std (harmless for the density; see
            # _make_gsde_distribution) and drop any stale base distribution left over from before the load.
            self.distribution = self._make_gsde_distribution()
        return True
