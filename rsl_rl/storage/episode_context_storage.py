# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout storage for the single-episode context transformer (:class:`ActorCriticEpisodeContext`).

The problem it solves: the actor's context (``L`` frames) is longer than the rollout (``num_steps_per_env``
frames), so re-inferring a window at update time needs frames collected *before* this rollout. Materializing the
context per row would cost ``L x`` the observation memory; instead every frame is stored exactly ONCE, in a
per-environment ring buffer of ``num_steps_per_env + min(L, T)`` slots, and the update reads a contiguous
``[prefix | window]`` slice out of it.

Everything else is the stock :class:`RolloutStorage`: the same rows, the same GAE, the same returns/advantages.
Only :meth:`recurrent_mini_batch_generator` is replaced, so that PPO's unmodified recurrent branch drives the
whole thing:

* the batch is time-major ``[W, B, ...]`` (window x environment chunk) -- exactly the layout the recurrent path
  already expects, and every stock loss line broadcasts over it unchanged,
* the prefix (and both parts' episode-step metadata) rides in the ``hidden_states`` slot as an
  :class:`EpisodeContextPrefix`; ``ppo.py`` never looks inside it, it just hands it back to ``policy.act``,
* every collected row is trained exactly once, in the update immediately after its rollout. Nothing is deferred,
  nothing is dropped, every lag is 0.
"""

from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.modules.actor_critic_episode_context import EpisodeContextPrefix
from rsl_rl.storage.rollout_storage import RolloutStorage


class EpisodeContextRolloutStorage(RolloutStorage):
    """:class:`RolloutStorage` plus a per-environment ring buffer of raw actor-observation frames."""

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        device: str = "cpu",
        actor_obs_groups: list[str] | None = None,
        context_length: int = 80,
        max_episode_length: int = 80,
        num_layers: int = 1,
        actor_obs_normalizer: torch.nn.Module | None = None,
        **kwargs,
    ) -> None:
        """
        Args:
            actor_obs_groups: Observation groups the actor's tokens are built from (``obs_groups["policy"]``).
                Defaults to every group in ``obs``.
            actor_obs_normalizer: The policy's observation normalizer. Frames are stored **normalized**, at
                collection time, i.e. exactly as the acting path saw them: a prefix frame is usually read back
                one or more rollouts (and therefore one or more normalizer commits) later, and re-normalizing it
                with the statistics of that later update would rebuild a different token than the behavior
                policy used. ``None`` means no normalization.
            context_length: ``L`` of the policy.
            max_episode_length: ``T`` of the environment.
            num_layers: Number of transformer blocks of the policy. It sizes the prefix: a row's receptive field
                is ``num_layers * (span - 1)`` frames, not ``span - 1`` (see
                :attr:`ActorCriticEpisodeContext.context_prefix_length`).

        The remaining arguments are :class:`RolloutStorage`'s. ``carry_steps`` is not accepted: there is no
        deferred training unit here (a row is trained in the very next update), and the carry region would only
        shift the row indices the ring buffer is aligned against.
        """
        if kwargs.get("carry_steps"):
            raise ValueError("EpisodeContextRolloutStorage does not support a carry region (carry_steps > 0).")
        kwargs.pop("carry_steps", None)
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device=device, **kwargs)

        self.actor_obs_groups = list(obs.keys()) if actor_obs_groups is None else list(actor_obs_groups)
        self.actor_obs_normalizer = actor_obs_normalizer
        self.context_length = int(context_length)
        self.max_episode_length = int(max_episode_length)
        # Reachable history, mirroring the policy: an episode is at most T frames long.
        self.context_span = max(1, min(self.context_length, self.max_episode_length))
        # Frames of history a window row can need. Mirrors ActorCriticEpisodeContext.context_prefix_length:
        # ``num_layers`` blocks reach ``num_layers * (span - 1)`` frames back, capped at one episode.
        self.prefix_length = min(self.max_episode_length - 1, int(num_layers) * (self.context_span - 1))
        # One rollout of new frames plus that history, i.e. the exact bound plus one spare slot. With the target
        # configuration (L = T = 80, W = 32) this is 32 + 80 = 112.
        self.ring_size = self.num_collect_steps + self.prefix_length + 1

        self.actor_obs_dim = sum(int(obs[group].shape[-1]) for group in self.actor_obs_groups)
        self.frame_obs = torch.zeros(self.ring_size, num_envs, self.actor_obs_dim, device=device)
        # Episode step of every stored frame (what the policy's positional embedding and attention span key on).
        self.frame_positions = torch.zeros(self.ring_size, num_envs, dtype=torch.long, device=device)
        # Running episode step per environment, kept in lockstep with the policy's own counter.
        self.episode_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Total frames ever written; the ring slot of global step ``g`` is ``g % ring_size``.
        self.total_steps = 0

    # ------------------------------------------------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------------------------------------------------

    def actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """The actor's view of an observation: the concatenated policy groups, normalized as the actor sees them."""
        actor_obs = torch.cat([obs[group] for group in self.actor_obs_groups], dim=-1)
        return actor_obs if self.actor_obs_normalizer is None else self.actor_obs_normalizer(actor_obs)

    def add_transitions(self, transition: RolloutStorage.Transition) -> None:
        """Store the transition as usual, and push its frame (+ episode step) into the ring buffer."""
        slot = self.total_steps % self.ring_size
        self.frame_obs[slot].copy_(self.actor_obs(transition.observations))
        self.frame_positions[slot].copy_(self.episode_step)

        super().add_transitions(transition)

        # The frame just stored acted at ``episode_step``; the NEXT one is one step later, or step 0 of a new
        # episode. This is exactly the policy's own counter update (increment on commit, zero on done).
        not_done = (~transition.dones.reshape(-1).bool()).to(self.episode_step.dtype)
        self.episode_step = (self.episode_step + 1) * not_done
        self.total_steps += 1

    # ------------------------------------------------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------------------------------------------------

    def context_slice(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The ``[prefix | window]`` frames of the rollout just collected, in temporal order.

        Returns:
            ``prefix_obs`` ``[P, N, obs_dim]`` (normalized, see :meth:`actor_obs`), ``prefix_positions``
            ``[P, N]`` and ``window_positions``
            ``[W, N]``. Prefix rows older than the very first collected frame (only possible in the first
            rollouts of a run) are zero-filled; the policy's mask never reaches them, because a row's attention
            span is bounded by its own episode step, which is bounded by the number of frames ever collected.
        """
        window = self.step
        prefix = self.prefix_length
        first = self.total_steps - window  # global step of the window's first row
        globals_ = torch.arange(first - prefix, first + window, device=self.device)
        slots = torch.remainder(globals_, self.ring_size)
        positions = self.frame_positions[slots]
        return self.frame_obs[slots[:prefix]], positions[:prefix], positions[prefix:]

    def recurrent_mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        """Environment-chunk minibatches shaped for PPO's recurrent branch.

        Yields the stock 10-tuple, with

        * the observation/action/target tensors as time-major ``[W, B, ...]`` slices of the rollout (no padding,
          no trajectory splitting: every row of every environment is live and is trained exactly once),
        * an :class:`EpisodeContextPrefix` in the actor's hidden-state slot and ``None`` in the critic's (the
          critic is a context-free MLP),
        * ``masks_batch = None``: there is nothing to mask, which keeps the stock loss reductions exact.
        """
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        if self.step != self.num_collect_steps:
            raise ValueError(
                f"The rollout is incomplete ({self.step} of {self.num_collect_steps} steps). The episode-context"
                " generator reconstructs a contiguous window and cannot skip rows."
            )
        num_mini_batches = min(num_mini_batches, self.num_envs)
        mini_batch_size = self.num_envs // num_mini_batches
        if mini_batch_size < 2 or self.num_collect_steps < 2:
            # ppo.py reduces with ``torch.squeeze(advantages_batch)``, which drops EVERY size-one dimension --
            # a one-environment (or one-step) minibatch would silently broadcast the surrogate into a [W, W]
            # outer product instead of failing. Refuse instead.
            raise ValueError(
                f"An episode-context minibatch must hold at least 2 environments and 2 steps (got"
                f" {mini_batch_size} envs x {self.num_collect_steps} steps). Lower num_mini_batches."
            )

        # The slice is identical in every epoch (only the parameters move), so it is gathered once.
        prefix_obs, prefix_positions, window_positions = self.context_slice()

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                # The last chunk absorbs the remainder, so that every environment is trained exactly once even
                # when num_envs is not a multiple of num_mini_batches.
                stop = self.num_envs if i == num_mini_batches - 1 else (i + 1) * mini_batch_size
                envs = slice(start, stop)

                hidden_state_a_batch = EpisodeContextPrefix(
                    obs=prefix_obs[:, envs],
                    positions=prefix_positions[:, envs],
                    window_positions=window_positions[:, envs],
                )

                yield (
                    self.observations[:, envs],
                    self.actions[:, envs],
                    self.values[:, envs],
                    self.advantages[:, envs],
                    self.returns[:, envs],
                    self.actions_log_prob[:, envs],
                    self.mu[:, envs],
                    self.sigma[:, envs],
                    (hidden_state_a_batch, None),
                    None,
                )
