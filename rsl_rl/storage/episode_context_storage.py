# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout storage for the single-episode context transformer (:class:`~rsl_rl.models.EpisodeContextModel`).

The actor's context (``L`` frames) is longer than the rollout (``num_steps_per_env``), so re-inferring a window at
update time needs frames collected BEFORE this rollout. Every frame is stored once, in a per-environment ring of
``num_steps_per_env + min(L, T)`` slots, and the update reads a contiguous ``[prefix | window]`` slice out of it.

Everything else is the stock :class:`RolloutStorage`. Only :meth:`recurrent_mini_batch_generator` is replaced: it
yields time-major ``[W, B, ...]`` environment chunks with the :class:`EpisodeContextPrefix` in BOTH hidden-state
slots and ``masks=None``. Every collected row is trained exactly once, in the update right after its rollout.

With ``memory_tokens > 0`` the storage also keeps, per environment, which episode's ``H`` feeds the memory of
the rows it holds: sparse ``H`` snapshots pushed at every done, folded into a persistent per-env source slot on
:meth:`clear`, and re-sliced per segment by :meth:`_segment_sources`.
"""

from __future__ import annotations

import torch
from collections.abc import Callable, Generator
from tensordict import TensorDict

from rsl_rl.models.episode_context_model import EpisodeContextPrefix
from rsl_rl.storage.rollout_storage import RolloutStorage


class EpisodeContextRolloutStorage(RolloutStorage):
    """:class:`RolloutStorage` plus a per-environment ring buffer of normalized actor-observation frames."""

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
        actor_obs_groups: list[str] | None = None,
        context_length: int = 80,
        max_episode_length: int = 80,
        num_layers: int = 1,
        actor_obs_normalizer: Callable[[torch.Tensor], torch.Tensor] | None = None,
        memory_tokens: int = 0,
        d_model: int = 0,
        num_eval_envs: int = 0,
    ) -> None:
        """
        Args:
            actor_obs_groups: Observation groups the actor's tokens are built from. Defaults to every group.
            actor_obs_normalizer: The model's :attr:`~rsl_rl.models.EpisodeContextModel.frame_normalizer`. Frames
                are stored NORMALIZED at collection time: a prefix frame is read back one or more normalizer
                commits later, and re-normalizing it then would rebuild a different token than the one acted on.
            context_length / max_episode_length / num_layers: ``L``, ``T`` and the trunk depth (which sizes the
                prefix: a row's receptive field is ``num_layers * (span - 1)`` frames).
            memory_tokens / d_model: ``M`` and the trunk width; ``M = 0`` allocates nothing memory-related.
            num_eval_envs: Trailing environments excluded from the update (see ``eval_env_fraction``).
        """
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device=device)

        self.num_eval_envs = int(num_eval_envs)
        assert 0 <= self.num_eval_envs < num_envs, (
            f"num_eval_envs={self.num_eval_envs} does not leave a training pool out of {num_envs} environments."
        )
        # The eval pool is the TRAILING block, so the training pool stays one contiguous slice.
        self.num_train_envs = num_envs - self.num_eval_envs

        self.actor_obs_groups = list(obs.keys()) if actor_obs_groups is None else list(actor_obs_groups)
        self.actor_obs_normalizer = actor_obs_normalizer
        self.context_length = int(context_length)
        self.max_episode_length = int(max_episode_length)
        self.context_span = max(1, min(self.context_length, self.max_episode_length))
        self.prefix_length = min(self.max_episode_length - 1, int(num_layers) * (self.context_span - 1))
        # One rollout of new frames plus that history, plus one spare slot (L = T = 80, W = 32 -> 112).
        self.ring_size = self.num_transitions_per_env + self.prefix_length + 1

        # Trial boundary per row (== dones unless the env publishes ``trial_done``): what GAE treats as terminal.
        self.trial_dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=device).byte()

        self.actor_obs_dim = sum(int(obs[group].shape[-1]) for group in self.actor_obs_groups)
        self.frame_obs = torch.zeros(self.ring_size, num_envs, self.actor_obs_dim, device=device)
        self.frame_positions = torch.zeros(self.ring_size, num_envs, dtype=torch.long, device=device)
        # Running episode step per environment, in lockstep with the model's own counter.
        self.episode_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        # Total frames ever written; the ring slot of global step ``g`` is ``g % ring_size``.
        self.total_steps = 0
        # Set by the algorithm when the auxiliary next-state loss is on.
        self.with_next_state = False
        self.next_state_target_group: str | None = None
        self.next_state_target_dim = 0
        self.frame_target: torch.Tensor | None = None

        self.num_memory_tokens = int(memory_tokens)
        self.d_model = int(d_model)
        self.has_memory = self.num_memory_tokens > 0
        self.hidden_span = self.num_memory_tokens + self.max_episode_length
        if self.has_memory:
            if self.d_model <= 0:
                raise ValueError("memory_tokens > 0 needs the policy's d_model to size the H snapshots.")
            # Persistent per-env state: the source episode of the episode each env is in at the START of the
            # rollout being held. Survives clear().
            self.source_hidden = torch.zeros(num_envs, self.hidden_span, self.d_model, device=device)
            self.source_valid = torch.zeros(num_envs, self.hidden_span, dtype=torch.bool, device=device)
            self.has_source = torch.zeros(num_envs, dtype=torch.bool, device=device)
            # Episode index inside the trial of every stored row; 0 means the row's episode has no source.
            self.row_episode_index = torch.zeros(
                self.num_transitions_per_env, num_envs, dtype=torch.long, device=device
            )
            self._pending_episode_index: torch.Tensor | None = None
            self._clear_snapshots()

    @property
    def collected_observations(self) -> TensorDict:
        """The observations of the rollout just collected (what a deferred normalizer commit consumes)."""
        return self.observations

    def enable_next_state_target(self, group: str, target_dim: int) -> None:
        """Also ring-buffer the RAW ``obs[group]`` frames (privileged, never normalized here)."""
        assert target_dim > 0, f"next-state target group '{group}' needs a positive width, got {target_dim}."
        self.next_state_target_group = group
        self.next_state_target_dim = int(target_dim)
        self.frame_target = torch.zeros(self.ring_size, self.num_envs, self.next_state_target_dim, device=self.device)

    def _clear_snapshots(self) -> None:
        self._snapshot_hidden: list[torch.Tensor] = []
        self._snapshot_valid: list[torch.Tensor] = []
        self._snapshot_envs: list[torch.Tensor] = []
        self._snapshot_steps: list[torch.Tensor] = []
        self._snapshot_trial_end: list[torch.Tensor] = []

    # ------------------------------------------------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------------------------------------------------

    def actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """The actor's view of an observation: the concatenated actor groups, normalized as the actor sees them."""
        actor_obs = torch.cat([obs[group] for group in self.actor_obs_groups], dim=-1)
        return actor_obs if self.actor_obs_normalizer is None else self.actor_obs_normalizer(actor_obs)

    def stage_episode_index(self, episode_index: torch.Tensor) -> None:
        """Label the row :meth:`add_transition` is about to write with the acting episode index (call BEFORE
        the transition is stored and before the model is reset)."""
        if self.has_memory:
            self._pending_episode_index = episode_index.reshape(-1).to(device=self.device, dtype=torch.long)

    def push_episode_hidden(
        self,
        env_ids: torch.Tensor,
        hidden: torch.Tensor,
        valid: torch.Tensor,
        trial_end: torch.Tensor,
    ) -> None:
        """Record the ``H`` ``[n, M + T, d]`` of the episodes of ``env_ids`` that end at the row being written.
        ``trial_end`` says whether that episode was the last of its trial (then it is nobody's source)."""
        if not self.has_memory or env_ids.numel() == 0:
            return
        self._snapshot_hidden.append(hidden.detach().to(self.device))
        self._snapshot_valid.append(valid.to(device=self.device, dtype=torch.bool))
        self._snapshot_envs.append(env_ids.to(device=self.device, dtype=torch.long))
        self._snapshot_steps.append(torch.full((env_ids.numel(),), self.step, dtype=torch.long, device=self.device))
        self._snapshot_trial_end.append(trial_end.reshape(-1).to(device=self.device, dtype=torch.bool))

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        """Store the transition as usual, and push its frame (+ episode step) into the ring buffer."""
        slot = self.total_steps % self.ring_size
        self.frame_obs[slot].copy_(self.actor_obs(transition.observations))
        self.frame_positions[slot].copy_(self.episode_step)
        if self.frame_target is not None:
            self.frame_target[slot].copy_(transition.observations[self.next_state_target_group])
        if self.has_memory:
            if self._pending_episode_index is None:
                raise RuntimeError(
                    "A memory storage needs stage_episode_index() before every add_transition(): the row has to"
                    " know which episode of its trial it was acted in. EpisodeContextPPO.process_env_step does it."
                )
            self.row_episode_index[self.step].copy_(self._pending_episode_index)
            self._pending_episode_index = None

        trial_dones = getattr(transition, "trial_dones", None)
        self.trial_dones[self.step].copy_((transition.dones if trial_dones is None else trial_dones).view(-1, 1))
        super().add_transition(transition)

        # Same counter update as the model's: increment on commit, zero on done.
        not_done = (~transition.dones.reshape(-1).bool()).to(self.episode_step.dtype)
        self.episode_step = (self.episode_step + 1) * not_done
        self.total_steps += 1

    # ------------------------------------------------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------------------------------------------------

    def clear(self) -> None:
        """Free the buffer for the next rollout, first folding this rollout's ``H`` snapshots into the sources."""
        if self.has_memory:
            self._commit_sources()
            self._clear_snapshots()
        super().clear()

    def _commit_sources(self) -> None:
        """Make each env's LAST finished episode of this rollout the source of the episode now running."""
        for env_ids, hidden, valid, trial_end in zip(
            self._snapshot_envs, self._snapshot_hidden, self._snapshot_valid, self._snapshot_trial_end
        ):
            # In step order, so a later done overwrites an earlier one; one done per env per chunk.
            self.source_hidden[env_ids] = hidden
            self.source_valid[env_ids] = valid
            self.has_source[env_ids] = ~trial_end

    def _snapshot_index(self) -> tuple[torch.Tensor, ...]:
        """Flatten the rollout-local snapshots; ``lookup[step, env]`` is the flat index of the done there (-1 none)."""
        if not self._snapshot_envs:
            empty = torch.zeros(0, self.hidden_span, self.d_model, device=self.device)
            lookup = torch.full((self.num_transitions_per_env, self.num_envs), -1, dtype=torch.long, device=self.device)
            return (
                lookup,
                empty,
                torch.zeros(0, self.hidden_span, dtype=torch.bool, device=self.device),
                torch.zeros(0, dtype=torch.bool, device=self.device),
            )
        envs = torch.cat(self._snapshot_envs)
        steps = torch.cat(self._snapshot_steps)
        hidden = torch.cat(self._snapshot_hidden)
        valid = torch.cat(self._snapshot_valid)
        trial_end = torch.cat(self._snapshot_trial_end)
        lookup = torch.full((self.num_transitions_per_env, self.num_envs), -1, dtype=torch.long, device=self.device)
        lookup[steps, envs] = torch.arange(envs.numel(), device=self.device)
        return lookup, hidden, valid, trial_end

    def _segment_sources(
        self,
        envs: slice,
        window_positions: torch.Tensor,
        lookup: torch.Tensor,
        hidden: torch.Tensor,
        valid: torch.Tensor,
        trial_end: torch.Tensor,
        num_prefix: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-segment memory sources of one environment chunk.

        Segment 0 holds the prefix plus every window row before the first episode start; every later segment
        opens at a window row with episode step 0. A segment opening at window row 0 (or segment 0) sources from
        the persistent slot, otherwise from the ``H`` snapshotted at the row right before it. A segment whose
        first row carries ``episode_idx == 0`` has no source (``z_init``).

        Returns ``source_hidden [B, n_seg, M+T, d]``, ``source_valid [B, n_seg, M+T]``, ``has_source [B, n_seg]``
        and ``memory_segments [P + W, B]``.
        """
        num_steps, batch_size = window_positions.shape
        device = self.device
        starts = window_positions == 0  # [W, B]
        window_segments = starts.long().cumsum(dim=0)
        num_segments = int(window_segments.max().item()) + 1

        env_index = torch.arange(self.num_envs, device=device)[envs]  # [B]
        rows = torch.arange(num_steps, device=device).unsqueeze(1)  # [W, 1]
        # First window row of every segment; ``num_steps`` marks "no window row" (segment 0 only).
        first_row = torch.stack([
            torch.where(window_segments == segment, rows, torch.full_like(rows, num_steps)).min(dim=0).values
            for segment in range(num_segments)
        ])  # [n_seg, B]
        has_rows = first_row < num_steps
        from_snapshot = has_rows & (first_row > 0)
        flat_index = lookup[(first_row - 1).clamp(min=0), env_index.unsqueeze(0).expand_as(first_row)]
        from_snapshot = from_snapshot & (flat_index >= 0)
        gather = flat_index.clamp(min=0)

        persistent = self.source_hidden[env_index].unsqueeze(0).expand(num_segments, -1, -1, -1)
        pick = from_snapshot.view(num_segments, batch_size, 1, 1)
        source_hidden = torch.where(pick, hidden[gather] if hidden.numel() else persistent, persistent)
        source_valid = torch.where(
            pick.squeeze(-1),
            valid[gather] if valid.numel() else self.source_valid[env_index].unsqueeze(0).expand(num_segments, -1, -1),
            self.source_valid[env_index].unsqueeze(0).expand(num_segments, -1, -1),
        )
        has_source = torch.where(
            from_snapshot,
            ~trial_end[gather] if trial_end.numel() else self.has_source[env_index].unsqueeze(0).expand_as(has_rows),
            self.has_source[env_index].unsqueeze(0).expand_as(has_rows),
        )
        # The acting model's own episode index is authoritative; it must agree with the trial_end flag.
        segment_episode_index = self.row_episode_index[first_row.clamp(max=num_steps - 1), env_index.unsqueeze(0)]
        has_source = has_source & torch.where(has_rows, segment_episode_index > 0, has_source)

        memory_segments = torch.zeros(num_prefix + num_steps, batch_size, dtype=torch.long, device=device)
        memory_segments[num_prefix:] = window_segments
        return (
            source_hidden.transpose(0, 1),
            source_valid.transpose(0, 1),
            has_source.transpose(0, 1),
            memory_segments,
        )

    def context_slice(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(prefix_obs [P, N, obs], prefix_positions [P, N], window_positions [W, N])`` of the rollout just
        collected. Prefix rows older than the first collected frame are zero-filled; the mask never reaches them."""
        window = self.step
        prefix = self.prefix_length
        first = self.total_steps - window  # global step of the window's first row
        globals_ = torch.arange(first - prefix, first + window, device=self.device)
        slots = torch.remainder(globals_, self.ring_size)
        positions = self.frame_positions[slots]
        return self.frame_obs[slots[:prefix]], positions[:prefix], positions[prefix:]

    def next_state_slice(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """``(next_delta [W, N, obs], next_valid [W, N], next_target_delta [W, N, target] | None)``: successor
        deltas of the window rows. Valid only if the successor is stored AND continues the same episode."""
        window = self.step
        first = self.total_steps - window
        globals_ = torch.arange(first, first + window, device=self.device)
        slots = torch.remainder(globals_, self.ring_size)
        next_slots = torch.remainder(globals_ + 1, self.ring_size)
        obs, positions = self.frame_obs[slots], self.frame_positions[slots]
        next_obs, next_positions = self.frame_obs[next_slots], self.frame_positions[next_slots]
        stored = (globals_ + 1 <= self.total_steps - 1).unsqueeze(1).expand(-1, self.num_envs)
        valid = stored & (next_positions == positions + 1)
        target_delta = None
        if self.frame_target is not None:
            target_delta = (self.frame_target[next_slots] - self.frame_target[slots]) * valid.unsqueeze(-1)
        return (next_obs - obs) * valid.unsqueeze(-1), valid, target_delta

    def mini_batch_env_chunks(self, num_mini_batches: int) -> list[tuple[int, int]]:
        """``[start, stop)`` training-environment ranges of the LOGICAL minibatches, in the order handed out."""
        num_mini_batches = min(num_mini_batches, self.num_train_envs)
        mini_batch_size = self.num_train_envs // num_mini_batches
        # The last chunk absorbs the remainder, so every environment is trained exactly once.
        return [
            (i * mini_batch_size, self.num_train_envs if i == num_mini_batches - 1 else (i + 1) * mini_batch_size)
            for i in range(num_mini_batches)
        ]

    def recurrent_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8, grad_accumulation_steps: int = 1
    ) -> Generator[RolloutStorage.Batch, None, None]:
        """Environment-chunk minibatches: time-major ``[W, B, ...]`` slices, the prefix in BOTH hidden-state
        slots, ``masks=None``. Eval environments are never handed out. With ``grad_accumulation_steps > 1``
        every logical minibatch is split into that many equal contiguous micro-batches."""
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        if self.step != self.num_transitions_per_env:
            raise ValueError(
                f"The rollout is incomplete ({self.step} of {self.num_transitions_per_env} steps). The"
                " episode-context generator reconstructs a contiguous window and cannot skip rows."
            )
        assert grad_accumulation_steps >= 1, f"grad_accumulation_steps must be >= 1, got {grad_accumulation_steps}."
        chunks = self.mini_batch_env_chunks(num_mini_batches)
        mini_batch_size = self.num_train_envs // len(chunks)
        if mini_batch_size < 2 or self.num_transitions_per_env < 2:
            # ``torch.squeeze(advantages)`` drops EVERY size-one dim; a 1-env (or 1-step) minibatch would
            # silently broadcast the surrogate into an outer product.
            raise ValueError(
                f"An episode-context minibatch must hold at least 2 environments and 2 steps (got"
                f" {mini_batch_size} envs x {self.num_transitions_per_env} steps). Lower num_mini_batches."
            )
        if grad_accumulation_steps > 1:
            num_micro_batches = len(chunks) * grad_accumulation_steps
            # Equal micro-batches are what makes the average of their means the mean of the minibatch.
            assert self.num_train_envs % num_micro_batches == 0, (
                f"{self.num_train_envs} training environments do not split evenly into {len(chunks)} minibatches"
                f" x {grad_accumulation_steps} accumulation steps."
            )
            micro_batch_size = self.num_train_envs // num_micro_batches
            assert micro_batch_size >= 2, (
                f"A micro-batch must hold at least 2 environments (got {micro_batch_size}). Lower"
                " grad_accumulation_steps."
            )
            chunks = [
                (start + step * micro_batch_size, start + (step + 1) * micro_batch_size)
                for start, _ in chunks
                for step in range(grad_accumulation_steps)
            ]

        # The slice is identical in every epoch (only the parameters move), so it is gathered once.
        prefix_obs, prefix_positions, window_positions = self.context_slice()
        num_prefix = prefix_obs.shape[0]
        snapshots = self._snapshot_index() if self.has_memory else None
        next_delta, next_valid, next_target_delta = (
            self.next_state_slice() if self.with_next_state else (None, None, None)
        )

        for _ in range(num_epochs):
            for start, stop in chunks:
                envs = slice(start, stop)

                memory_fields: dict[str, torch.Tensor] = {}
                if self.has_memory:
                    source_hidden, source_valid, has_source, memory_segments = self._segment_sources(
                        envs, window_positions[:, envs], *snapshots, num_prefix=num_prefix
                    )
                    memory_fields = dict(
                        source_hidden=source_hidden,
                        source_valid=source_valid,
                        segment_has_source=has_source,
                        memory_segments=memory_segments,
                    )

                next_state_fields: dict[str, torch.Tensor] = {}
                if next_delta is not None:
                    next_state_fields = dict(next_delta=next_delta[:, envs], next_valid=next_valid[:, envs])
                    if next_target_delta is not None:
                        next_state_fields["next_target_delta"] = next_target_delta[:, envs]

                prefix = EpisodeContextPrefix(
                    obs=prefix_obs[:, envs],
                    positions=prefix_positions[:, envs],
                    window_positions=window_positions[:, envs],
                    **memory_fields,
                    **next_state_fields,
                )

                yield RolloutStorage.Batch(
                    observations=self.observations[:, envs],
                    actions=self.actions[:, envs],
                    values=self.values[:, envs],
                    advantages=self.advantages[:, envs],
                    returns=self.returns[:, envs],
                    old_actions_log_prob=self.actions_log_prob[:, envs],
                    old_distribution_params=tuple(p[:, envs] for p in self.distribution_params),  # type: ignore
                    hidden_states=(prefix, prefix),  # type: ignore
                    masks=None,
                )
