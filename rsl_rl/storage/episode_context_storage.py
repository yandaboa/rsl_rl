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

With a memory policy (``memory_tokens > 0``) the storage additionally keeps, per environment, **which episode's
readouts** feed the memory of the rows it is holding:

* ``H`` snapshots, pushed by :class:`~rsl_rl.algorithms.EpisodeContextPPO` at every episode end (``[M + T, d]``:
  the ``M`` memory rows the episode ran with, then its readouts; detached). Within a rollout they are kept
  SPARSELY -- a handful of ``[n, M + T, d]`` chunks, one per step that had at least one done -- because a dense
  ``[num_envs, k, M + T, d]`` slot would be gigabytes at 16k
  environments. At :meth:`clear` the last snapshot of each environment is folded into the ONE dense per-
  environment slot that survives the rollout boundary: the source of the episode that is currently running.
* per-row ``episode_idx``, the acting policy's episode index inside its trial. ``0`` means "no source, read
  ``z_init``".

:meth:`recurrent_mini_batch_generator` turns that into per-segment source data and hands it to the policy inside
the same ``hidden_states`` carrier as the prefix; the writer itself runs in the policy's update graph.
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
        memory_tokens: int = 0,
        d_model: int = 0,
        num_eval_envs: int = 0,
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
            memory_tokens: ``M`` of the policy. ``0`` (the default) switches every memory buffer and every memory
                code path off -- the storage is then exactly the memory-free one.
            num_eval_envs: Trailing environments that act deterministically and are EXCLUDED from the update
                (see ``EpisodeContextPPO.eval_env_fraction``). Their rows are still collected, still get
                returns/advantages, and still feed the ring buffer -- the minibatch generator simply never
                hands them out, and the advantage normalization ignores them.
            d_model: Trunk width of the policy. Only read when ``memory_tokens > 0`` (it sizes the ``H``
                snapshots); one dense ``[num_envs, M + T, d]`` slot is allocated, which is the whole persistent
                cost of the feature.

        The remaining arguments are :class:`RolloutStorage`'s. ``carry_steps`` is not accepted: there is no
        deferred training unit here (a row is trained in the very next update), and the carry region would only
        shift the row indices the ring buffer is aligned against.
        """
        if kwargs.get("carry_steps"):
            raise ValueError("EpisodeContextRolloutStorage does not support a carry region (carry_steps > 0).")
        kwargs.pop("carry_steps", None)
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device=device, **kwargs)

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

        # -- cross-episode memory (all of this is absent with memory_tokens = 0) --
        self.num_memory_tokens = int(memory_tokens)
        self.d_model = int(d_model)
        self.has_memory = self.num_memory_tokens > 0
        # One row per memory token plus one per step of an episode, mirroring the policy's
        # ``hidden_history_span``: an ``H`` snapshot is the whole sequence the pass ran over.
        self.hidden_span = self.num_memory_tokens + self.max_episode_length
        if self.has_memory:
            if self.d_model <= 0:
                raise ValueError("memory_tokens > 0 needs the policy's d_model to size the H snapshots.")
            # Persistent per-environment state: the source episode of the episode each environment is CURRENTLY
            # in, as of the START of the rollout being held. Survives clear() -- that is the whole point: a
            # window can begin in the middle of an episode whose source ended one or more rollouts ago.
            self.source_hidden = torch.zeros(num_envs, self.hidden_span, self.d_model, device=device)
            self.source_valid = torch.zeros(num_envs, self.hidden_span, dtype=torch.bool, device=device)
            # False == "this environment's current episode is episode 0 of its trial" (read z_init).
            self.has_source = torch.zeros(num_envs, dtype=torch.bool, device=device)
            # Episode index inside the trial of every stored row; 0 means the row's episode has no source.
            self.row_episode_index = torch.zeros(
                self.num_transitions_per_env, num_envs, dtype=torch.long, device=device
            )
            self._pending_episode_index: torch.Tensor | None = None
            self._clear_snapshots()

    def _clear_snapshots(self) -> None:
        """Drop the rollout-local ``H`` snapshots (kept as a list of ``[n, ...]`` chunks, in step order)."""
        self._snapshot_hidden: list[torch.Tensor] = []
        self._snapshot_valid: list[torch.Tensor] = []
        self._snapshot_envs: list[torch.Tensor] = []
        self._snapshot_steps: list[torch.Tensor] = []
        self._snapshot_trial_end: list[torch.Tensor] = []

    # ------------------------------------------------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------------------------------------------------

    def actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """The actor's view of an observation: the concatenated policy groups, normalized as the actor sees them."""
        actor_obs = torch.cat([obs[group] for group in self.actor_obs_groups], dim=-1)
        return actor_obs if self.actor_obs_normalizer is None else self.actor_obs_normalizer(actor_obs)

    def stage_episode_index(self, episode_index: torch.Tensor) -> None:
        """Label the row that :meth:`add_transitions` is about to write with the acting episode index.

        Called by :class:`~rsl_rl.algorithms.EpisodeContextPPO` **before** the transition is stored and before
        the policy is reset, i.e. while the counter still describes the episode the row was acted in.
        """
        if self.has_memory:
            self._pending_episode_index = episode_index.reshape(-1).to(device=self.device, dtype=torch.long)

    def push_episode_hidden(
        self,
        env_ids: torch.Tensor,
        hidden: torch.Tensor,
        valid: torch.Tensor,
        trial_end: torch.Tensor,
    ) -> None:
        """Record the ``H`` of the episodes that end at the row currently being written.

        Args:
            env_ids: The environments that are done, ``[n]``.
            hidden: Their ``H`` ``[n, M + T, d]``, detached (the snapshot the writer will be re-run on).
            valid: Validity of every row of ``H``, ``[n, M + T]``.
            trial_end: Whether the finished episode was the LAST of its trial, ``[n]``. If it was, the episode
                that follows starts from ``z_init`` and this snapshot is not its source.

        Chunks are appended in step order, which is what makes "the last snapshot wins" a plain sequence of
        writes in :meth:`_commit_sources` (no duplicate-index scatter, no host sync).
        """
        if not self.has_memory or env_ids.numel() == 0:
            return
        self._snapshot_hidden.append(hidden.detach().to(self.device))
        self._snapshot_valid.append(valid.to(device=self.device, dtype=torch.bool))
        self._snapshot_envs.append(env_ids.to(device=self.device, dtype=torch.long))
        self._snapshot_steps.append(torch.full((env_ids.numel(),), self.row, dtype=torch.long, device=self.device))
        self._snapshot_trial_end.append(trial_end.reshape(-1).to(device=self.device, dtype=torch.bool))

    def add_transitions(self, transition: RolloutStorage.Transition) -> None:
        """Store the transition as usual, and push its frame (+ episode step) into the ring buffer."""
        slot = self.total_steps % self.ring_size
        self.frame_obs[slot].copy_(self.actor_obs(transition.observations))
        self.frame_positions[slot].copy_(self.episode_step)
        if self.has_memory:
            if self._pending_episode_index is None:
                raise RuntimeError(
                    "A memory storage needs stage_episode_index() before every add_transitions(): the row has to"
                    " know which episode of its trial it was acted in. EpisodeContextPPO.process_env_step does it."
                )
            self.row_episode_index[self.row].copy_(self._pending_episode_index)
            self._pending_episode_index = None

        super().add_transitions(transition)

        # The frame just stored acted at ``episode_step``; the NEXT one is one step later, or step 0 of a new
        # episode. This is exactly the policy's own counter update (increment on commit, zero on done).
        not_done = (~transition.dones.reshape(-1).bool()).to(self.episode_step.dtype)
        self.episode_step = (self.episode_step + 1) * not_done
        self.total_steps += 1

    # ------------------------------------------------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------------------------------------------------

    def compute_returns(
        self, last_values: torch.Tensor, gamma: float, lam: float, normalize_advantage: bool = True
    ) -> None:
        """Stock GAE, but the advantage normalization statistics come from the TRAINING pool only.

        GAE itself still runs over every environment (harmless, and it keeps the tensors uniform); the eval
        rows are simply never trained on, so letting them move the mean/std of the rows that are would be a
        silent leak of the deterministic pool into the update.
        """
        if self.num_eval_envs == 0:
            super().compute_returns(last_values, gamma, lam, normalize_advantage)
            return
        super().compute_returns(last_values, gamma, lam, normalize_advantage=False)
        if normalize_advantage:
            train_advantages = self.advantages[:, : self.num_train_envs]
            self.advantages = (self.advantages - train_advantages.mean()) / (train_advantages.std() + 1e-8)

    def clear(self) -> None:
        """Free the buffer for the next rollout, first folding this rollout's ``H`` snapshots into the sources.

        The persistent slot must describe the rollout that is about to be COLLECTED, so it is advanced here --
        after the update has consumed the generator, which needs the state as it was at the rollout's start.
        """
        if self.has_memory:
            self._commit_sources()
            self._clear_snapshots()
        super().clear()

    def _commit_sources(self) -> None:
        """Make each environment's LAST finished episode of this rollout the source of the episode now running."""
        for env_ids, hidden, valid, trial_end in zip(
            self._snapshot_envs, self._snapshot_hidden, self._snapshot_valid, self._snapshot_trial_end
        ):
            # In step order, so a later done simply overwrites an earlier one. An environment appears at most
            # once per chunk (one done per step), so no index inside a single write is duplicated.
            self.source_hidden[env_ids] = hidden
            self.source_valid[env_ids] = valid
            # A trial end closes the chain: the next episode is episode 0 and reads z_init.
            self.has_source[env_ids] = ~trial_end

    def _snapshot_index(self) -> tuple[torch.Tensor, ...]:
        """Flatten the rollout-local snapshots and index them by ``(row, environment)``.

        Returns:
            ``lookup`` ``[num_steps, num_envs]`` (the flat snapshot index of the done at that row, ``-1`` for no
            done) plus the flattened ``hidden``, ``valid`` and ``trial_end``.
        """
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

        A segment is a maximal contiguous run of rows of ONE episode inside ``[prefix | window]``. Segment ``0``
        holds the prefix (whose rows belong to the first window row's episode, and are inert for any other one)
        plus every window row before the first episode start; every later segment opens at a window row whose
        episode step is ``0``. The source of a segment is

        * the environment's persistent source (the state at the START of this rollout) when the segment opens at
          window row ``0`` or is segment ``0`` -- its episode began in an earlier rollout, or right at the
          boundary,
        * otherwise the ``H`` snapshotted at the row immediately before it, i.e. the episode that just ended.

        A segment whose first window row carries ``episode_idx == 0`` has NO source: it is the first episode of a
        trial, so its memory is ``z_init``.

        Returns:
            ``source_hidden`` ``[B, n_seg, M + T, d]``, ``source_valid`` ``[B, n_seg, M + T]``,
            ``has_source`` ``[B, n_seg]`` and ``memory_segments`` ``[P + W, B]``.
        """
        num_steps, batch_size = window_positions.shape
        device = self.device
        starts = window_positions == 0  # [W, B]
        window_segments = starts.long().cumsum(dim=0)  # prefix and everything before the first start are 0
        num_segments = int(window_segments.max().item()) + 1

        env_index = torch.arange(self.num_envs, device=device)[envs]  # [B]
        rows = torch.arange(num_steps, device=device).unsqueeze(1)  # [W, 1]
        # First window row of every segment; ``num_steps`` marks "this segment has no window row", which can only
        # happen to segment 0 when the window opens exactly on an episode start.
        first_row = torch.stack([
            torch.where(window_segments == segment, rows, torch.full_like(rows, num_steps)).min(dim=0).values
            for segment in range(num_segments)
        ])  # [n_seg, B]
        has_rows = first_row < num_steps
        # Only a segment that opens strictly inside the window has its source among THIS rollout's snapshots.
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
        # The authoritative statement of "this episode has a source" is the acting policy's own episode index,
        # recorded per row. It agrees with the trial_end flag above; keeping both is what makes a mismatch a
        # test failure rather than a silently wrong memory.
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
        * the SAME :class:`EpisodeContextPrefix` in both hidden-state slots. The actor consumes it in
          ``policy.act``; the critic slot is what ``ppo.py`` hands to ``policy.evaluate``, which a
          ``separate_trunk`` critic needs to run its own windowed pass (the ``privileged`` critic ignores it and
          the ``shared_trunk`` one reuses the actor's ``h``). With a memory policy the prefix additionally
          carries the per-segment source episodes and the explicit segment ids (see :meth:`_segment_sources`),
        * ``masks_batch = None``: there is nothing to mask, which keeps the stock loss reductions exact.

        With an eval pool (``num_eval_envs > 0``) the chunks cover ``[0, num_train_envs)`` only: those
        environments acted deterministically, so their rows are off-policy for this update and are dropped.
        """
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        if self.step != self.num_collect_steps:
            raise ValueError(
                f"The rollout is incomplete ({self.step} of {self.num_collect_steps} steps). The episode-context"
                " generator reconstructs a contiguous window and cannot skip rows."
            )
        num_mini_batches = min(num_mini_batches, self.num_train_envs)
        mini_batch_size = self.num_train_envs // num_mini_batches
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
        num_prefix = prefix_obs.shape[0]
        snapshots = self._snapshot_index() if self.has_memory else None

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                # The last chunk absorbs the remainder, so that every environment is trained exactly once even
                # when num_envs is not a multiple of num_mini_batches.
                stop = self.num_train_envs if i == num_mini_batches - 1 else (i + 1) * mini_batch_size
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

                hidden_state_a_batch = EpisodeContextPrefix(
                    obs=prefix_obs[:, envs],
                    positions=prefix_positions[:, envs],
                    window_positions=window_positions[:, envs],
                    **memory_fields,
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
                    (hidden_state_a_batch, hidden_state_a_batch),
                    None,
                )
