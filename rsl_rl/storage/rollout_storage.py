# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.networks import HiddenState
from rsl_rl.utils import split_and_pad_trajectories


def _broadcast_mask(mask: torch.Tensor, ndim: int) -> torch.Tensor:
    """View a ``[S, B]`` mask so that it broadcasts against a ``[S, B, ...]`` tensor with ``ndim`` dimensions."""
    return mask.view(*mask.shape, *([1] * (ndim - mask.dim())))


def _mask_leading(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero every ``[S, B, ...]`` row whose ``[S, B]`` mask entry is ``False``."""
    return tensor * _broadcast_mask(mask, tensor.dim()).to(tensor.dtype)


class RolloutStorage:
    class Transition:
        def __init__(self) -> None:
            self.observations: TensorDict | None = None
            self.actions: torch.Tensor | None = None
            self.privileged_actions: torch.Tensor | None = None
            self.rewards: torch.Tensor | None = None
            self.dones: torch.Tensor | None = None
            """Episode termination flag (one env reset to the next)."""
            self.trial_dones: torch.Tensor | None = None
            """Trial termination flag. If None, it defaults to :attr:`dones` when the transition is stored.

            A trial is a sequence of one or more consecutive episodes over which memory (and any per-trial
            latent) persists. GAE bootstrapping and the lambda-trace are cut at trial boundaries only, so
            episode boundaries inside a trial are treated as non-terminal.
            """
            self.values: torch.Tensor | None = None
            self.actions_log_prob: torch.Tensor
            self.action_mean: torch.Tensor | None = None
            self.action_sigma: torch.Tensor | None = None
            self.hidden_states: tuple[HiddenState, HiddenState] = (None, None)
            self.raw_rewards: torch.Tensor | None = None
            """Rewards exactly as the acting path saw them, i.e. *before* the intrinsic-reward and
            timeout-bootstrap modifications applied to :attr:`rewards`.

            Trial-memory tokens carry ``r_{t-1}``, so the training-time reconstruction must use the very same
            numbers the acting path put into its tokens. Defaults to :attr:`rewards` when not set.
            """
            self.terminal_observations: TensorDict | None = None
            """Optional true terminal observation ``o_{T+1}`` of the environments that just terminated.

            In IsaacLab the observation returned by ``step()`` is already the *post-reset* observation, so it must
            not be used for the terminal token. When the environment exposes the real terminal observation, storing
            it here lets the reconstruction pass rebuild the terminal token exactly; otherwise the observation slot
            of the terminal token is zeros on both paths.
            """

        def clear(self) -> None:
            self.__init__()

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        device: str = "cpu",
        carry_steps: int = 0,
        max_policy_lag: int = 1,
    ) -> None:
        """
        Args:
            num_transitions_per_env: Steps **collected** per environment per rollout.
            carry_steps: Extra rows reserved *in front of* the rollout for a trial that is still open when the
                rollout ends. With ``carry_steps > 0`` the training-data unit (a trial) is decoupled from the
                collection unit (a rollout): an unfinished trial is carried across the rollout boundary instead
                of being thrown away, and is trained on once it completes. It must be at least as long as the
                longest trial (``K * T``) for nothing to be lost. ``0`` (the default) keeps the stock behavior.
            max_policy_lag: How many updates old a trial's *behavior* policy may be. A trial that spans a
                rollout boundary was partly collected under the previous policy, so it is off-policy by one
                update; anything older than this is dropped (and reported) rather than trained on with a stale
                PPO denominator.
        """
        self.training_type = training_type
        self.device = device
        # Rows written per rollout, versus total rows in the buffer (carry region + rollout region).
        # ``num_transitions_per_env`` keeps meaning "buffer rows" for every indexing helper below.
        self.num_collect_steps = num_transitions_per_env
        self.carry_steps = int(carry_steps)
        self.num_transitions_per_env = self.carry_steps + num_transitions_per_env
        num_transitions_per_env = self.num_transitions_per_env
        self.max_policy_lag = int(max_policy_lag)
        self.num_envs = num_envs
        self.actions_shape = actions_shape

        # Core
        self.observations = TensorDict(
            {key: torch.zeros(num_transitions_per_env, *value.shape, device=device) for key, value in obs.items()},
            batch_size=[num_transitions_per_env, num_envs],
            device=self.device,
        )
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # Rewards as fed to the policy's tokens (no intrinsic reward, no timeout bootstrap).
        self.raw_rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        # Trial boundaries. Defaults to the episode dones when the caller does not provide them,
        # which recovers the stock single-episode-per-trial behavior.
        self.trial_dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # For reinforcement learning
        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        # For RNN networks
        self.saved_hidden_state_a = None
        self.saved_hidden_state_c = None

        # For trial-memory policies (lazily allocated on the first transition that provides them)
        self.terminal_observations: TensorDict | None = None

        # -- carry-over of a trial that is still open when the rollout ends (see ``carry_steps``) --
        # Behavior-policy version of every stored step: the number of PPO updates that had happened when the
        # step was collected. A trial's lag is measured on its FIRST step, since that is where its memory
        # (and therefore every log-prob in it) starts from.
        self.policy_versions = torch.zeros(num_transitions_per_env, num_envs, 1, dtype=torch.long, device=device)
        self.policy_version = 0
        # Rows of the carry region that actually hold data, per environment. The carry region is
        # RIGHT-aligned, so environment ``e``'s window starts at row ``carry_steps - carry_len[e]``.
        self.carry_len = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        # Whether that first valid row is genuinely the first step of a trial. False only when an open trial
        # was longer than the carry region and its start had to be dropped.
        self.at_trial_start = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        # Cached episode/trial segmentation of the current rollout (invalidated whenever the data changes)
        self._pair_index: dict[str, torch.Tensor | int] | None = None

        # Counter for the number of transitions stored
        self.step = 0

    @property
    def row(self) -> int:
        """Buffer row the next transition will be written to (the carry region sits in front of it)."""
        return self.carry_steps + self.step

    @property
    def collected_observations(self) -> TensorDict:
        """The rows written by the most recent rollout, i.e. excluding the carried-over region.

        Used for the deferred observation-normalizer commit: the carry region has already been seen by the
        normalizer in the rollout that collected it, and counting it twice would skew the statistics.

        Note: still valid right after :meth:`clear`, which only writes rows *below* ``carry_steps``.
        """
        return self.observations[self.carry_steps :]

    def add_transitions(self, transition: Transition) -> None:
        # Check if the transition is valid
        if self.step >= self.num_collect_steps:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")
        step = self.row

        # Core
        self.observations[step].copy_(transition.observations)
        self.actions[step].copy_(transition.actions)
        self.rewards[step].copy_(transition.rewards.view(-1, 1))
        self.dones[step].copy_(transition.dones.view(-1, 1))
        self.policy_versions[step].fill_(self.policy_version)
        # Fall back to the episode dones so that callers unaware of trials keep the stock behavior
        trial_dones = transition.trial_dones if transition.trial_dones is not None else transition.dones
        self.trial_dones[step].copy_(trial_dones.view(-1, 1))
        # Token rewards. Fall back to the (possibly modified) rewards for callers that do not set them.
        raw_rewards = transition.raw_rewards if transition.raw_rewards is not None else transition.rewards
        self.raw_rewards[step].copy_(raw_rewards.view(-1, 1))
        # Optional true terminal observations (see Transition.terminal_observations)
        if transition.terminal_observations is not None:
            if self.terminal_observations is None:
                self.terminal_observations = TensorDict(
                    {
                        key: torch.zeros(self.num_transitions_per_env, *value.shape, device=self.device)
                        for key, value in transition.terminal_observations.items()
                    },
                    batch_size=[self.num_transitions_per_env, self.num_envs],
                    device=self.device,
                )
            self.terminal_observations[step].copy_(transition.terminal_observations)
        # The rollout changed, so any cached segmentation is stale
        self._pair_index = None

        # For distillation
        if self.training_type == "distillation":
            self.privileged_actions[step].copy_(transition.privileged_actions)

        # For reinforcement learning
        if self.training_type == "rl":
            self.values[step].copy_(transition.values)
            self.actions_log_prob[step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[step].copy_(transition.action_mean)
            self.sigma[step].copy_(transition.action_sigma)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # Increment the counter
        self.step += 1

    def _save_hidden_states(self, hidden_states: tuple[HiddenState, HiddenState]) -> None:
        if hidden_states == (None, None):
            return
        # Make a tuple out of GRU hidden states to match the LSTM format
        hidden_state_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hidden_state_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)
        # Initialize hidden states if needed
        if self.saved_hidden_state_a is None:
            self.saved_hidden_state_a = [
                torch.zeros(self.observations.shape[0], *hidden_state_a[i].shape, device=self.device)
                for i in range(len(hidden_state_a))
            ]
            self.saved_hidden_state_c = [
                torch.zeros(self.observations.shape[0], *hidden_state_c[i].shape, device=self.device)
                for i in range(len(hidden_state_c))
            ]
        # Copy the states
        for i in range(len(hidden_state_a)):
            self.saved_hidden_state_a[i][self.row].copy_(hidden_state_a[i])
            self.saved_hidden_state_c[i][self.row].copy_(hidden_state_c[i])

    def clear(self) -> None:
        """Free the buffer for the next rollout, carrying any still-open trial across the boundary.

        With ``carry_steps == 0`` this is the stock "reset the write pointer" behavior. Otherwise the steps of
        the trial that is still running -- everything after each environment's last trial done -- is moved to
        the carry region so that the next update can train on it once the trial completes. Nothing is ever
        discarded for the mere reason that it straddles a rollout boundary.
        """
        if self.step > 0:
            if self.carry_steps > 0:
                self._roll_carry_over()
            else:
                # Without a carry region there is nothing to hand over, so all that survives is the phase:
                # the next rollout begins on a trial boundary exactly when this one ended on one.
                self.at_trial_start = self.trial_dones[self.step - 1].reshape(-1).bool().clone()
        self.step = 0
        self._pair_index = None

    def _roll_carry_over(self) -> None:
        """Move each environment's open-trial tail into the (right-aligned) carry region."""
        num_steps, num_envs = self.num_transitions_per_env, self.num_envs
        end = self.carry_steps + self.step  # exclusive; == num_steps after a full rollout
        rows = torch.arange(num_steps, device=self.device).unsqueeze(1)
        valid_start = (self.carry_steps - self.carry_len).unsqueeze(0)  # [1, N]

        # Last step of the window that closed a trial; -1 when the whole window is one open trial.
        closed = self.trial_dones.reshape(num_steps, num_envs).bool() & (rows >= valid_start) & (rows < end)
        last_close = torch.where(closed, rows.expand_as(closed), torch.full_like(closed, -1, dtype=torch.long))
        last_close = last_close.max(dim=0).values
        # Steps since the trial started (the whole valid window when no trial ever closed in it)
        valid_len = end - (self.carry_steps - self.carry_len)
        open_len = torch.where(last_close >= 0, end - 1 - last_close, valid_len)

        # The tail sits in the last ``open_len`` rows; shifting the whole buffer down by the number of rows
        # this rollout wrote puts it exactly at rows [carry_steps - open_len, carry_steps). One uniform shift
        # for every environment, because both the source and the destination are anchored at the window END.
        self._shift_rows(self.step)
        # A trial longer than the carry region loses its first steps, so its memory can no longer be
        # reconstructed: mark it, and build_trial_pairs drops it (once) when it completes.
        self.at_trial_start = open_len <= self.carry_steps
        self.carry_len = open_len.clamp(max=self.carry_steps)

    def _shift_rows(self, shift: int) -> None:
        """Move rows ``[shift, shift + carry_steps)`` of every stored buffer down to ``[0, carry_steps)``."""
        carry = self.carry_steps

        def move(tensor: torch.Tensor) -> None:
            tensor[:carry].copy_(tensor[shift : shift + carry].clone())

        def move_td(td: TensorDict) -> None:
            for value in td.values():
                move(value) if isinstance(value, torch.Tensor) else move_td(value)

        move_td(self.observations)
        if self.terminal_observations is not None:
            move_td(self.terminal_observations)
        for name in (
            "rewards",
            "raw_rewards",
            "actions",
            "dones",
            "trial_dones",
            "policy_versions",
            "privileged_actions",
            "values",
            "actions_log_prob",
            "mu",
            "sigma",
            "returns",
        ):
            tensor = getattr(self, name, None)
            if isinstance(tensor, torch.Tensor):
                move(tensor)

    def compute_returns(
        self, last_values: torch.Tensor, gamma: float, lam: float, normalize_advantage: bool = True
    ) -> None:
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            # 1 if the trial continues into the next step, 0 otherwise.
            # Note: this gates both the value bootstrap and the lambda-trace, so it keys on the trial
            # boundary rather than the episode boundary. Episode boundaries inside a trial are
            # non-terminal: memory (and the latent) carry over, so credit must flow across them.
            next_is_not_terminal = 1.0 - self.trial_dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # Compute the advantages
        self.advantages = self.returns - self.values
        # Normalize the advantages if flag is set
        # Note: This is to prevent double normalization (i.e. if per minibatch normalization is used)
        if normalize_advantage:
            if self.carry_steps == 0:
                self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
            else:
                # Rows in front of an environment's window are stale (already-consumed or never-written) and
                # would otherwise drag the normalization statistics around.
                live = self.valid_mask().unsqueeze(-1)
                live_advantages = self.advantages[live]
                self.advantages = (self.advantages - live_advantages.mean()) / (live_advantages.std() + 1e-8)

    def valid_mask(self) -> torch.Tensor:
        """``[num_transitions_per_env, num_envs]`` mask of rows that hold live data for their environment.

        Everything from row ``carry_steps - carry_len[e]`` on: the carried-over open trial followed by the
        rollout just collected. Always all-True without a carry region.
        """
        rows = torch.arange(self.num_transitions_per_env, device=self.device).unsqueeze(1)
        return rows >= (self.carry_steps - self.carry_len).unsqueeze(0)

    # For distillation
    def generator(self) -> Generator:
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            yield self.observations[i], self.actions[i], self.privileged_actions[i], self.dones[i]

    # For reinforcement learning with feedforward networks
    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        if self.carry_steps > 0:
            raise ValueError(
                "The flat mini-batch generator cannot be used with a carry region: it would train on the"
                " carried rows (an unfinished trial) and on stale rows in front of them. Use"
                " trial_pair_mini_batch_generator, or construct the storage with carry_steps=0."
            )
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)

        # For PPO
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                # Create the mini-batch
                obs_batch = observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                hidden_state_a_batch = None
                hidden_state_c_batch = None
                masks_batch = None

                # Yield the mini-batch
                yield (
                    obs_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (
                        hidden_state_a_batch,
                        hidden_state_c_batch,
                    ),
                    masks_batch,
                )

    # For reinforcement learning with recurrent networks
    def recurrent_mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # Reshape to [num_envs, time, num layers, hidden dim]
                # Original shape: [time, num_layers, num_envs, hidden_dim])
                last_was_done = last_was_done.permute(1, 0)
                # Take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                hidden_state_a_batch = [
                    saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_state in self.saved_hidden_state_a
                ]
                hidden_state_c_batch = [
                    saved_hidden_state.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_state in self.saved_hidden_state_c
                ]
                # Remove the tuple for GRU
                hidden_state_a_batch = (
                    hidden_state_a_batch[0] if len(hidden_state_a_batch) == 1 else hidden_state_a_batch
                )
                hidden_state_c_batch = (
                    hidden_state_c_batch[0] if len(hidden_state_c_batch) == 1 else hidden_state_c_batch
                )

                # Yield the mini-batch
                yield (
                    obs_batch,
                    actions_batch,
                    values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    (
                        hidden_state_a_batch,
                        hidden_state_c_batch,
                    ),
                    masks_batch,
                )

                first_traj = last_traj

    # ----------------------------------------------------------------------------------------------------------
    # Trial-memory (hierarchical) training: episode segmentation, memory sweep and pair minibatches
    # ----------------------------------------------------------------------------------------------------------

    def build_trial_pairs(self, verbose: bool = True) -> dict[str, torch.Tensor | int]:
        """Segment the rollout into episodes, group them into trials and form adjacent episode **pairs**.

        The training unit of the trial-memory policy is a pair ``(tau_e, tau_{e+1})``: the source episode
        ``tau_e`` supplies a differentiable memory context and the target episode ``tau_{e+1}`` takes the PPO
        loss. For a trial of ``K`` episodes the pairs are ``(None, tau_1), (tau_1, tau_2), ... (tau_{K-1},
        tau_K)`` -- so every episode is a target exactly once and receives exactly one PPO loss.

        Episode boundaries are **derived from the stored dones**, never assumed to sit at fixed indices: a
        genuine terminal (e.g. ``abnormal_robot``) resets an environment early and desynchronizes it.

        A trial that is still open at the end of the window is **not** dropped: it is *deferred*, i.e. left in
        the buffer for :meth:`clear` to carry into the next rollout, and trained on once it completes. The only
        losses are trials older than :attr:`max_policy_lag` updates and trials longer than the carry region;
        both are counted and reported separately, because they are real data loss and must never be silent.

        Returns (and caches) a dictionary of 1-D tensors indexed by *valid* episode:

        * ``ep_start`` / ``ep_len`` / ``ep_env``: buffer row of the first transition, number of acting steps,
          environment index,
        * ``ep_pos``: index of the episode inside its trial (0 == first episode, i.e. the degenerate pair),
        * ``ep_lag``: how many updates old the episode's trial is (0 == fully on-policy),
        * ``num_pairs``: number of valid episodes == number of pairs,
        * ``max_len``: longest valid episode, ``deferred_*`` / ``dropped_*``: accounting.
        """
        if self._pair_index is not None:
            return self._pair_index

        seg = self._segment_rollout()
        num_envs, device = self.num_envs, self.device
        ep_start, ep_len, ep_env, ep_pos = seg["ep_start"], seg["ep_len"], seg["ep_env"], seg["ep_pos"]
        ep_trial = seg["ep_trial"]
        trial_env, trial_lag = seg["trial_env"], seg["trial_lag"]
        trial_closed, trial_all_closed = seg["trial_closed"], seg["trial_all_closed"]
        trial_first_row, trial_last_row = seg["trial_first_row"], seg["trial_last_row"]
        trial_window_start = seg["trial_window_start"]

        # Entirely in front of the window: consumed by an earlier update (or never written). Not data loss.
        trial_stale = trial_last_row < trial_window_start
        # The window's first row is a genuine trial start unless a carried trial was longer than the carry
        # region, in which case its first steps -- and with them its memory -- are gone.
        trial_started = (trial_first_row > trial_window_start) | self.at_trial_start.to(device)[trial_env]
        live = ~trial_stale
        # Still running: keep it in the buffer, train on it when it completes. NOT dropped.
        trial_deferred = live & trial_started & ~trial_closed
        trial_truncated = live & ~trial_started
        trial_lagged = live & trial_started & trial_closed & trial_all_closed & (trial_lag > self.max_policy_lag)
        valid_trial = live & trial_started & trial_closed & trial_all_closed & (trial_lag <= self.max_policy_lag)

        keep = valid_trial[ep_trial]
        selected = keep.nonzero(as_tuple=False).squeeze(-1)
        ep_start_c = ep_start[selected]
        ep_len_c = ep_len[selected]
        ep_env_c = ep_env[selected]
        ep_pos_c = ep_pos[selected]
        ep_lag_c = trial_lag[ep_trial][selected]

        # Filtering removes whole trials, so a kept episode's predecessor inside its trial stays at index - 1.
        if selected.numel() > 0:
            successor = ep_pos_c > 0
            if bool(successor.any()):
                prev = successor.nonzero(as_tuple=False).squeeze(-1) - 1
                assert bool((ep_env_c[prev] == ep_env_c[successor]).all()), "pair source/target env mismatch"
                assert bool((ep_pos_c[prev] + 1 == ep_pos_c[successor]).all()), "pair source/target are not adjacent"

        # -- accounting. "Dropped" means lost for good; a deferred trial is neither kept nor lost yet. --
        def episodes_in(trial_mask: torch.Tensor) -> int:
            return int(trial_mask[ep_trial].sum().item())

        dropped = trial_truncated | trial_lagged
        dropped_eps = episodes_in(dropped)
        dropped_trials = int(dropped.sum().item())
        dropped_envs = int(torch.unique(trial_env[dropped]).numel()) if dropped_trials else 0
        deferred_trials = int(trial_deferred.sum().item())
        deferred_eps = episodes_in(trial_deferred)
        # An environment that contributes nothing to this update is the signature of the failure mode this
        # buffer exists to prevent (a permanently phase-shifted environment silently going dark).
        envs_without_data = num_envs - int(torch.unique(ep_env_c).numel())
        lag0_pairs = int((ep_lag_c == 0).sum().item())
        if verbose:
            print(
                f"[RolloutStorage] trial pool: {selected.numel()} episodes in"
                f" {int(valid_trial.sum().item())} trials ({lag0_pairs} pairs at lag 0);"
                f" deferred {deferred_eps} episodes in {deferred_trials} open trials;"
                f" dropped {dropped_eps} episodes in {dropped_trials} trials"
                f" ({int(trial_lagged.sum().item())} too lagged, {int(trial_truncated.sum().item())} whose"
                f" start no longer fits the buffer) across {dropped_envs} envs;"
                f" {envs_without_data}/{num_envs} envs contributed nothing."
            )

        self._pair_index = {
            "ep_start": ep_start_c,
            "ep_len": ep_len_c,
            "ep_env": ep_env_c,
            "ep_pos": ep_pos_c,
            "ep_lag": ep_lag_c,
            "num_pairs": int(selected.numel()),
            "num_trials": int(valid_trial.sum().item()),
            "max_len": int(ep_len_c.max().item()) if selected.numel() > 0 else 0,
            "dropped_episodes": dropped_eps,
            "dropped_trials": dropped_trials,
            "dropped_envs": dropped_envs,
            "dropped_lagged_trials": int(trial_lagged.sum().item()),
            "dropped_truncated_trials": int(trial_truncated.sum().item()),
            "deferred_episodes": deferred_eps,
            "deferred_trials": deferred_trials,
            "envs_without_data": envs_without_data,
            "lag0_pairs": lag0_pairs,
        }
        return self._pair_index

    def advantage_outcome_audit(self) -> dict[str, float]:
        """Does the advantage a row carries agree with how that row's episode actually ended?

        Over every row of every *trainable* (i.e. completed, see :meth:`build_trial_pairs`) episode in the
        buffer:

        * ``adv/return_corr`` -- Pearson correlation between the per-row advantage and the total **raw** return
          of the episode the row belongs to. A trial-level credit assignment that has degenerated into "the
          whole trial was good/bad" shows up here as a correlation near 1; a value function that has stopped
          tracking outcomes shows up as one near 0,
        * ``adv/frac_pos_in_top_quartile_episodes`` / ``adv/frac_pos_in_bottom_quartile_episodes`` -- the
          fraction of rows with a positive advantage inside the best / worst quarter of episodes (ranked by
          raw return within this batch). Healthy PPO reinforces most of the good episodes and few of the bad
          ones; the two numbers collapsing onto each other means the advantages no longer separate outcomes.

        Returns an empty dict when there is nothing to audit.
        """
        index = self.build_trial_pairs()
        num_episodes = index["num_pairs"]
        if num_episodes == 0 or self.training_type != "rl":
            return {}
        device, num_steps, num_envs = self.device, self.num_transitions_per_env, self.num_envs
        starts, lengths, envs = index["ep_start"], index["ep_len"], index["ep_env"]

        offsets = torch.arange(int(lengths.max().item()), device=device).unsqueeze(1)  # [S, 1]
        mask = offsets < lengths.unsqueeze(0)  # [S, B]
        flat = (starts.unsqueeze(0) + offsets).clamp(max=num_steps - 1) * num_envs + envs.unsqueeze(0)
        advantages = self.advantages.reshape(-1)[flat]  # [S, B]
        rewards = self.raw_rewards.reshape(-1)[flat]
        episode_return = (rewards * mask.to(rewards.dtype)).sum(dim=0)  # [B]

        row_advantage = advantages[mask]
        row_return = episode_return.unsqueeze(0).expand_as(advantages)[mask]
        centered_advantage = row_advantage - row_advantage.mean()
        centered_return = row_return - row_return.mean()
        denominator = centered_advantage.norm() * centered_return.norm()
        correlation = 0.0 if float(denominator) <= 0.0 else float((centered_advantage @ centered_return) / denominator)

        # Episodes ranked by outcome; the quartiles are at least one episode wide.
        quartile = max(1, num_episodes // 4)
        order = torch.argsort(episode_return)
        audit = {"adv/return_corr": correlation}
        for label, episode_ids in (
            ("adv/frac_pos_in_top_quartile_episodes", order[-quartile:]),
            ("adv/frac_pos_in_bottom_quartile_episodes", order[:quartile]),
        ):
            selected = advantages[:, episode_ids][mask[:, episode_ids]]
            audit[label] = float((selected > 0).to(torch.float32).mean()) if selected.numel() > 0 else 0.0
        return audit

    def _segment_rollout(self) -> dict[str, torch.Tensor | int]:
        """Split the buffer into episodes and trials, with **no** filtering (stale rows included).

        The raw segmentation shared by :meth:`build_trial_pairs` (which then keeps only trainable trials) and
        :meth:`live_episode_index` (which keeps every trial that still holds live data, including the open one).
        Everything is derived from the stored dones -- boundaries are never assumed at fixed indices.
        """
        num_steps, num_envs, device = self.num_transitions_per_env, self.num_envs, self.device
        # A trial end always closes the episode it ends, even if the caller only flagged the trial.
        dones = self.dones.reshape(num_steps, num_envs).bool() | self.trial_dones.reshape(num_steps, num_envs).bool()
        trial_dones = self.trial_dones.reshape(num_steps, num_envs).bool()

        # Rows in front of an environment's window hold stale (already consumed) data. Force a segment break
        # just before the window so that no episode/trial spans the two -- otherwise the first live episode
        # would inherit the stale rows' start index and look like it began outside the window.
        rows = torch.arange(num_steps, device=device).unsqueeze(1)
        window_start = (self.carry_steps - self.carry_len.to(device)).unsqueeze(0)  # [1, N]
        boundary = rows == (window_start - 1)

        # Per-step segment ids: the index (within the environment) of the episode/trial the step belongs to.
        # Exclusive cumulative sum, because a done flags the *last* step of its segment.
        ep_breaks = (dones | boundary).long()
        trial_breaks = (trial_dones | boundary).long()
        ep_id = torch.cumsum(ep_breaks, dim=0) - ep_breaks
        trial_id = torch.cumsum(trial_breaks, dim=0) - trial_breaks

        eps_per_env = ep_id[-1] + 1
        trials_per_env = trial_id[-1] + 1
        ep_offset = torch.cumsum(eps_per_env, dim=0) - eps_per_env
        trial_offset = torch.cumsum(trials_per_env, dim=0) - trials_per_env
        num_eps = int(eps_per_env.sum().item())
        num_trials = int(trials_per_env.sum().item())

        # Flatten (env, segment) into a single global index, ordered by (env, time).
        global_ep = ep_id + ep_offset.unsqueeze(0)
        global_trial = trial_id + trial_offset.unsqueeze(0)
        flat_ep = global_ep.reshape(-1)

        step_idx = torch.arange(num_steps, device=device).unsqueeze(1).expand(num_steps, num_envs).reshape(-1)
        env_idx = torch.arange(num_envs, device=device).unsqueeze(0).expand(num_steps, num_envs).reshape(-1)

        ep_len = torch.bincount(flat_ep, minlength=num_eps)
        ep_start = torch.full((num_eps,), num_steps, dtype=torch.long, device=device).scatter_reduce(
            0, flat_ep, step_idx, reduce="amin"
        )
        ep_env = torch.zeros(num_eps, dtype=torch.long, device=device).scatter_reduce(
            0, flat_ep, env_idx, reduce="amax"
        )
        ep_last_step = ep_start + ep_len - 1
        ep_trial = global_trial[ep_last_step, ep_env]
        ep_closed = dones[ep_last_step, ep_env]
        ep_ends_trial = trial_dones[ep_last_step, ep_env]

        # Position of an episode inside its trial (episodes are ordered by (env, time), so trials are contiguous)
        ep_index = torch.arange(num_eps, device=device)
        first_ep_of_trial = torch.full((num_trials,), num_eps, dtype=torch.long, device=device).scatter_reduce(
            0, ep_trial, ep_index, reduce="amin"
        )
        ep_pos = ep_index - first_ep_of_trial[ep_trial]

        # -- classify every trial: consumed / deferred / dropped / trainable --
        trial_env = torch.zeros(num_trials, dtype=torch.long, device=device).scatter_reduce(
            0, ep_trial, ep_env, reduce="amax"
        )
        trial_closed = (
            torch.zeros(num_trials, dtype=torch.long, device=device)
            .scatter_reduce(0, ep_trial, ep_ends_trial.long(), reduce="amax")
            .bool()
        )
        trial_all_closed = (
            torch.ones(num_trials, dtype=torch.long, device=device)
            .scatter_reduce(0, ep_trial, ep_closed.long(), reduce="amin")
            .bool()
        )
        trial_first_row = torch.full((num_trials,), num_steps, dtype=torch.long, device=device).scatter_reduce(
            0, ep_trial, ep_start, reduce="amin"
        )
        trial_last_row = torch.zeros(num_trials, dtype=torch.long, device=device).scatter_reduce(
            0, ep_trial, ep_last_step, reduce="amax"
        )
        trial_window_start = window_start.reshape(-1)[trial_env]

        # How many updates old the trial's behavior policy is, read off its FIRST step: that is where its
        # memory starts, so every log-prob in it was produced by that policy version.
        versions = self.policy_versions.reshape(num_steps, num_envs)
        trial_lag = self.policy_version - versions[trial_first_row, trial_env]

        return {
            "ep_start": ep_start,
            "ep_len": ep_len,
            "ep_env": ep_env,
            "ep_pos": ep_pos,
            "ep_trial": ep_trial,
            "ep_last_step": ep_last_step,
            "ep_closed": ep_closed,
            "ep_ends_trial": ep_ends_trial,
            "num_eps": num_eps,
            "num_trials": num_trials,
            "trial_env": trial_env,
            "trial_closed": trial_closed,
            "trial_all_closed": trial_all_closed,
            "trial_first_row": trial_first_row,
            "trial_last_row": trial_last_row,
            "trial_window_start": trial_window_start,
            "trial_lag": trial_lag,
            "window_start": window_start.reshape(-1),
        }

    def live_episode_index(self) -> dict[str, torch.Tensor | int]:
        """Every episode that still holds live data, **including** the ones of open (deferred) trials.

        Same layout as :meth:`build_trial_pairs` (``ep_start`` / ``ep_len`` / ``ep_env`` / ``ep_pos`` plus
        ``num_pairs`` / ``max_len``, so that :meth:`episode_batch` can consume it), but the only filter is
        "the trial has at least one row inside the environment's window". This is what the critic value sweep
        needs: GAE runs over the whole buffer and the advantage normalization reduces over every live row, so
        the still-open trial's rows need a value too -- they simply are not trained on yet.

        A trial whose start was truncated away (``at_trial_start`` False for a carried trial longer than the
        carry region) is kept here and its first live episode is treated as position 0, i.e. its memory
        restarts from ``Z_init``. That is an approximation, but such a trial is dropped by
        :meth:`build_trial_pairs` when it completes, so it never reaches a PPO loss.
        """
        seg = self._segment_rollout()
        ep_trial = seg["ep_trial"]
        live_trial = seg["trial_last_row"] >= seg["trial_window_start"]
        selected = live_trial[ep_trial].nonzero(as_tuple=False).squeeze(-1)
        ep_len = seg["ep_len"][selected]
        return {
            "ep_start": seg["ep_start"][selected],
            "ep_len": ep_len,
            "ep_env": seg["ep_env"][selected],
            "ep_pos": seg["ep_pos"][selected],
            "ep_closed": seg["ep_closed"][selected],
            "num_pairs": int(selected.numel()),
            "max_len": int(ep_len.max().item()) if selected.numel() > 0 else 0,
        }

    def episode_batch(
        self,
        episode_ids: torch.Tensor,
        include_terminal_token: bool,
        include_targets: bool,
        index: dict[str, torch.Tensor | int] | None = None,
    ) -> dict[str, torch.Tensor | TensorDict]:
        """Gather a padded ``[S, B, ...]`` batch of episodes from the flat rollout buffers.

        Args:
            episode_ids: Indices into the arrays of ``index``.
            include_terminal_token: Whether to append the terminal token ``x_{T+1} = Embed(o_{T+1}, a_T, r_T,
                d_T)``. Sources need it (the writer must see the episode outcome); targets do not, since it emits
                no action and takes no loss.
            include_targets: Whether to also gather the PPO targets (actions, old log-probs, advantages, ...).
            index: Episode index to resolve ``episode_ids`` against. Defaults to the trainable pool of
                :meth:`build_trial_pairs`; the critic value sweep passes :meth:`live_episode_index` instead.

        Returns:
            A dictionary with the token inputs (``obs``, ``prev_actions``, ``prev_rewards``, ``prev_dones``), the
            validity ``mask`` ``[S, B]`` and -- with ``include_targets`` -- the per-step PPO targets plus
            ``loss_mask`` ``[S, B]``, which is ``True`` only on the acting steps.
        """
        index = self.build_trial_pairs() if index is None else index
        num_steps, num_envs, device = self.num_transitions_per_env, self.num_envs, self.device
        starts = index["ep_start"][episode_ids]
        lengths = index["ep_len"][episode_ids]
        envs = index["ep_env"][episode_ids]
        num_tokens = int(lengths.max().item()) + (1 if include_terminal_token else 0)

        offsets = torch.arange(num_tokens, device=device).unsqueeze(1)  # [S, 1]
        action_mask = offsets < lengths.unsqueeze(0)  # [S, B] acting steps
        is_terminal = offsets == lengths.unsqueeze(0)  # [S, B] the extra terminal row
        token_mask = action_mask | is_terminal if include_terminal_token else action_mask

        # Clamp the *unclamped* row index separately for the two gathers: an episode that ends on the very last
        # buffer row has its terminal token at row ``num_steps``, and reading ``previous`` off the clamped
        # ``current`` would then take row ``num_steps - 2`` instead of the episode's actual last step.
        raw_rows = starts.unsqueeze(0) + offsets
        current = raw_rows.clamp(max=num_steps - 1)
        previous = (raw_rows - 1).clamp(min=0, max=num_steps - 1)
        current_flat = current * num_envs + envs.unsqueeze(0)
        previous_flat = previous * num_envs + envs.unsqueeze(0)
        # The token at t carries (o_t, a_{t-1}, r_{t-1}, d_{t-1}); the first token of an episode carries zeros,
        # exactly like the acting path after reset_episode().
        has_previous = token_mask & (offsets > 0)

        observations = self.observations.reshape(-1)[current_flat]
        if include_terminal_token:
            if self.terminal_observations is not None:
                # Real terminal observation, recorded on the episode's last acting step
                last_flat = (starts + lengths - 1).clamp(min=0) * num_envs + envs
                terminal_obs = self.terminal_observations.reshape(-1)[last_flat.unsqueeze(0).expand(num_tokens, -1)]
                observations = observations.apply(
                    lambda value, terminal: torch.where(
                        _broadcast_mask(is_terminal, value.dim()), terminal, value
                    ),
                    terminal_obs,
                )
            else:
                # No true terminal observation available: the observation slot is zeros on both paths. The
                # load-bearing content of the terminal token is r_T and d_T.
                observations = observations.apply(lambda value: _mask_leading(value, ~is_terminal))
        observations = observations.apply(lambda value: _mask_leading(value, token_mask))

        prev_actions = _mask_leading(
            self.actions.reshape(-1, *self.actions_shape)[previous_flat], has_previous
        )
        prev_rewards = _mask_leading(self.raw_rewards.reshape(-1, 1)[previous_flat], has_previous)
        prev_dones = _mask_leading(self.dones.reshape(-1, 1)[previous_flat].float(), has_previous)

        batch = {
            "obs": observations,
            "prev_actions": prev_actions,
            "prev_rewards": prev_rewards,
            "prev_dones": prev_dones,
            "mask": token_mask,
        }
        if include_targets:
            batch.update({
                "actions": _mask_leading(self.actions.reshape(-1, *self.actions_shape)[current_flat], action_mask),
                "old_actions_log_prob": _mask_leading(self.actions_log_prob.reshape(-1, 1)[current_flat], action_mask),
                "old_mu": _mask_leading(self.mu.reshape(-1, *self.actions_shape)[current_flat], action_mask),
                "old_sigma": _mask_leading(self.sigma.reshape(-1, *self.actions_shape)[current_flat], action_mask),
                "advantages": _mask_leading(self.advantages.reshape(-1, 1)[current_flat], action_mask),
                "returns": _mask_leading(self.returns.reshape(-1, 1)[current_flat], action_mask),
                "values": _mask_leading(self.values.reshape(-1, 1)[current_flat], action_mask),
                "loss_mask": action_mask,
                # Per-ROW policy lag ``[S, B]``: how many updates ago *this step* was collected. NOT the
                # trial's lag (which is read off the trial's first step): a trial that spans several rollouts
                # closes -- and is trained on -- many updates after it started, so its last rows can be fully
                # on-policy while its first rows are many updates old. Rows outside ``loss_mask`` are
                # meaningless, exactly like every other entry here.
                "row_lags": self.policy_version - self.policy_versions.reshape(-1)[current_flat],
            })
        return batch

    def compute_memory_checkpoints(
        self, policy, chunk_size: int | None = None, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        """One ``no_grad`` sweep over every trial computing the incoming memory checkpoint of every episode.

        ``Zbar_1 = Z_init`` and ``Zbar_{e+1} = G(Zbar_e, H_e)``, all from the raw stored data and with the
        *current* parameters. Once this is done the pairs are mutually independent, so they can be shuffled and
        minibatched freely; each pair then only needs its own (detached) ``Zbar_e``.

        Args:
            policy: The trial-memory policy (needs ``forward_sequence`` / ``write_memory`` / ``z_init``).
            chunk_size: Maximum number of episodes forwarded at once. ``None`` means all of them.
            dtype: Storage dtype of the cache, e.g. ``torch.float16`` to halve its footprint. Defaults to the
                policy's parameter dtype.

        Returns:
            ``Zbar`` ``[num_pairs, M, d]``, detached, indexed like :meth:`build_trial_pairs`.
        """
        return self._memory_checkpoints(self.build_trial_pairs(), policy, chunk_size, dtype)

    def _memory_checkpoints(
        self,
        index: dict[str, torch.Tensor | int],
        policy,
        chunk_size: int | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """:meth:`compute_memory_checkpoints` over an arbitrary episode ``index`` (pool or live set)."""
        num_pairs = index["num_pairs"]
        param_dtype = policy.z_init.dtype
        cache_dtype = param_dtype if dtype is None else dtype
        memory = torch.zeros(
            num_pairs, policy.num_memory_tokens, policy.d_model, device=self.device, dtype=cache_dtype
        )
        if num_pairs == 0:
            return memory

        max_tokens = getattr(policy, "max_tokens", None)
        if max_tokens is not None and index["max_len"] + 1 > max_tokens:
            raise ValueError(
                f"An episode of {index['max_len']} steps does not fit the policy's {max_tokens} token slots"
                " (T + 1). Check max_episode_length."
            )

        positions = index["ep_pos"]
        memory[positions == 0] = policy.z_init.detach().to(device=self.device, dtype=cache_dtype)
        max_position = int(positions.max().item())
        chunk = num_pairs if chunk_size is None else int(chunk_size)

        with torch.no_grad():
            for position in range(max_position):
                sources = (positions == position).nonzero(as_tuple=False).squeeze(-1)
                targets = sources + 1
                inside = targets < num_pairs
                sources, targets = sources[inside], targets[inside]
                adjacent = positions[targets] == position + 1
                sources, targets = sources[adjacent], targets[adjacent]
                for start in range(0, sources.numel(), chunk):
                    source_ids = sources[start : start + chunk]
                    target_ids = targets[start : start + chunk]
                    batch = self.episode_batch(
                        source_ids, include_terminal_token=True, include_targets=False, index=index
                    )
                    incoming = memory[source_ids].to(param_dtype)
                    hidden = policy.forward_sequence(
                        batch["obs"],
                        batch["prev_actions"],
                        batch["prev_rewards"],
                        batch["prev_dones"],
                        memory=incoming,
                        mask=batch["mask"],
                    )
                    outgoing, _ = policy.write_memory(
                        incoming, hidden.transpose(0, 1), mask=batch["mask"].transpose(0, 1)
                    )
                    memory[target_ids] = outgoing.to(cache_dtype)
        return memory

    def compute_critic_values(self, policy, chunk_size: int | None = None, last_obs=None) -> torch.Tensor:
        """Fill :attr:`values` for every live row with one batched sweep of the **separate** critic trunk.

        Used instead of a per-step ``policy.evaluate()`` during collection: with ``critic_trunk="separate"`` the
        value function has no incremental path, so the values are produced here, after the rollout, in the same
        chunked/``no_grad`` shape as :meth:`compute_memory_checkpoints`:

        1. segment every live episode (:meth:`live_episode_index`) -- including the still-open trial's, whose
           rows GAE and the advantage normalization both touch,
        2. roll the writer through each trial to get every episode's incoming memory ``Zbar`` (actor-side, the
           memory the episode was acted with; it enters the critic detached),
        3. one :meth:`~rsl_rl.modules.ActorCriticTrialMemory.forward_sequence_critic` per chunk, scattered back
           into :attr:`values`.

        Args:
            policy: The trial-memory policy, built with ``critic_trunk="separate"``.
            chunk_size: Maximum number of episodes forwarded at once. ``None`` means all of them.
            last_obs: The observation returned by the final ``env.step()`` of the rollout. When given, the GAE
                bootstrap value is computed exactly, as one extra token appended to each environment's open
                episode; without it the bootstrap is zero.

        Returns:
            ``last_values`` ``[num_envs, 1]``, the GAE bootstrap for :meth:`compute_returns`.
        """
        device, num_steps, num_envs = self.device, self.num_transitions_per_env, self.num_envs
        last_values = torch.zeros(num_envs, 1, device=device, dtype=self.values.dtype)
        index = self.live_episode_index()
        num_episodes = index["num_pairs"]
        if num_episodes == 0:
            return last_values

        max_tokens = getattr(policy, "max_tokens", None)
        if max_tokens is not None and index["max_len"] + 1 > max_tokens:
            raise ValueError(
                f"An episode of {index['max_len']} steps does not fit the policy's {max_tokens} token slots"
                " (T + 1). Check max_episode_length."
            )

        param_dtype = policy.z_init.dtype
        memory = self._memory_checkpoints(index, policy, chunk_size)
        starts, envs = index["ep_start"], index["ep_env"]
        chunk = num_episodes if chunk_size is None else int(chunk_size)

        with torch.no_grad():
            for begin in range(0, num_episodes, chunk):
                episode_ids = torch.arange(begin, min(begin + chunk, num_episodes), device=device)
                batch = self.episode_batch(
                    episode_ids, include_terminal_token=False, include_targets=False, index=index
                )
                values = policy.forward_sequence_critic(
                    batch["obs"],
                    batch["prev_actions"],
                    batch["prev_rewards"],
                    batch["prev_dones"],
                    memory=memory[episode_ids].to(param_dtype),
                    mask=batch["mask"],
                )
                # Scatter [S, B, 1] back onto the (row, env) grid the buffer is indexed by.
                mask = batch["mask"]
                offsets = torch.arange(mask.shape[0], device=device).unsqueeze(1)
                rows = starts[episode_ids].unsqueeze(0) + offsets
                flat = rows.clamp(max=num_steps - 1) * num_envs + envs[episode_ids].unsqueeze(0)
                self.values.reshape(-1, 1)[flat[mask]] = values[mask].to(self.values.dtype)

            if last_obs is not None:
                last_values = self._bootstrap_values(policy, index, memory, last_obs)
        return last_values

    def _bootstrap_values(
        self, policy, index: dict[str, torch.Tensor | int], memory: torch.Tensor, last_obs
    ) -> torch.Tensor:
        """``V(s_{end})`` per environment, for the GAE bootstrap at the rollout boundary.

        The pending observation is one extra token on top of the environment's last stored episode:

        * the episode is still running (``done`` False on the last stored row) -- the token continues it, so the
          whole episode prefix is re-run with the extra row appended and its value is read off that row,
        * the episode ended on the last stored row -- the token is the FIRST token of the next episode
          (``prev_action``/``prev_reward``/``prev_done`` zero, episode-start marker set, position 0), with the
          memory the acting path would have moved on to: ``Z_init`` if the trial also ended, else
          ``G(Zbar_e, H_e)``. (When the trial ended the bootstrap is multiplied by zero in
          :meth:`compute_returns` anyway; it is computed for uniformity.)
        """
        device, num_steps, num_envs = self.device, self.num_transitions_per_env, self.num_envs
        param_dtype = policy.z_init.dtype
        last_values = torch.zeros(num_envs, 1, device=device, dtype=self.values.dtype)
        end = self.carry_steps + self.step  # exclusive
        if end == 0:
            return last_values
        starts, lengths, envs = index["ep_start"], index["ep_len"], index["ep_env"]

        # The last live episode of every environment (episodes are ordered by (env, time)).
        episode_ids = torch.arange(index["num_pairs"], device=device)
        last_episode = torch.full((num_envs,), -1, dtype=torch.long, device=device).scatter_reduce(
            0, envs, episode_ids, reduce="amax", include_self=True
        )
        has_episode = last_episode >= 0
        # Only environments whose last episode actually ends at the rollout boundary can be bootstrapped.
        at_boundary = torch.zeros(num_envs, dtype=torch.bool, device=device)
        valid = has_episode.nonzero(as_tuple=False).squeeze(-1)
        ends = starts[last_episode[valid]] + lengths[last_episode[valid]]
        at_boundary[valid] = ends == end
        env_ids = at_boundary.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return last_values

        obs = policy.get_actor_obs(last_obs)
        episode_of_env = last_episode[env_ids]
        just_done = self.dones.reshape(num_steps, num_envs)[end - 1, env_ids].bool()

        # -- environments whose episode is still running: prefix + one appended token --
        running = (~just_done).nonzero(as_tuple=False).squeeze(-1)
        if running.numel() > 0:
            ids = episode_of_env[running]
            running_envs = env_ids[running]
            batch = self.episode_batch(ids, include_terminal_token=False, include_targets=False, index=index)
            lengths_r = lengths[ids]
            num_tokens = int(lengths_r.max().item()) + 1
            pad = num_tokens - batch["mask"].shape[0]
            offsets = torch.arange(num_tokens, device=device).unsqueeze(1)
            is_pending = offsets == lengths_r.unsqueeze(0)  # [S, B]
            token_mask = (offsets < lengths_r.unsqueeze(0)) | is_pending

            # The pending row carries (o_end, a_{end-1}, r_{end-1}, d_{end-1}) -- exactly what the acting path's
            # non-committing bootstrap token carries. Everything is worked in the concatenated observation
            # tensor, so the pending observation and the stored prefix are gathered the same way.
            def _append(value: torch.Tensor, row: torch.Tensor) -> torch.Tensor:
                if pad > 0:
                    filler = torch.zeros_like(value[:1]).repeat(pad, *([1] * (value.dim() - 1)))
                    value = torch.cat([value, filler])
                return torch.where(_broadcast_mask(is_pending, value.dim()), row.unsqueeze(0), value)

            flat_last = (end - 1) * num_envs + running_envs
            token_obs = _append(policy.get_actor_obs(batch["obs"]), obs[running_envs])
            prev_actions = _append(batch["prev_actions"], self.actions.reshape(-1, *self.actions_shape)[flat_last])
            prev_rewards = _append(batch["prev_rewards"], self.raw_rewards.reshape(-1, 1)[flat_last])
            prev_dones = _append(batch["prev_dones"], self.dones.reshape(-1, 1)[flat_last].float())
            values = policy.forward_sequence_critic(
                token_obs,
                prev_actions,
                prev_rewards,
                prev_dones,
                memory=memory[ids].to(param_dtype),
                mask=token_mask,
            )
            last_values[running_envs] = values[is_pending].to(self.values.dtype)

        # -- environments that just reset: the pending token starts a fresh episode --
        reset = just_done.nonzero(as_tuple=False).squeeze(-1)
        if reset.numel() > 0:
            ids = episode_of_env[reset]
            reset_envs = env_ids[reset]
            trial_over = self.trial_dones.reshape(num_steps, num_envs)[end - 1, reset_envs].bool()
            batch = self.episode_batch(ids, include_terminal_token=True, include_targets=False, index=index)
            hidden = policy.forward_sequence(
                batch["obs"],
                batch["prev_actions"],
                batch["prev_rewards"],
                batch["prev_dones"],
                memory=memory[ids].to(param_dtype),
                mask=batch["mask"],
            )
            written, _ = policy.write_memory(
                memory[ids].to(param_dtype), hidden.transpose(0, 1), mask=batch["mask"].transpose(0, 1)
            )
            initial = policy.initial_memory(int(reset.numel()), device=device).to(param_dtype)
            next_memory = torch.where(trial_over.view(-1, 1, 1), initial, written)
            num_reset = int(reset.numel())
            zeros_action = torch.zeros(1, num_reset, *self.actions_shape, device=device, dtype=self.actions.dtype)
            zeros_scalar = torch.zeros(1, num_reset, 1, device=device, dtype=self.rewards.dtype)
            values = policy.forward_sequence_critic(
                obs[reset_envs].unsqueeze(0),
                zeros_action,
                zeros_scalar,
                zeros_scalar,
                memory=next_memory,
                mask=torch.ones(1, num_reset, dtype=torch.bool, device=device),
            )
            last_values[reset_envs] = values[0].to(self.values.dtype)
        return last_values

    def trial_pair_mini_batch_generator(
        self,
        policy,
        num_mini_batches: int,
        num_epochs: int = 8,
        sweep_chunk_size: int | None = None,
        memory_dtype: torch.dtype | None = None,
    ) -> Generator:
        """Minibatches of adjacent episode **pairs** for the trial-memory policy.

        Every epoch starts with one :meth:`compute_memory_checkpoints` sweep; afterwards the pairs are
        independent, so they are shuffled and cut into ``num_mini_batches`` minibatches *of pairs* (not of
        environments and not of timesteps). Following rsl-rl's convention the epoch loop lives in here.

        Yields a dictionary per minibatch::

            {
                "source": None | {obs, prev_actions, prev_rewards, prev_dones, mask},   # [S_src, n_src, ...]
                "source_slots": LongTensor [n_src],   # row of "target"/"memory" each source belongs to
                "target": {obs, prev_actions, prev_rewards, prev_dones, mask, loss_mask,
                           actions, old_actions_log_prob, old_mu, old_sigma,
                           advantages, returns, values,
                           row_lags},                                                   # [S_tgt, B, ...]
                "memory": Tensor [B, M, d],          # detached Zbar_e of every pair
                "positions": LongTensor [B],         # episode index inside its trial (0 == degenerate pair)
                "lags": LongTensor [B],              # updates since the pair's TRIAL started (0 == on-policy)
                "episode_ids": LongTensor [B],       # index into build_trial_pairs(), for diagnostics/tests
            }

        ``"source"`` is ``None`` when every pair in the minibatch is degenerate. Pairs whose source is missing
        (``positions == 0``) are simply absent from ``source_slots``; the caller must use the learned ``Z_init``
        for those rows.

        Two different lags are yielded and they are not interchangeable:

        * ``target["row_lags"]`` ``[S, B]`` -- per step, "how many updates ago was this row collected". This is
          what the lag-aware KL control uses: at ``num_steps_per_env=32`` with a 240-step carry region a trial
          closes seven or eight updates after its first step, so its *trial* lag is never <= 1 while its last
          rollout's rows are perfectly fresh,
        * ``lags`` ``[B]`` -- per pair, the lag of the whole trial (measured at its first step). This is the
          right one for the reconstruction canary: a pair is only guaranteed to reproduce its behavior
          log-probs when its *whole* trial, and therefore the memory checkpoint it is rebuilt from, is
          on-policy.
        """
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        index = self.build_trial_pairs()
        num_pairs = index["num_pairs"]
        if num_pairs == 0:
            raise ValueError(
                "No trial completed during this rollout, so there is nothing to update on. Every trial is"
                f" still open ({index['deferred_trials']}) or was dropped ({index['dropped_trials']}). Check"
                " that trial_done is published and that num_steps_per_env is not far shorter than a trial."
            )
        # A pool smaller than the requested number of minibatches is a transient (e.g. the first rollout,
        # where trials have not finished yet), not a configuration error -- shrink the split instead of
        # throwing the data away.
        num_mini_batches = min(num_mini_batches, num_pairs)
        mini_batch_size = num_pairs // num_mini_batches

        positions = index["ep_pos"]
        lags = index["ep_lag"]
        for _ in range(num_epochs):
            checkpoints = self.compute_memory_checkpoints(policy, sweep_chunk_size, memory_dtype)
            shuffled = torch.randperm(num_pairs, device=self.device)
            for i in range(num_mini_batches):
                episode_ids = shuffled[i * mini_batch_size : (i + 1) * mini_batch_size]
                target = self.episode_batch(episode_ids, include_terminal_token=False, include_targets=True)

                pair_positions = positions[episode_ids]
                source_slots = (pair_positions > 0).nonzero(as_tuple=False).squeeze(-1)
                if source_slots.numel() > 0:
                    source = self.episode_batch(
                        episode_ids[source_slots] - 1, include_terminal_token=True, include_targets=False
                    )
                else:
                    source = None

                # The memory the *pair* starts from is the incoming checkpoint of its source episode, Zbar_e.
                # The degenerate pair has no source, so it starts from its own checkpoint, which is Z_init.
                memory_ids = episode_ids - (pair_positions > 0).long()

                yield {
                    "source": source,
                    "source_slots": source_slots,
                    "target": target,
                    "memory": checkpoints[memory_ids].detach(),
                    "positions": pair_positions,
                    "lags": lags[episode_ids],
                    "episode_ids": episode_ids,
                }
