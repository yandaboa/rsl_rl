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
    ) -> None:
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
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
        # Whether each environment sat on a trial boundary (Z == Z_init) at step 0 of this rollout. Rolled
        # forward by :meth:`clear` from the trial dones of the last stored step.
        self.at_trial_start = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        # Cached episode/trial segmentation of the current rollout (invalidated whenever the data changes)
        self._pair_index: dict[str, torch.Tensor | int] | None = None

        # Counter for the number of transitions stored
        self.step = 0

    def add_transitions(self, transition: Transition) -> None:
        # Check if the transition is valid
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")

        # Core
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        # Fall back to the episode dones so that callers unaware of trials keep the stock behavior
        trial_dones = transition.trial_dones if transition.trial_dones is not None else transition.dones
        self.trial_dones[self.step].copy_(trial_dones.view(-1, 1))
        # Token rewards. Fall back to the (possibly modified) rewards for callers that do not set them.
        raw_rewards = transition.raw_rewards if transition.raw_rewards is not None else transition.rewards
        self.raw_rewards[self.step].copy_(raw_rewards.view(-1, 1))
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
            self.terminal_observations[self.step].copy_(transition.terminal_observations)
        # The rollout changed, so any cached segmentation is stale
        self._pair_index = None

        # For distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # For reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

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
            self.saved_hidden_state_a[i][self.step].copy_(hidden_state_a[i])
            self.saved_hidden_state_c[i][self.step].copy_(hidden_state_c[i])

    def clear(self) -> None:
        # Carry the trial phase into the next rollout: an environment starts the next rollout on a trial
        # boundary (memory back at Z_init) exactly when the last stored step closed its trial.
        if self.step > 0:
            self.at_trial_start = self.trial_dones[self.step - 1].reshape(-1).bool().clone()
        self.step = 0
        self._pair_index = None

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
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

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
        genuine terminal (e.g. ``abnormal_robot``) resets an environment early and desynchronizes it. Trials
        that are not fully contained in the rollout (open at either edge) are dropped, and the number of
        dropped episodes/trials is reported.

        Returns (and caches) a dictionary of 1-D tensors indexed by *valid* episode:

        * ``ep_start`` / ``ep_len`` / ``ep_env``: rollout step of the first transition, number of acting steps,
          environment index,
        * ``ep_pos``: index of the episode inside its trial (0 == first episode, i.e. the degenerate pair),
        * ``num_pairs``: number of valid episodes == number of pairs,
        * ``max_len``: longest valid episode, ``dropped_*``: accounting of what was thrown away.
        """
        if self._pair_index is not None:
            return self._pair_index

        num_steps, num_envs, device = self.num_transitions_per_env, self.num_envs, self.device
        # A trial end always closes the episode it ends, even if the caller only flagged the trial.
        dones = self.dones.reshape(num_steps, num_envs).bool() | self.trial_dones.reshape(num_steps, num_envs).bool()
        trial_dones = self.trial_dones.reshape(num_steps, num_envs).bool()

        # Per-step segment ids: the index (within the environment) of the episode/trial the step belongs to.
        # Exclusive cumulative sum, because a done flags the *last* step of its segment.
        ep_id = torch.cumsum(dones.long(), dim=0) - dones.long()
        trial_id = torch.cumsum(trial_dones.long(), dim=0) - trial_dones.long()

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

        # A trial is usable only if it both started and ended inside this rollout, and all of its episodes closed.
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
        trial_local = torch.arange(num_trials, device=device) - trial_offset[trial_env]
        trial_started = (trial_local > 0) | self.at_trial_start.to(device)[trial_env]
        valid_trial = trial_closed & trial_all_closed & trial_started

        keep = valid_trial[ep_trial]
        selected = keep.nonzero(as_tuple=False).squeeze(-1)
        ep_start_c = ep_start[selected]
        ep_len_c = ep_len[selected]
        ep_env_c = ep_env[selected]
        ep_pos_c = ep_pos[selected]

        # Filtering removes whole trials, so a kept episode's predecessor inside its trial stays at index - 1.
        if selected.numel() > 0:
            successor = ep_pos_c > 0
            if bool(successor.any()):
                prev = successor.nonzero(as_tuple=False).squeeze(-1) - 1
                assert bool((ep_env_c[prev] == ep_env_c[successor]).all()), "pair source/target env mismatch"
                assert bool((ep_pos_c[prev] + 1 == ep_pos_c[successor]).all()), "pair source/target are not adjacent"

        dropped_eps = int((~keep).sum().item())
        dropped_trials = int((~valid_trial).sum().item())
        dropped_envs = int(torch.unique(ep_env[~keep]).numel()) if dropped_eps else 0
        if verbose and dropped_eps > 0:
            print(
                f"[RolloutStorage] trial pairs: kept {selected.numel()} episodes in"
                f" {int(valid_trial.sum().item())} trials; dropped {dropped_eps} episodes in {dropped_trials}"
                f" incomplete/edge trials across {dropped_envs} environments."
            )

        self._pair_index = {
            "ep_start": ep_start_c,
            "ep_len": ep_len_c,
            "ep_env": ep_env_c,
            "ep_pos": ep_pos_c,
            "num_pairs": int(selected.numel()),
            "num_trials": int(valid_trial.sum().item()),
            "max_len": int(ep_len_c.max().item()) if selected.numel() > 0 else 0,
            "dropped_episodes": dropped_eps,
            "dropped_trials": dropped_trials,
            "dropped_envs": dropped_envs,
        }
        return self._pair_index

    def episode_batch(
        self, episode_ids: torch.Tensor, include_terminal_token: bool, include_targets: bool
    ) -> dict[str, torch.Tensor | TensorDict]:
        """Gather a padded ``[S, B, ...]`` batch of episodes from the flat rollout buffers.

        Args:
            episode_ids: Indices into the arrays returned by :meth:`build_trial_pairs`.
            include_terminal_token: Whether to append the terminal token ``x_{T+1} = Embed(o_{T+1}, a_T, r_T,
                d_T)``. Sources need it (the writer must see the episode outcome); targets do not, since it emits
                no action and takes no loss.
            include_targets: Whether to also gather the PPO targets (actions, old log-probs, advantages, ...).

        Returns:
            A dictionary with the token inputs (``obs``, ``prev_actions``, ``prev_rewards``, ``prev_dones``), the
            validity ``mask`` ``[S, B]`` and -- with ``include_targets`` -- the per-step PPO targets plus
            ``loss_mask`` ``[S, B]``, which is ``True`` only on the acting steps.
        """
        index = self.build_trial_pairs()
        num_steps, num_envs, device = self.num_transitions_per_env, self.num_envs, self.device
        starts = index["ep_start"][episode_ids]
        lengths = index["ep_len"][episode_ids]
        envs = index["ep_env"][episode_ids]
        num_tokens = int(lengths.max().item()) + (1 if include_terminal_token else 0)

        offsets = torch.arange(num_tokens, device=device).unsqueeze(1)  # [S, 1]
        action_mask = offsets < lengths.unsqueeze(0)  # [S, B] acting steps
        is_terminal = offsets == lengths.unsqueeze(0)  # [S, B] the extra terminal row
        token_mask = action_mask | is_terminal if include_terminal_token else action_mask

        current = (starts.unsqueeze(0) + offsets).clamp(max=num_steps - 1)
        previous = (current - 1).clamp(min=0)
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
        index = self.build_trial_pairs()
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
                    batch = self.episode_batch(source_ids, include_terminal_token=True, include_targets=False)
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
                           advantages, returns, values},                                # [S_tgt, B, ...]
                "memory": Tensor [B, M, d],          # detached Zbar_e of every pair
                "positions": LongTensor [B],         # episode index inside its trial (0 == degenerate pair)
                "episode_ids": LongTensor [B],       # index into build_trial_pairs(), for diagnostics/tests
            }

        ``"source"`` is ``None`` when every pair in the minibatch is degenerate. Pairs whose source is missing
        (``positions == 0``) are simply absent from ``source_slots``; the caller must use the learned ``Z_init``
        for those rows.
        """
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        index = self.build_trial_pairs()
        num_pairs = index["num_pairs"]
        mini_batch_size = num_pairs // num_mini_batches
        if mini_batch_size == 0:
            raise ValueError(
                f"Only {num_pairs} complete episode pairs survived the rollout, which is fewer than"
                f" num_mini_batches={num_mini_batches}. Collect longer rollouts or use fewer minibatches."
            )

        positions = index["ep_pos"]
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
                    "episode_ids": episode_ids,
                }
