# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import torch
import torch.nn as nn
import torch.optim as optim
from collections.abc import Iterator
from contextlib import AbstractContextManager
from itertools import chain
from tensordict import TensorDict

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, ActorCriticTrialMemory
from rsl_rl.modules.actor_critic import upcast_from_half
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import string_to_callable

# Accepted spellings of the mixed-precision dtype, see :class:`PPO`'s ``amp_dtype``.
_AMP_DTYPES: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def _zeros_like_obs(obs: TensorDict | torch.Tensor) -> TensorDict | torch.Tensor:
    """An all-zero copy of an observation (works for both a ``TensorDict`` and a plain tensor)."""
    if isinstance(obs, torch.Tensor):
        return torch.zeros_like(obs)
    return obs.apply(torch.zeros_like)


class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic | ActorCriticRecurrent
    """The actor critic module."""

    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        adaptive_lr_max: float = 1e-2,
        kl_early_stop_factor: float | None = None,
        device: str = "cpu",
        normalize_advantage_per_mini_batch: bool = False,
        defer_obs_normalization: bool = False,
        # Trial-memory parameters (only used when the policy is an ActorCriticTrialMemory)
        trial_memory_sweep_chunk_size: int | None = None,
        trial_memory_cache_dtype: str | None = None,
        trial_carry_steps: int | None = None,
        max_policy_lag: int = 1,
        terminal_obs_key: str | None = None,
        amp_dtype: str | None = None,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg is not None:
            # Extract parameters used in ppo
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    f"Symmetry configuration exists but the function is not callable: "
                    f"{symmetry_cfg['data_augmentation_func']}"
                )
            # Check if the policy is compatible with symmetry
            if isinstance(policy, ActorCriticRecurrent):
                raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.policy = policy
        self.policy.to(self.device)

        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Create rollout storage
        self.storage: RolloutStorage | None = None
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        # Ceiling for the adaptive-KL LR controller. Upstream hardcodes 1e-2, which is ~100x the
        # base LR: with gSDE the first minibatches on fresh data read KL ~ 0, so the controller
        # multiplies LR x1.5 per minibatch (epochs x minibatches chances per update) and can blast
        # a transformer trunk apart mid-update before KL catches up (run-225872 postmortem:
        # sustained surrogate ~0.25 / clip_frac 0.7 / ratio_max 28 at nominal lr 1e-4).
        self.adaptive_lr_max = float(adaptive_lr_max)
        # Abort an update's remaining minibatches once the measured KL exceeds
        # ``kl_early_stop_factor * desired_kl`` (None = off). Batch-size-invariant overshoot guard;
        # only wired into the trial-pair update path.
        self.kl_early_stop_factor = None if kl_early_stop_factor is None else float(kl_early_stop_factor)
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        # If set, the observation normalizers are not updated during collection. The runner is then
        # responsible for calling :meth:`commit_obs_normalization` once the rollout has been consumed,
        # so that a rollout is acted and reconstructed under identical normalizer statistics.
        self.defer_obs_normalization = defer_obs_normalization

        # Trial-memory (hierarchical multi-episode) training.
        # Note: the routing is on the policy *class*, not on ``is_recurrent`` -- the recurrent minibatch
        #   generator splits on episode dones and would be wrong here (it knows nothing about trials or pairs).
        self.is_trial_memory = isinstance(policy, ActorCriticTrialMemory)
        # With a separate critic trunk the value function has NO incremental path: nothing can be evaluated
        # per collection step, so the values (and with them the timeout bootstrap and the GAE bootstrap) are
        # produced by a batched sweep in :meth:`compute_returns`. See ActorCriticTrialMemory's ``critic_trunk``.
        self.separate_critic_trunk = self.is_trial_memory and getattr(policy, "critic_trunk", "shared") == "separate"
        # Per-step ``time_outs`` of the current rollout, needed to re-apply the timeout bootstrap after the
        # sweep (at collection time the values are still zero, so the bootstrap there is a no-op).
        self._time_out_flags: torch.Tensor | None = None
        self.trial_memory_sweep_chunk_size = trial_memory_sweep_chunk_size
        # Rows reserved for a trial that is still open when a rollout ends, so that it is carried into the
        # next rollout instead of being thrown away (see RolloutStorage's ``carry_steps``). ``None`` (or any
        # negative value, which is how a config that cannot express None spells it) means one full rollout,
        # covering any trial no longer than the rollout itself. Only used for trial memory: with one episode
        # per trial there is nothing to carry that PPO could still use.
        self.trial_carry_steps = None if trial_carry_steps is None or trial_carry_steps < 0 else trial_carry_steps
        self.max_policy_lag = int(max_policy_lag)
        self.trial_memory_cache_dtype = (
            None if trial_memory_cache_dtype is None else getattr(torch, trial_memory_cache_dtype)
        )
        # Key under which the environment publishes the *true* terminal observation in ``extras``. In IsaacLab
        # the observation returned by ``step()`` is already the post-reset one, so without such a key the
        # terminal token's observation slot is zeros (see :meth:`process_env_step`).
        self.terminal_obs_key = terminal_obs_key
        if self.is_trial_memory:
            if self.rnd is not None:
                raise ValueError("RND is not supported with ActorCriticTrialMemory (the pair update has no RND path).")
            if self.symmetry is not None and (
                self.symmetry["use_data_augmentation"] or self.symmetry["use_mirror_loss"]
            ):
                raise ValueError("Symmetry augmentation is not supported for the trial-memory policy.")

        # Mixed precision (opt-in, trial-memory update only). ``None`` keeps the fp32 path bit-for-bit.
        # The activation memory of the pair update is what forces this: at 16k envs the trunk activations are
        # ~80 GB in fp32 at 8 minibatches (design doc section 7); fp16 halves that and is what makes 4 viable.
        self.amp_dtype = self._resolve_amp_dtype(amp_dtype)
        self._amp_device_type = torch.device(self.device).type
        # fp16 needs loss scaling (its gradients underflow); bf16 has fp32's exponent range and does not.
        # The scaler is CUDA-only here -- CPU fp16 autocast exists but is a debug path, not a training one.
        scaler_enabled = self.amp_dtype == torch.float16 and self._amp_device_type == "cuda"
        self.grad_scaler = torch.amp.GradScaler(self._amp_device_type) if scaler_enabled else None
        if self.amp_dtype is not None and not self.is_trial_memory:
            raise ValueError(
                "amp_dtype is only implemented for the trial-memory (episode-pair) update; the flat PPO path"
                " would silently keep running in fp32."
            )

    # ----------------------------------------------------------------------------------------------------------
    # Mixed precision helpers
    # ----------------------------------------------------------------------------------------------------------

    @staticmethod
    def _resolve_amp_dtype(amp_dtype: str | torch.dtype | None) -> torch.dtype | None:
        """Map the ``amp_dtype`` argument to a torch dtype (``None`` == full precision)."""
        if amp_dtype is None:
            return None
        if isinstance(amp_dtype, torch.dtype):
            return None if amp_dtype == torch.float32 else amp_dtype
        key = str(amp_dtype).lower()
        if key in ("none", "fp32", "float32", ""):
            return None
        if key not in _AMP_DTYPES:
            raise ValueError(f"Unknown amp_dtype '{amp_dtype}'. Expected one of {sorted(_AMP_DTYPES)} or None.")
        return _AMP_DTYPES[key]

    def _autocast(self) -> AbstractContextManager:
        """Autocast context for the trial-memory forward passes.

        Returns a genuine no-op when mixed precision is off, so that ``amp_dtype=None`` is numerically identical
        to the code before this flag existed (an ``autocast(enabled=False)`` block would also *disable* any
        enclosing autocast, which is a different statement).
        """
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self._amp_device_type, dtype=self.amp_dtype)

    def _autocast_generator(self, generator: Iterator[dict]) -> Iterator[dict]:
        """Advance ``generator`` under autocast, yield outside of it.

        The per-epoch ``no_grad`` memory sweep runs *inside* the pair generator (it is what the first ``next()``
        of every epoch does), so this is the only way to put the sweep under autocast without reaching into
        ``RolloutStorage``. Yielding outside the context keeps the caller's own autocast scoping explicit.
        """
        iterator = iter(generator)
        while True:
            with self._autocast():
                try:
                    batch = next(iterator)
                except StopIteration:
                    return
            yield batch

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        # A trial is the training-data unit but a rollout is the collection unit, and the two do not divide
        # each other once an environment terminates early. The carry region decouples them: an unfinished
        # trial survives the rollout boundary instead of being discarded.
        carry_steps = 0
        if self.is_trial_memory:
            carry_steps = num_transitions_per_env if self.trial_carry_steps is None else int(self.trial_carry_steps)
        # Create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
            carry_steps=carry_steps,
            max_policy_lag=self.max_policy_lag,
        )
        if self.separate_critic_trunk:
            self._time_out_flags = torch.zeros(
                self.storage.num_transitions_per_env, num_envs, 1, device=self.device
            )

    def act(self, obs: TensorDict) -> torch.Tensor:
        # Note: the trial-memory policy's "hidden state" is the whole memory Z [N, M, d]; storing it per step
        #   would cost tens of GB and it is recomputed from the raw data anyway, so it is not saved.
        if self.policy.is_recurrent and not self.is_trial_memory:
            self.transition.hidden_states = self.policy.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        if self.separate_critic_trunk:
            # No per-step critic: placeholder zeros, overwritten by the sweep in compute_returns(). They also
            # make the collection-time timeout bootstrap below an exact no-op, which is why it is re-applied
            # (from ``raw_rewards``) once the real values exist.
            self.transition.values = torch.zeros(
                self.transition.actions.shape[0], 1, device=self.device, dtype=self.transition.actions.dtype
            )
        else:
            self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        # Update the normalizers
        # Note: When deferred, the runner commits the update after update() via commit_obs_normalization()
        if not self.defer_obs_normalization:
            self.commit_obs_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        # The rewards the acting path put into its tokens, kept separately from the ones GAE consumes
        self.transition.raw_rewards = rewards.clone()
        self.transition.dones = dones
        # Trial boundary. Defaults to the episode done, i.e. one episode per trial.
        trial_dones = extras.get("trial_done", dones) if extras is not None else dones
        self.transition.trial_dones = trial_dones.to(self.device)

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        # Note: Only at a trial end. Inside a trial, GAE keeps bootstrapping through the episode
        #   boundary itself (next_is_not_terminal stays 1), so adding the value here would count it twice.
        if "time_outs" in extras:
            time_outs = extras["time_outs"].unsqueeze(1).to(self.device).float()
            trial_ends = self.transition.trial_dones.view(-1, 1).to(self.device).float()
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * time_outs * trial_ends, 1)
        else:
            time_outs = None

        # Remember the timeout flags of this rollout so that the bootstrap can be re-applied once the batched
        # critic sweep has produced the values (with the placeholder zeros the line above changed nothing).
        if self._time_out_flags is not None:
            if self.storage.step == 0:
                self._time_out_flags.zero_()
            row = self.storage.row
            if time_outs is None:
                self._time_out_flags[row].zero_()
            else:
                self._time_out_flags[row].copy_(time_outs)

        # Advance the trial-memory policy's acting state (before the episode/trial resets below)
        if self.is_trial_memory:
            self._advance_trial_memory(obs, rewards, dones, extras)

        # Record the transition
        trial_dones = self.transition.trial_dones
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        # Two distinct reset signals: an episode boundary clears the short-term tokens and the KV cache, a trial
        # boundary additionally resets the persistent memory Z back to Z_init.
        if self.is_trial_memory:
            self.policy.reset(dones, trial_dones=trial_dones)
        else:
            self.policy.reset(dones)

    def _advance_trial_memory(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Run the acting-boundary protocol of :class:`ActorCriticTrialMemory` for one environment step.

        Order (documented by the policy): ``record_transition`` -> ``append_terminal_token`` -> ``update_memory``,
        with the episode/trial resets happening afterwards in :meth:`process_env_step`.

        The terminal token exists so that the memory writer can see the outcome of the attempt; its content is
        ``(o_{T+1}, a_T, r_T, d_T)``. In IsaacLab the observation handed to this method is already the
        **post-reset** observation of a terminated environment, so it must not be used: it would leak the next
        episode's state into ``H_e``. If the environment publishes the true terminal observation under
        :attr:`terminal_obs_key` we use (and store) that, otherwise the observation slot is zeros on both the
        acting and the reconstruction path -- the load-bearing content is the final reward and done flag.
        """
        terminal_obs = None
        if self.terminal_obs_key is not None and extras is not None and self.terminal_obs_key in extras:
            terminal_obs = extras[self.terminal_obs_key].to(self.device)
            self.transition.terminal_observations = terminal_obs

        # Stash r_T / d_T so that the terminal token can carry them
        self.policy.record_transition(rewards, dones)

        done_ids = dones.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            token_obs = terminal_obs if terminal_obs is not None else _zeros_like_obs(obs)
            self.policy.append_terminal_token(token_obs, env_ids=done_ids)
            self.policy.update_memory(done_ids)

    def commit_obs_normalization(self, obs_batch: TensorDict) -> None:
        """Update the observation normalizers from a batch of observations.

        Called on every collection step unless :attr:`defer_obs_normalization` is set, in which case the
        runner calls it once after :meth:`update`, so that acting and reconstruction of a rollout share
        the same normalizer statistics.

        Args:
            obs_batch: Observations, either a single step ``[num_envs, ...]`` or a stored rollout
                ``[num_transitions, num_envs, ...]``. Extra leading dimensions are flattened.
        """
        # The normalizers reduce over dim 0 only, so collapse any extra leading (time) dimensions
        batch_dims = getattr(obs_batch, "batch_dims", 1)
        if batch_dims > 1:
            obs_batch = obs_batch.flatten(0, batch_dims - 1)
        self.policy.update_normalization(obs_batch)
        if self.rnd:
            self.rnd.update_normalization(obs_batch)

    def compute_returns(self, obs: TensorDict) -> None:
        if self.separate_critic_trunk:
            # Batched critic sweep instead of the per-step values: fills storage.values for every live row and
            # returns the GAE bootstrap. Deliberately NOT under autocast -- it is a once-per-rollout no_grad
            # pass whose output becomes the regression target of the whole update, so it stays in fp32.
            last_values = self.storage.compute_critic_values(
                self.policy, chunk_size=self.trial_memory_sweep_chunk_size, last_obs=obs
            )
            self._reapply_timeout_bootstrap()
        else:
            # Compute value for the last step
            last_values = self.policy.evaluate(obs).detach()
        self.storage.compute_returns(
            last_values, self.gamma, self.lam, normalize_advantage=not self.normalize_advantage_per_mini_batch
        )

    def _reapply_timeout_bootstrap(self) -> None:
        """Re-apply the trial-gated timeout bootstrap now that the swept values exist.

        Identical arithmetic to the collection-time line in :meth:`process_env_step`
        (``rewards += gamma * values * time_outs * trial_ends``), just recomputed from ``raw_rewards`` -- which
        is exactly what that buffer is for. Recomputing from the raw rewards (rather than adding to the stored
        ones) also makes this idempotent, so a carried row that is swept again in a later rollout never
        accumulates two bootstraps.
        """
        storage = self.storage
        time_outs = self._time_out_flags
        if time_outs is None:
            return
        trial_ends = storage.trial_dones.float()
        # Same association as the collection-time expression, so the result is bit-identical to what a
        # per-step critic would have produced: raw + gamma * ((values * time_outs) * trial_ends).
        storage.rewards.copy_(storage.raw_rewards + self.gamma * (storage.values * time_outs * trial_ends))

    def update(self) -> dict[str, float]:
        # The trial-memory policy trains on adjacent episode pairs, not on flat transitions
        if self.is_trial_memory:
            return self._update_trial_memory()

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None
        # Probability-ratio diagnostics (per-minibatch reductions of
        # ratio = exp(new_logp - old_logp)). Mean/std/clip_frac are averaged
        # over minibatches; min/max are taken over all minibatches.
        mean_ratio = 0.0
        mean_ratio_std = 0.0
        mean_ratio_clip_frac = 0.0
        ratio_min = float("inf")
        ratio_max = float("-inf")

        # Get mini batch generator
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            num_aug = 1  # Number of augmentations per sample. Starts at 1 for no augmentation.
            original_batch_size = obs_batch.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # Compute number of augmentations per sample
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            # Note: We only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(self.adaptive_lr_max, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Probability-ratio diagnostics
            with torch.no_grad():
                flat_ratio = ratio.detach().reshape(-1)
                mean_ratio += flat_ratio.mean().item()
                mean_ratio_std += flat_ratio.std().item()
                mean_ratio_clip_frac += ((flat_ratio - 1.0).abs() > self.clip_param).float().mean().item()
                batch_min = flat_ratio.min().item()
                batch_max = flat_ratio.max().item()
                if batch_min < ratio_min:
                    ratio_min = batch_min
                if batch_max > ratio_max:
                    ratio_max = batch_max

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # Compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the action_batch from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            # TODO: Move this processing to inside RND module.
            if self.rnd:
                # Extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_ratio /= num_updates
        mean_ratio_std /= num_updates
        mean_ratio_clip_frac /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Clear the storage
        self.storage.clear()
        self.storage.policy_version += 1

        # Construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "ratio_mean": mean_ratio,
            "ratio_std": mean_ratio_std,
            "ratio_min": ratio_min,
            "ratio_max": ratio_max,
            "ratio_clip_frac": mean_ratio_clip_frac,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    # ----------------------------------------------------------------------------------------------------------
    # Trial-memory update (adjacent episode pairs)
    # ----------------------------------------------------------------------------------------------------------

    def trial_pair_forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct one minibatch of pairs and return the target hidden states and the memory fed to them.

        Implements the pair recipe:

        1. ``Zbar_e`` comes from the per-epoch sweep and is already **stopgrad**,
        2. with a source: ``H_e^theta = forward_sequence(tau_e, Zbar_e)`` *with grad*, then
           ``Z_{e+1}^theta = write_memory(Zbar_e, H_e^theta)``,
        3. without a source (the degenerate ``(None, tau_1)`` pair): ``Z_{e+1}^theta = Z_init``,
        4. ``H_{e+1}^theta = forward_sequence(tau_{e+1}, Z_{e+1}^theta)`` -- ``Z_{e+1}^theta`` is **not**
           detached; detaching it would delete the entire memory-learning signal.

        Returns:
            ``H_{e+1}`` ``[S, B, d]`` and ``Z_{e+1}`` ``[B, M, d]``.
        """
        target = batch["target"]
        param_dtype = self.policy.z_init.dtype
        # Zbar_e, detached by the generator (the cache may be kept in a smaller dtype than the parameters)
        incoming = batch["memory"].to(dtype=param_dtype)
        batch_size = incoming.shape[0]

        # Note: the degenerate pair uses the *differentiable* Z_init rather than its detached copy from the
        #   sweep, so that a first-episode loss still trains the learned NO_MEMORY tokens (the values are
        #   identical, only the gradient path differs).
        memory = self.policy.initial_memory(batch_size, device=incoming.device).clone()
        source_slots = batch["source_slots"]
        if batch["source"] is not None and source_slots.numel() > 0:
            source = batch["source"]
            source_memory = incoming[source_slots]
            source_hidden = self.policy.forward_sequence(
                source["obs"],
                source["prev_actions"],
                source["prev_rewards"],
                source["prev_dones"],
                memory=source_memory,
                mask=source["mask"],
            )
            written, _ = self.policy.write_memory(
                source_memory, source_hidden.transpose(0, 1), mask=source["mask"].transpose(0, 1)
            )
            memory = memory.index_copy(0, source_slots, written)

        hidden = self.policy.forward_sequence(
            target["obs"],
            target["prev_actions"],
            target["prev_rewards"],
            target["prev_dones"],
            memory=memory,
            mask=target["mask"],
        )
        return hidden, memory

    def _update_trial_memory(self) -> dict[str, float]:
        """PPO over adjacent episode pairs: the clipped loss is taken on the target episode only."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        # Probability-ratio diagnostics -- the reconstruction canary of the design doc lives here.
        # Note: the canary statement ("with unchanged parameters the ratio is 1") only holds for trials whose
        #   behavior policy IS the current one. A trial carried across a rollout boundary was partly collected
        #   under the previous policy, so its ratio is legitimately != 1 at epoch 0. Reporting one number over
        #   both would make a healthy run look broken, so the lag-0 subset is measured separately and the
        #   *_lag0 keys are the ones to read as the canary.
        mean_ratio = 0.0
        mean_ratio_std = 0.0
        mean_ratio_clip_frac = 0.0
        ratio_min = float("inf")
        ratio_max = float("-inf")
        lag0_ratio_sum = 0.0
        lag0_clip_frac_sum = 0.0
        lag0_ratio_min = float("inf")
        lag0_ratio_max = float("-inf")
        lag0_batches = 0
        # The canary proper: the very FIRST minibatch, on lag-0 pairs. It is the only point in the update at
        # which the parameters are still exactly the ones that acted -- the optimizer steps after every
        # minibatch, so even the rest of epoch 0 is already off-policy and its ratio is *supposed* to move.
        first_batch_lag0_dev = 0.0
        num_updates = 0
        kl_early_stopped = False
        # Snapshot the pool accounting before update() consumes and clears the storage
        pool = dict(self.storage.build_trial_pairs())

        generator = self.storage.trial_pair_mini_batch_generator(
            self.policy,
            self.num_mini_batches,
            self.num_learning_epochs,
            sweep_chunk_size=self.trial_memory_sweep_chunk_size,
            memory_dtype=self.trial_memory_cache_dtype,
        )

        for batch in self._autocast_generator(generator):
            target = batch["target"]
            # Autocast covers the memory-hungry part: the source recompute, the writer, the target recompute and
            # the heads. Everything downstream of it is per-step scalars, so it is upcast to fp32 below -- that is
            # both the AMP convention (losses in fp32, scaled) and what keeps the ratio diagnostics meaningful.
            with self._autocast():
                hidden, memory = self.trial_pair_forward(batch)

                # Every quantity below lives on the target episode's acting steps only (no terminal token,
                # no padding)
                loss_mask = target["loss_mask"]
                distribution = self.policy.update_distribution_from_hidden(hidden)
                actions_log_prob = distribution.log_prob(target["actions"]).sum(dim=-1)[loss_mask]
                entropy_batch = distribution.entropy().sum(dim=-1)[loss_mask]
                mu_batch = distribution.mean[loss_mask]
                sigma_batch = distribution.stddev[loss_mask]
                if self.separate_critic_trunk:
                    # Second trunk, same tokens: the value head consumes the critic pathway's readout, never
                    # the actor's. ``memory`` is Z_{e+1}, the memory the TARGET episode was acted with (not the
                    # source's Zbar_e); forward_sequence_critic detaches it, so no value gradient reaches the
                    # writer, z_init or the actor trunk.
                    value_batch = self.policy.forward_sequence_critic(
                        target["obs"],
                        target["prev_actions"],
                        target["prev_rewards"],
                        target["prev_dones"],
                        memory=memory,
                        mask=target["mask"],
                    )[loss_mask]
                else:
                    value_batch = self.policy.value_from_hidden(hidden)[loss_mask]

            # Out of autocast: the loss arithmetic runs in fp32. The upcast is a no-op for anything that is not
            # half precision, so the fp32 path (and the fp64 exactness tests) are unchanged.
            actions_log_prob = upcast_from_half(actions_log_prob)
            entropy_batch = upcast_from_half(entropy_batch)
            mu_batch = upcast_from_half(mu_batch)
            sigma_batch = upcast_from_half(sigma_batch)
            value_batch = upcast_from_half(value_batch)

            # The PPO denominator is always the stored behavior log-prob; it is never recomputed.
            old_actions_log_prob_batch = target["old_actions_log_prob"][loss_mask].squeeze(-1)
            advantages_batch = target["advantages"][loss_mask].squeeze(-1)
            returns_batch = target["returns"][loss_mask]
            target_values_batch = target["values"][loss_mask]
            old_mu_batch = target["old_mu"][loss_mask]
            old_sigma_batch = target["old_sigma"][loss_mask]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Compute KL divergence and adapt the learning rate.
            # Note: fp32 by construction -- ``mu_batch``/``sigma_batch`` were upcast above. The KL drives the
            #   learning rate, so it must not inherit fp16's ~1e-3 relative error (nor its exponent range: the
            #   ratio sigma_new / sigma_old and the squares below can underflow in half precision).
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(self.adaptive_lr_max, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

                # KL early stop: once this update has moved the policy past ``factor x desired_kl``,
                # abort the remaining minibatches (this batch's step included -- the KL above is
                # measured BEFORE stepping on it). The per-minibatch LR throttle reacts too slowly at
                # scale: large coherent batches give Adam full-size normalized steps, so epochs x
                # minibatches steps overshoot before KL feedback catches up (the 226401 collapse:
                # success 0.65 -> 0.14 in two updates while the LR sat at its floor). ``kl_mean`` is
                # already all-reduced, so every rank breaks in the same place.
                if (
                    self.kl_early_stop_factor is not None
                    and num_updates > 0  # never abort before one step (also keeps the /num_updates means valid)
                    and kl_mean > self.kl_early_stop_factor * self.desired_kl
                ):
                    kl_early_stopped = True
                    break

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - old_actions_log_prob_batch)
            surrogate = -advantages_batch * ratio
            surrogate_clipped = -advantages_batch * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Probability-ratio diagnostics
            with torch.no_grad():
                flat_ratio = ratio.detach().reshape(-1)
                mean_ratio += flat_ratio.mean().item()
                mean_ratio_std += flat_ratio.std().item()
                mean_ratio_clip_frac += ((flat_ratio - 1.0).abs() > self.clip_param).float().mean().item()
                ratio_min = min(ratio_min, flat_ratio.min().item())
                ratio_max = max(ratio_max, flat_ratio.max().item())
                # ... and the same, restricted to the fully on-policy trials (the actual canary)
                lag0 = (batch["lags"] == 0).unsqueeze(0).expand_as(loss_mask)[loss_mask]
                if bool(lag0.any()):
                    lag0_ratio = flat_ratio[lag0]
                    lag0_ratio_sum += lag0_ratio.mean().item()
                    lag0_clip_frac_sum += ((lag0_ratio - 1.0).abs() > self.clip_param).float().mean().item()
                    lag0_ratio_min = min(lag0_ratio_min, lag0_ratio.min().item())
                    lag0_ratio_max = max(lag0_ratio_max, lag0_ratio.max().item())
                    lag0_batches += 1
                    if num_updates == 0:  # parameters still bit-identical to the ones that acted
                        first_batch_lag0_dev = (lag0_ratio - 1.0).abs().max().item()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step.
            # Order under fp16: backward on the SCALED loss -> (multi-GPU all-reduce, still scaled) ->
            #   ``scaler.unscale_(optimizer)`` -> clip -> ``scaler.step``. The unscale MUST come before the clip:
            #   clipping scaled gradients would compare a ~65536x inflated norm against ``max_grad_norm`` and clip
            #   every step to a ~1/65536 effective threshold, which looks like "training just got slower" rather
            #   than like a bug. Reducing while still scaled is fine and deliberate: the scale is one scalar shared
            #   by all ranks, so averaging commutes with it, and a rank whose backward overflowed propagates the
            #   inf to everyone, keeping the skip decision (and hence the scale) identical across ranks.
            self.optimizer.zero_grad()
            if self.grad_scaler is not None:
                self.grad_scaler.scale(loss).backward()
            else:
                loss.backward()
            if self.is_multi_gpu:
                # Not every parameter is reached by every minibatch (Z_init is only trained by the degenerate
                # pairs), and reduce_parameters() flattens exactly the parameters that have a gradient -- so
                # ranks would disagree on the buffer layout. Materialize the missing gradients as zeros.
                # (Zeros are scale-invariant, so doing this on scaled gradients is safe.)
                for param in self.policy.parameters():
                    if param.grad is None:
                        param.grad = torch.zeros_like(param)
                self.reduce_parameters()
            if self.grad_scaler is not None:
                self.grad_scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            if self.grad_scaler is not None:
                # Skips the step (and lowers the scale) if the unscale found an inf/NaN.
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            num_updates += 1

        num_updates = max(num_updates, 1)
        # Carries every still-open trial into the next rollout, then moves on to the next policy version.
        self.storage.clear()
        self.storage.policy_version += 1

        lag0_batches = max(lag0_batches, 1)
        return {
            "value_function": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "ratio_mean": mean_ratio / num_updates,
            "ratio_std": mean_ratio_std / num_updates,
            "ratio_min": ratio_min,
            "ratio_max": ratio_max,
            "ratio_clip_frac": mean_ratio_clip_frac / num_updates,
            # The reconstruction canary: on-policy trials only (see the note above)
            "ratio_mean_lag0": lag0_ratio_sum / lag0_batches,
            "ratio_min_lag0": lag0_ratio_min,
            "ratio_max_lag0": lag0_ratio_max,
            "ratio_clip_frac_lag0": lag0_clip_frac_sum / lag0_batches,
            # max |ratio - 1| over the first minibatch, lag-0 pairs only: ~0 iff the reconstruction reproduces
            # the behavior policy. This is the design doc's section 13 canary. 0.0 also means "no lag-0 pair
            # landed in the first minibatch", so read it together with pool_pairs_lag0.
            "ratio_max_dev_lag0_first_mb": first_batch_lag0_dev,
            # Trial-pool accounting. Anything but zeros in the ``dropped``/``envs_without_data`` lines means
            # collected experience is being thrown away.
            # The LIVE adaptive LR after this update -- the run-225872 collapse (LR silently ramping
            # x1.5/minibatch to the ceiling) was invisible without it.
            "adaptive_lr": float(self.learning_rate),
            # 1.0 when the KL early stop aborted this update; ``num_updates`` (via pool metrics and
            # the per-update means) shows how many minibatches actually ran.
            "kl_early_stopped": 1.0 if kl_early_stopped else 0.0,
            "kl_minibatches_run": float(num_updates),
            "pool_pairs": float(pool["num_pairs"]),
            "pool_pairs_lag0": float(pool["lag0_pairs"]),
            "pool_trials": float(pool["num_trials"]),
            "pool_deferred_trials": float(pool["deferred_trials"]),
            "pool_dropped_episodes": float(pool["dropped_episodes"]),
            "pool_dropped_lagged_trials": float(pool["dropped_lagged_trials"]),
            "pool_dropped_truncated_trials": float(pool["dropped_truncated_trials"]),
            "pool_envs_without_data": float(pool["envs_without_data"]),
        }

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Broadcast policy parameters
        for param in self.policy.parameters():
            torch.distributed.broadcast(param.data, src=0)

        # Broadcast RND parameters if they exist
        if self.rnd:
            for param in self.rnd.predictor.parameters():
                torch.distributed.broadcast(param.data, src=0)

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if self.rnd:
            grads += [param.grad.view(-1) for param in self.rnd.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)

        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        # Get all parameters
        all_params = self.policy.parameters()
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())

        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
