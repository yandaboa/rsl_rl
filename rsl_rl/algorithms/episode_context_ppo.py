# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO for the single-episode context transformer.

This is the **stock** :class:`~rsl_rl.algorithms.ppo.PPO` -- ``compute_returns`` and the whole ``update()``
are inherited byte-for-byte from the known-good implementation. There are four overrides:

* :meth:`init_storage`, which builds an :class:`EpisodeContextRolloutStorage` instead of a plain
  :class:`RolloutStorage`; that storage's ``recurrent_mini_batch_generator`` then feeds PPO's existing recurrent
  branch (``policy.is_recurrent`` is ``True``) with ``[W, B, ...]`` minibatches whose prefix rides in the
  hidden-state slot,
* :meth:`act`, and ONLY when ``eval_env_fraction > 0``: the last envs execute the distribution mean instead
  of a sample (a deterministic eval pool, excluded from the update by the storage),
* :meth:`update`, and ONLY when ``grad_accumulation_steps > 1``: a logical minibatch is processed as several
  micro-batches. ``1`` (the default) delegates straight to the frozen ``ppo.py`` update,
* :meth:`process_env_step`, and ONLY when the policy carries a cross-episode memory (``memory_tokens > 0``): the
  frozen ``ppo.py`` ends its step with ``policy.reset(dones)``, which -- with no trial signal -- would put ``Z``
  back to ``z_init`` at every episode boundary, and it has nowhere to hand the storage the ``H`` of the episode
  that just finished. With ``memory_tokens == 0`` the override is a plain ``super()`` call.

Everything else a reader might expect to be different here -- the ratio, the KL schedule, the clipping, the
diagnostics -- is not. The one loss addition is opt-in and off by default: ``noise_prior_kl_coef > 0`` adds
``KL(pi || N(0, I))`` on the first ``noise_prior_dims`` action dims (the hook itself lives in ``ppo.py``, next
to the entropy term).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import warnings
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.actor_critic_episode_context import ActorCriticEpisodeContext
from rsl_rl.storage.episode_context_storage import EpisodeContextRolloutStorage


class EpisodeContextPPO(PPO):
    """PPO whose rollout storage keeps the per-environment frame ring the context transformer re-infers from."""

    def __init__(
        self,
        policy: ActorCriticEpisodeContext,
        *args,
        defer_obs_normalization: bool = True,
        noise_prior_kl_coef: float = 0.0,
        noise_prior_dims: int = 0,
        eval_env_fraction: float = 0.0,
        grad_accumulation_steps: int = 1,
        **kwargs,
    ) -> None:
        """Same signature as :class:`PPO`, except that ``defer_obs_normalization`` defaults to ``True``.

        It has to: the update re-infers every window from the RAW stored frames, so if the normalizer moved
        during collection the reconstruction would run under different statistics than the acting path did and
        the PPO ratio would be off at epoch 0 by construction.

        ``noise_prior_kl_coef`` / ``noise_prior_dims`` add ``coef * KL(pi_z || N(0, I))`` on the first
        ``noise_prior_dims`` action dims (RFS: those dims are the frozen flow's x0, which is only valid inside
        the unit Gaussian it was trained on). ``0`` leaves the loss untouched.

        ``eval_env_fraction`` carves a deterministic eval pool out of the LAST environments: they act on the
        distribution MEAN and their rows never enter the update. Their success rate is what a success-gated
        curriculum should grade on -- a sampled rollout under gSDE runs well below deterministic capability,
        so a curriculum gated on it stalls. ``0`` (the default) is the stock behavior, bit for bit.

        ``grad_accumulation_steps`` splits every minibatch into that many micro-batches, each forwarded and
        backwarded on its own (loss scaled by ``1/G``) before ONE optimizer step. The transitions per optimizer
        step are unchanged -- only the peak activation memory is divided by ``G``, which is what makes a long
        context at a large environment count fit. ``1`` (the default) runs the frozen ``ppo.py`` update.
        """
        if not isinstance(policy, ActorCriticEpisodeContext):
            raise ValueError(
                "EpisodeContextPPO requires an ActorCriticEpisodeContext policy (it is what owns the KV cache and"
                f" the windowed forward); got {type(policy).__name__}."
            )
        if not defer_obs_normalization and getattr(policy, "actor_obs_normalization", False):
            warnings.warn(
                "EpisodeContextPPO with defer_obs_normalization=False: the rollout is acted and reconstructed"
                " under different observation-normalizer statistics, so the epoch-0 PPO ratio will not be 1.",
                stacklevel=2,
            )
        super().__init__(policy, *args, defer_obs_normalization=defer_obs_normalization, **kwargs)
        assert noise_prior_kl_coef >= 0.0, "noise_prior_kl_coef must be non-negative."
        assert 0 <= noise_prior_dims <= policy.num_actions, (
            f"noise_prior_dims={noise_prior_dims} does not fit the policy's {policy.num_actions}-d action."
        )
        self.noise_prior_kl_coef = float(noise_prior_kl_coef)
        self.noise_prior_dims = int(noise_prior_dims)
        assert 0.0 <= eval_env_fraction < 1.0, f"eval_env_fraction must be in [0, 1), got {eval_env_fraction}."
        self.eval_env_fraction = float(eval_env_fraction)
        assert grad_accumulation_steps >= 1, f"grad_accumulation_steps must be >= 1, got {grad_accumulation_steps}."
        self.grad_accumulation_steps = int(grad_accumulation_steps)
        # Filled in by init_storage, which is the first place the environment count is known.
        self.eval_env_ids: torch.Tensor | None = None
        # ``None`` (not an all-False mask) with no eval pool: act() then takes the untouched stock path.
        self._eval_mask: torch.Tensor | None = None

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        num_eval_envs = round(self.eval_env_fraction * num_envs)
        assert num_eval_envs < num_envs, (
            f"eval_env_fraction={self.eval_env_fraction} leaves no training environment ({num_eval_envs} of"
            f" {num_envs} would be eval)."
        )
        if num_eval_envs > 0:
            self.eval_env_ids = torch.arange(num_envs, device=self.device)[-num_eval_envs:]
            self._eval_mask = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
            self._eval_mask[self.eval_env_ids] = True
        else:
            self.eval_env_ids = torch.zeros(0, dtype=torch.long, device=self.device)
            self._eval_mask = None

        self.storage = EpisodeContextRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
            num_eval_envs=num_eval_envs,
            actor_obs_groups=self.policy.obs_groups["policy"],
            context_length=self.policy.context_length,
            max_episode_length=self.policy.max_episode_length,
            num_layers=self.policy.num_layers,
            actor_obs_normalizer=self.policy.actor_obs_normalizer,
            memory_tokens=self.policy.num_memory_tokens,
            d_model=self.policy.d_model,
            max_policy_lag=self.max_policy_lag,
        )

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Stock :meth:`PPO.act`, except that the eval pool executes the distribution MEAN.

        The eval environments run the very same forward pass as any other one -- their KV ring advances in
        lockstep -- so only the executed action is swapped, off the distribution the policy just left behind
        (no second trunk pass, which is what ``act_inference`` would cost and what would desynchronize the
        cache). The log-prob is recomputed against the executed action so the stored row stays self-consistent;
        the update never sees it (see :class:`EpisodeContextRolloutStorage`).
        """
        actions = super().act(obs)
        if self._eval_mask is None:
            return actions
        actions = torch.where(self._eval_mask.unsqueeze(-1), self.policy.action_mean.detach(), actions)
        self.transition.actions = actions
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(actions).detach()
        return actions

    def update(self) -> dict[str, float]:
        """Stock :meth:`PPO.update`, except that a minibatch is processed as ``grad_accumulation_steps`` parts.

        ``grad_accumulation_steps == 1`` delegates to the frozen ``ppo.py`` update itself, so the default path
        cannot drift from it. With ``G > 1`` the storage hands out ``num_mini_batches * G`` environment chunks
        per epoch and this loop:

        * zeroes the gradients at the first micro-batch of a minibatch, backwards ``loss / G`` on each,
        * averages the KL over the ``G`` micro-batches and adapts the learning rate ONCE, right before the step
          (the stock order: the minibatch's own KL sets the rate its step runs at),
        * reduces across GPUs once, clips once, steps once.

        Every reduction that is not elementwise therefore still spans the whole minibatch: the advantage
        normalization (which reads the minibatch's statistics from the storage, not the micro-batch's), the KL,
        and the gradient itself. The logged losses are averaged over micro-batches, so their expected value is
        the same as at ``G = 1``.
        """
        if self.grad_accumulation_steps == 1:
            return super().update()
        assert not self.is_trial_memory, "Gradient accumulation is not implemented for the trial-memory update."
        assert self.symmetry is None, "Gradient accumulation is not implemented for the symmetry losses."

        num_micro_batches = self.grad_accumulation_steps
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_noise_prior_kl = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_ratio = 0.0
        mean_ratio_std = 0.0
        mean_ratio_clip_frac = 0.0
        ratio_min = float("inf")
        ratio_max = float("-inf")

        # Advantage statistics of the LOGICAL minibatch. A micro-batch is a slice of it, so normalizing on the
        # slice would apply a different affine map to each part of one gradient. Advantages do not move during
        # the update, so this is computed once.
        chunks = self.storage.mini_batch_env_chunks(self.num_mini_batches)
        advantage_stats = None
        if self.normalize_advantage_per_mini_batch:
            with torch.no_grad():
                advantage_stats = [
                    (advantages.mean(), advantages.std())
                    for advantages in (self.storage.advantages[:, start:stop] for start, stop in chunks)
                ]

        generator = self.storage.recurrent_mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs, num_micro_batches
        )

        kl_sum = torch.zeros((), device=self.device)
        for index, (
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
        ) in enumerate(generator):
            micro_step = index % num_micro_batches
            mini_batch_index = (index // num_micro_batches) % len(chunks)
            if micro_step == 0:
                self.optimizer.zero_grad()
                if self.rnd:
                    self.rnd_optimizer.zero_grad()
                kl_sum = torch.zeros((), device=self.device)

            original_batch_size = obs_batch.batch_size[0]

            if advantage_stats is not None:
                mean_advantage, std_advantage = advantage_stats[mini_batch_index]
                advantages_batch = (advantages_batch - mean_advantage) / (std_advantage + 1e-8)

            # Recompute actions log prob and entropy for current batch of transitions
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL of this micro-batch; the learning rate is adapted from their mean, at the step below.
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_sum += torch.mean(kl)

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
                ratio_min = min(ratio_min, flat_ratio.min().item())
                ratio_max = max(ratio_max, flat_ratio.max().item())

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

            # Gaussian-prior penalty on the noise dims (RFS)
            if self.noise_prior_kl_coef > 0.0 and self.noise_prior_dims > 0:
                noise_prior_kl = self._noise_prior_kl(mu_batch, sigma_batch, masks_batch)
                loss = loss + self.noise_prior_kl_coef * noise_prior_kl
                mean_noise_prior_kl += noise_prior_kl.item()

            # RND loss
            if self.rnd:
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                rnd_loss = torch.nn.MSELoss()(predicted_embedding, target_embedding)

            # Each micro-batch contributes 1/G of the minibatch gradient (equal micro-batch sizes, see the
            # storage's divisibility assert).
            (loss / num_micro_batches).backward()
            if self.rnd:
                (rnd_loss / num_micro_batches).backward()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()

            if micro_step < num_micro_batches - 1:
                continue

            # -- one optimizer step per logical minibatch --
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_mean = kl_sum / num_micro_batches

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Collect gradients from all GPUs (once, on the accumulated gradient)
            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

        # Divide the losses by the number of micro-batches seen
        num_updates = self.num_learning_epochs * self.num_mini_batches * num_micro_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_noise_prior_kl /= num_updates
        mean_ratio /= num_updates
        mean_ratio_std /= num_updates
        mean_ratio_clip_frac /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates

        # Clear the storage
        self.storage.clear()
        self.storage.policy_version += 1

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
        if self.noise_prior_kl_coef > 0.0:
            loss_dict["noise_prior_kl"] = mean_noise_prior_kl
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        return loss_dict

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """One collection step. With a memory, also the whole cross-episode bookkeeping.

        The body below the memory branch is the stock :meth:`PPO.process_env_step` (``ppo.py`` is byte-frozen, so
        it is replicated rather than called: its last statement, ``policy.reset(dones)``, is exactly the one
        thing that has to change). The order matters:

        1. ``episode_index_in_trial`` is read BEFORE anything resets -- it labels the row about to be stored with
           the episode of the trial that acted it,
        2. the done environments get their ``H`` snapshotted into the storage (the ``M`` memory rows the episode
           ran with, then its readouts), together with whether the episode that just ended was the last of its
           trial. This MUST happen before ``policy.reset``, which consumes the very same ``H`` for its own write
           and then clears it,
        3. the transition is stored, and only then is the policy reset -- with the trial signal, so that ``Z``
           survives an episode boundary inside a trial and is restored to ``z_init`` at a trial boundary.
        """
        if self.policy.num_memory_tokens == 0:
            super().process_env_step(obs, rewards, dones, extras)
            return

        # Trial boundary. Defaults to the episode done, i.e. one episode per trial -- the safe fallback for an
        # environment that does not publish the signal (the memory then never survives a done).
        trial_dones = extras.get("trial_done", dones) if extras is not None else dones
        trial_dones = trial_dones.reshape(-1).to(self.device)

        # -- memory bookkeeping, before any reset --
        self.storage.stage_episode_index(self.policy.episode_index_in_trial)
        done_ids = dones.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            hidden, valid = self.policy.get_episode_hidden(done_ids)
            self.storage.push_episode_hidden(done_ids, hidden, valid, trial_dones[done_ids])

        # -- stock PPO.process_env_step from here on --
        if not self.defer_obs_normalization:
            self.commit_obs_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.raw_rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.trial_dones = trial_dones
        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards
        # Bootstrapping on time outs, at a TRIAL end only: inside a trial GAE keeps bootstrapping through the
        # episode boundary itself, so adding the value here would count it twice.
        if extras is not None and "time_outs" in extras:
            time_outs = extras["time_outs"].unsqueeze(1).to(self.device).float()
            trial_ends = trial_dones.view(-1, 1).float()
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * time_outs * trial_ends, 1)
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones, trial_dones=trial_dones)
