# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO for the single-episode context transformer.

This is the **stock** :class:`~rsl_rl.algorithms.ppo.PPO` -- ``compute_returns`` and the whole ``update()``
are inherited byte-for-byte from the known-good implementation. There are three overrides:

* :meth:`init_storage`, which builds an :class:`EpisodeContextRolloutStorage` instead of a plain
  :class:`RolloutStorage`; that storage's ``recurrent_mini_batch_generator`` then feeds PPO's existing recurrent
  branch (``policy.is_recurrent`` is ``True``) with ``[W, B, ...]`` minibatches whose prefix rides in the
  hidden-state slot,
* :meth:`act`, and ONLY when ``eval_env_fraction > 0``: the last envs execute the distribution mean instead
  of a sample (a deterministic eval pool, excluded from the update by the storage),
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
