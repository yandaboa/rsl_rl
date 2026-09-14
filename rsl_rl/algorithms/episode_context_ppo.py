# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO for the single-episode context transformer (:class:`~rsl_rl.models.EpisodeContextModel`).

The joint actor-critic does not decompose into two independent models (one trunk, one KV cache serves both
``act`` and ``evaluate``), so the algorithm owns the joint module and exposes it to the runner through the thin
:class:`~rsl_rl.models.EpisodeContextActorView` / :class:`~rsl_rl.models.EpisodeContextCriticView` adapters.
Every stage -- act, process_env_step, compute_returns, update, save/load -- is overridden against the joint
module; the PPO math is the rsl-rl 3.1 ``stable-ppo`` one (ratio, KL schedule, clipping, diagnostics).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import warnings
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models.episode_context_model import (
    EpisodeContextActorView,
    EpisodeContextCriticView,
    EpisodeContextModel,
    EpisodeContextPrefix,
)
from rsl_rl.storage.episode_context_storage import EpisodeContextRolloutStorage
from rsl_rl.utils import pad_state_dict_to_model, resolve_callable, resolve_obs_groups


class EpisodeContextPPO(PPO):
    """PPO whose rollout storage keeps the per-environment frame ring the context transformer re-infers from."""

    model: EpisodeContextModel
    storage: EpisodeContextRolloutStorage

    def __init__(
        self,
        model: EpisodeContextModel,
        storage: EpisodeContextRolloutStorage,
        *,
        defer_obs_normalization: bool = True,
        noise_prior_kl_coef: float = 0.0,
        noise_prior_dims: int = 0,
        next_state_coef: float = 0.0,
        eval_env_fraction: float = 0.0,
        grad_accumulation_steps: int = 1,
        **kwargs,
    ) -> None:
        """:class:`PPO`'s keyword arguments plus:

        ``defer_obs_normalization`` (default ``True``): commit the observation-normalizer statistics once, after
        the update, so a rollout is acted and reconstructed under identical statistics (else the epoch-0 ratio
        is off by construction). ``noise_prior_kl_coef`` / ``noise_prior_dims``: ``coef * KL(pi || N(0, I))`` on
        the leading action dims. ``next_state_coef``: auxiliary next-state MSE on the trunk readout (needs a model
        built with ``next_state_head_hidden_dims``). ``eval_env_fraction``: trailing deterministic eval pool,
        excluded from the update (the storage must have been built with the matching ``num_eval_envs``; see
        :meth:`build_storage`). ``grad_accumulation_steps``: micro-batches per minibatch, one optimizer step per
        logical minibatch (same transitions per step, ``1/G`` the peak activation memory).
        """
        if not isinstance(model, EpisodeContextModel):
            raise ValueError(f"EpisodeContextPPO requires an EpisodeContextModel; got {type(model).__name__}.")
        if not isinstance(storage, EpisodeContextRolloutStorage):
            raise ValueError(
                f"EpisodeContextPPO requires an EpisodeContextRolloutStorage; got {type(storage).__name__}."
            )
        if not defer_obs_normalization and getattr(model, "actor_obs_normalization", False):
            warnings.warn(
                "EpisodeContextPPO with defer_obs_normalization=False: the rollout is acted and reconstructed"
                " under different observation-normalizer statistics, so the epoch-0 PPO ratio will not be 1.",
                stacklevel=2,
            )
        self.model = model
        super().__init__(EpisodeContextActorView(model), EpisodeContextCriticView(model), storage, **kwargs)
        assert self.symmetry is None, "Symmetry augmentation is not supported for the episode-context policy."
        self.defer_obs_normalization = bool(defer_obs_normalization)

        assert noise_prior_kl_coef >= 0.0, "noise_prior_kl_coef must be non-negative."
        assert 0 <= noise_prior_dims <= model.num_actions, (
            f"noise_prior_dims={noise_prior_dims} does not fit the policy's {model.num_actions}-d action."
        )
        self.noise_prior_kl_coef = float(noise_prior_kl_coef)
        self.noise_prior_dims = int(noise_prior_dims)
        assert next_state_coef >= 0.0, "next_state_coef must be non-negative."
        assert next_state_coef == 0.0 or model.next_state_enabled, (
            "next_state_coef > 0 needs a policy built with next_state_head_hidden_dims."
        )
        self.next_state_coef = float(next_state_coef)
        assert grad_accumulation_steps >= 1, f"grad_accumulation_steps must be >= 1, got {grad_accumulation_steps}."
        self.grad_accumulation_steps = int(grad_accumulation_steps)

        assert 0.0 <= eval_env_fraction < 1.0, f"eval_env_fraction must be in [0, 1), got {eval_env_fraction}."
        self.eval_env_fraction = float(eval_env_fraction)
        num_envs = storage.num_envs
        num_eval_envs = self._num_eval_envs(eval_env_fraction, num_envs)
        assert storage.num_eval_envs == num_eval_envs, (
            f"the storage was built with num_eval_envs={storage.num_eval_envs}, but eval_env_fraction="
            f"{eval_env_fraction} x {num_envs} envs = {num_eval_envs}. Build it with EpisodeContextPPO.build_storage."
        )
        if num_eval_envs > 0:
            self.eval_env_ids = torch.arange(num_envs, device=self.device)[-num_eval_envs:]
            self._eval_mask: torch.Tensor | None = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
            self._eval_mask[self.eval_env_ids] = True
        else:
            self.eval_env_ids = torch.zeros(0, dtype=torch.long, device=self.device)
            # ``None`` (not an all-False mask): act() then takes the untouched stock path.
            self._eval_mask = None

        self.storage.with_next_state = self.next_state_coef > 0.0
        target_group = model.next_state_target_group
        if self.storage.with_next_state and target_group is not None:
            groups = list(self.storage.observations.keys())
            assert target_group in groups, (
                f"next_state_target_group='{target_group}' is not an observation group ({groups})."
            )
            width = int(self.storage.observations[target_group].shape[-1])
            assert width == model.next_state_target_dim, (
                f"next_state_target_dim={model.next_state_target_dim} != the env's '{target_group}' width {width}."
            )
            self.storage.enable_next_state_target(target_group, model.next_state_target_dim)

        self.intrinsic_rewards: torch.Tensor | None = None

    # --------------------------------------------------------------------------------------------------------
    # Construction
    # --------------------------------------------------------------------------------------------------------

    @staticmethod
    def _num_eval_envs(eval_env_fraction: float, num_envs: int) -> int:
        num_eval_envs = round(float(eval_env_fraction) * num_envs)
        assert num_eval_envs < num_envs, (
            f"eval_env_fraction={eval_env_fraction} leaves no training environment ({num_eval_envs} of {num_envs})."
        )
        return num_eval_envs

    @staticmethod
    def build_storage(
        model: EpisodeContextModel,
        num_envs: int,
        num_steps_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
        eval_env_fraction: float = 0.0,
    ) -> EpisodeContextRolloutStorage:
        """The storage sized for ``model`` (ring, prefix, memory snapshots, eval pool)."""
        return EpisodeContextRolloutStorage(
            "rl",
            num_envs,
            num_steps_per_env,
            obs,
            actions_shape,
            device,
            num_eval_envs=EpisodeContextPPO._num_eval_envs(eval_env_fraction, num_envs),
            actor_obs_groups=model.obs_groups["actor"],
            context_length=model.context_length,
            max_episode_length=model.max_episode_length,
            num_layers=model.num_layers,
            actor_obs_normalizer=model.frame_normalizer,
            memory_tokens=model.num_memory_tokens,
            d_model=model.d_model,
        )

    @classmethod
    def create(
        cls,
        model: EpisodeContextModel,
        num_envs: int,
        num_steps_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        **kwargs,
    ) -> EpisodeContextPPO:
        """Build the storage for ``model`` and the algorithm on top of it (``kwargs`` are the constructor's)."""
        storage = cls.build_storage(
            model,
            num_envs,
            num_steps_per_env,
            obs,
            actions_shape,
            kwargs.get("device", "cpu"),
            kwargs.get("eval_env_fraction", 0.0),
        )
        return cls(model, storage, **kwargs)

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> EpisodeContextPPO:
        """Runner entry point. The joint model's kwargs come from ``cfg["actor"]``; ``cfg["critic"]`` is optional
        and merged (``hidden_dims`` -> ``critic_hidden_dims``, ``obs_normalization`` -> ``critic_obs_normalization``,
        every other key passed through)."""
        alg_class: type[EpisodeContextPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        model_cfg = dict(cfg["actor"])
        model_class: type[EpisodeContextModel] = resolve_callable(model_cfg.pop("class_name"))  # type: ignore
        for key, target in (("hidden_dims", "actor_hidden_dims"), ("obs_normalization", "actor_obs_normalization")):
            if key in model_cfg:
                model_cfg[target] = model_cfg.pop(key)
        distribution_cfg = model_cfg.pop("distribution_cfg", None)
        if distribution_cfg:
            # 5.2 ``distribution_cfg`` -> the joint model's own noise parameterization.
            dist_class = str(distribution_cfg.get("class_name", "GaussianDistribution"))
            if "Gsde" in dist_class:
                model_cfg["noise_std_type"] = "gsde"
            elif "Gaussian" in dist_class:
                model_cfg["noise_std_type"] = distribution_cfg.get("std_type", "scalar")
            else:
                raise ValueError(f"EpisodeContextModel supports Gaussian / gSDE noise only, got {dist_class!r}.")
            if "init_std" in distribution_cfg:
                model_cfg["init_noise_std"] = distribution_cfg["init_std"]
        critic_cfg = dict(cfg.get("critic") or {})
        critic_cfg.pop("class_name", None)
        critic_cfg.pop("distribution_cfg", None)
        for key, value in critic_cfg.items():
            renamed = {"hidden_dims": "critic_hidden_dims", "obs_normalization": "critic_obs_normalization"}
            model_cfg[renamed.get(key, key)] = value

        default_sets = ["actor", "critic"]
        if cfg["algorithm"].get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)
        cfg["algorithm"].pop("share_cnn_encoders", None)  # RslRlPpoAlgorithmCfg default; meaningless here

        model = model_class(obs, cfg["obs_groups"], env.num_actions, **model_cfg).to(device)
        print(f"Episode-context model: {model}")
        storage = alg_class.build_storage(
            model,
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
            cfg["algorithm"].get("eval_env_fraction", 0.0),
        )
        alg = alg_class(model, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])
        alg.compile(cfg.get("torch_compile_mode"))
        return alg

    def compile(self, mode: str | None = None) -> None:
        """No-op: the KV-cached transformer is not ``torch.compile``d. ``actor``/``critic`` stay the raw views."""
        if mode is not None:
            print(f"[EpisodeContextPPO] torch_compile_mode={mode!r} ignored: the episode-context model is not compiled")

    # --------------------------------------------------------------------------------------------------------
    # Collection
    # --------------------------------------------------------------------------------------------------------

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Stock :meth:`PPO.act` through the views; the eval pool executes the distribution MEAN.

        The eval envs run the same forward pass (their KV ring advances in lockstep); only the executed action is
        swapped, off the distribution the model just left behind. The log-prob is recomputed against it.
        """
        actions = super().act(obs)
        if self._eval_mask is None:
            return actions
        actions = torch.where(self._eval_mask.unsqueeze(-1), self.model.action_mean.detach(), actions)
        self.transition.actions = actions
        self.transition.actions_log_prob = self.model.get_actions_log_prob(actions).detach()
        return actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """One collection step. With a memory, the cross-episode bookkeeping happens BEFORE the model reset:
        the row is labeled with its episode index, the done envs' ``H`` is snapshotted into the storage, and
        the reset is given the trial signal so ``Z`` survives an episode boundary inside a trial."""
        has_memory = self.model.num_memory_tokens > 0
        trial_dones = extras.get("trial_done", dones) if extras is not None else dones
        trial_dones = trial_dones.reshape(-1).to(self.device)

        if has_memory:
            self.storage.stage_episode_index(self.model.episode_index_in_trial)
            done_ids = dones.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                hidden, valid = self.model.get_episode_hidden(done_ids)
                self.storage.push_episode_hidden(done_ids, hidden, valid, trial_dones[done_ids])

        if not self.defer_obs_normalization:
            self.commit_obs_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.trial_dones = trial_dones  # type: ignore[attr-defined]
        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards
        # Bootstrap on time-outs at a TRIAL end only: inside a trial GAE bootstraps through the episode boundary.
        if extras is not None and "time_outs" in extras:
            time_outs = extras["time_outs"].unsqueeze(1).to(self.device).float()
            trial_ends = trial_dones.view(-1, 1).float()
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * time_outs * trial_ends, 1)
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.model.reset(dones, trial_dones=trial_dones if has_memory else None)

    def commit_obs_normalization(self, obs_batch: TensorDict) -> None:
        """Update the observation normalizers from ``[N, ...]`` or ``[T, N, ...]`` observations."""
        batch_dims = getattr(obs_batch, "batch_dims", 1)
        if batch_dims > 1:
            obs_batch = obs_batch.flatten(0, batch_dims - 1)
        self.model.update_normalization(obs_batch)
        if self.rnd:
            self.rnd.update_normalization(obs_batch)

    def compute_returns(self, obs: TensorDict) -> None:
        """Stock GAE keyed on the TRIAL boundary (== the episode boundary without a trial signal); the advantage
        normalization statistics come from the TRAINING pool only."""
        st = self.storage
        last_values = self.model.evaluate(obs).detach()  # peeks: the terminal frame must not enter the KV cache
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.trial_dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            train_advantages = st.advantages[:, : st.num_train_envs]
            st.advantages = (st.advantages - train_advantages.mean()) / (train_advantages.std() + 1e-8)

    # --------------------------------------------------------------------------------------------------------
    # Update
    # --------------------------------------------------------------------------------------------------------

    def _noise_prior_kl(self, mu: torch.Tensor, sigma: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Mean KL(N(mu, sigma^2) || N(0, I)) over the first ``noise_prior_dims`` action dims."""
        mu = mu[..., : self.noise_prior_dims]
        sigma = sigma[..., : self.noise_prior_dims]
        kl = 0.5 * (mu.pow(2) + sigma.pow(2) - 1.0 - 2.0 * torch.log(sigma.clamp_min(1e-8))).sum(-1)
        if mask is not None:
            kl = kl[mask]
        return kl.mean()

    def _next_state_loss(
        self, actions: torch.Tensor, prefix: EpisodeContextPrefix | None
    ) -> tuple[torch.Tensor, float, float, float]:
        """Masked MSE of the aux head against the stored deltas: ``(total, obs part, target part, valid frac)``.
        Reuses the ``h`` the surrogate's ``act()`` just produced; the trunk is NOT re-run."""
        assert prefix is not None and prefix.next_delta is not None, (
            "next_state_coef > 0 needs the next-state targets in the hidden-state slot (set"
            " EpisodeContextRolloutStorage.with_next_state)."
        )
        model = self.model
        hidden = model._window_hidden
        assert hidden is not None, "The next-state head reads the window hidden state act() leaves behind."
        pred = model.next_state_from_hidden(hidden, actions)
        target = prefix.next_delta[..., model.next_state_pred_start : model.next_state_pred_end]
        valid = prefix.next_valid
        denominator = valid.sum().clamp_min(1)
        obs_dim = model.next_state_obs_dim
        obs_loss = ((pred[..., :obs_dim] - target).pow(2).mean(-1) * valid).sum() / denominator
        loss, target_value = obs_loss, 0.0
        normalizer = model.next_state_target_normalizer
        if normalizer is not None:
            assert prefix.next_target_delta is not None, (
                f"the policy predicts the '{model.next_state_target_group}' group, but the storage carries no"
                " target ring (call EpisodeContextRolloutStorage.enable_next_state_target)."
            )
            with torch.no_grad():
                rows = prefix.next_target_delta[valid]
                if rows.numel() > 0:
                    normalizer.update(rows)
                target_delta = normalizer(prefix.next_target_delta)
            target_loss = ((pred[..., obs_dim:] - target_delta).pow(2).mean(-1) * valid).sum() / denominator
            loss = loss + target_loss
            target_value = target_loss.item()
        return loss, obs_loss.item(), target_value, valid.float().mean().item()

    def _adapt_learning_rate(self, kl_mean: torch.Tensor) -> None:
        """The stock adaptive schedule (rank 0 decides, everybody follows)."""
        if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
        if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    @staticmethod
    def _grad_norm(params) -> torch.Tensor | None:
        total = None
        for p in params:
            if p.grad is None:
                continue
            sq = p.grad.detach().pow(2).sum()
            total = sq if total is None else total + sq
        return None if total is None else total.sqrt()

    def update(self) -> dict[str, float]:
        """One PPO update over the rollout: ``num_learning_epochs x num_mini_batches`` optimizer steps, each
        minibatch processed as ``grad_accumulation_steps`` micro-batches (``loss / G`` each, one step). The KL of
        a minibatch is the mean over its micro-batches, applied to the learning rate right before its step."""
        num_micro_batches = self.grad_accumulation_steps
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_noise_prior_kl = 0.0
        mean_next_state = 0.0
        mean_next_state_obs = 0.0
        mean_next_state_target = 0.0
        mean_next_state_valid = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_ratio = 0.0
        mean_ratio_std = 0.0
        mean_ratio_clip_frac = 0.0
        ratio_min = float("inf")
        ratio_max = float("-inf")
        grad_norms: dict[str, list[torch.Tensor]] = {"actor": [], "critic": [], "combined": []}

        # Advantage statistics of the LOGICAL minibatch: a micro-batch is a slice of it.
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
        for index, batch in enumerate(generator):
            micro_step = index % num_micro_batches
            mini_batch_index = (index // num_micro_batches) % len(chunks)
            if micro_step == 0:
                self.optimizer.zero_grad()
                if self.rnd:
                    self.rnd.optimizer.zero_grad()
                kl_sum = torch.zeros((), device=self.device)

            original_batch_size = batch.observations.batch_size[0]
            advantages = batch.advantages
            if advantage_stats is not None:
                mean_advantage, std_advantage = advantage_stats[mini_batch_index]
                advantages = (advantages - mean_advantage) / (std_advantage + 1e-8)

            # Re-infer the window; ``evaluate`` reuses the ``h`` this ``act`` leaves behind (shared trunk).
            self.actor(
                batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            mu, sigma = (p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, (mu, sigma))
                    kl_sum += torch.mean(kl)

            # Surrogate loss (log-ratio clamped at +-20: inactive for healthy Gaussians, arrests exp overflow).
            log_ratio = actions_log_prob - torch.squeeze(batch.old_actions_log_prob)
            ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
            surrogate = -torch.squeeze(advantages) * ratio
            surrogate_clipped = -torch.squeeze(advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            with torch.no_grad():
                flat_ratio = ratio.detach().reshape(-1)
                mean_ratio += flat_ratio.mean().item()
                mean_ratio_std += flat_ratio.std().item()
                mean_ratio_clip_frac += ((flat_ratio - 1.0).abs() > self.clip_param).float().mean().item()
                ratio_min = min(ratio_min, flat_ratio.min().item())
                ratio_max = max(ratio_max, flat_ratio.max().item())

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            if self.noise_prior_kl_coef > 0.0 and self.noise_prior_dims > 0:
                noise_prior_kl = self._noise_prior_kl(mu, sigma, batch.masks)
                loss = loss + self.noise_prior_kl_coef * noise_prior_kl
                mean_noise_prior_kl += noise_prior_kl.item()

            if self.next_state_coef > 0.0:
                next_state_loss, obs_part, target_part, valid_frac = self._next_state_loss(
                    batch.actions, batch.hidden_states[0]
                )
                loss = loss + self.next_state_coef * next_state_loss
                mean_next_state += next_state_loss.item()
                mean_next_state_obs += obs_part
                mean_next_state_target += target_part
                mean_next_state_valid += valid_frac

            rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None

            # Each micro-batch contributes 1/G of the minibatch gradient (equal micro-batch sizes).
            (loss / num_micro_batches).backward()
            if self.rnd:
                (rnd_loss / num_micro_batches).backward()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()

            if micro_step < num_micro_batches - 1:
                continue

            # -- one optimizer step per logical minibatch --
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    self._adapt_learning_rate(kl_sum / num_micro_batches)
            if self.is_multi_gpu:
                self.reduce_parameters()
            with torch.no_grad():
                actor_norm = self._grad_norm(self.actor.parameters())
                critic_norm = self._grad_norm(self.critic.parameters())
                combined_norm = self._grad_norm(self.model.parameters())
                for key, value in (("actor", actor_norm), ("critic", critic_norm), ("combined", combined_norm)):
                    if value is not None:
                        grad_norms[key].append(value)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd:
                self.rnd.optimizer.step()

        num_updates = self.num_learning_epochs * self.num_mini_batches * num_micro_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_noise_prior_kl /= num_updates
        mean_next_state /= num_updates
        mean_next_state_obs /= num_updates
        mean_next_state_target /= num_updates
        mean_next_state_valid /= num_updates
        mean_ratio /= num_updates
        mean_ratio_std /= num_updates
        mean_ratio_clip_frac /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates

        self.storage.clear()
        # Deferred normalizer commit: after the update, so acting and reconstruction shared one statistic.
        if self.defer_obs_normalization:
            self.commit_obs_normalization(self.storage.collected_observations)

        loss_dict = {
            "value": mean_value_loss,
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
        if self.next_state_coef > 0.0:
            loss_dict["next_state"] = mean_next_state
            loss_dict["next_state_valid_frac"] = mean_next_state_valid
            if self.model.next_state_target_normalizer is not None:
                loss_dict["next_state_obs"] = mean_next_state_obs
                loss_dict["next_state_obj"] = mean_next_state_target
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        for component, values in grad_norms.items():
            if not values:
                continue
            stacked = torch.stack(values)
            loss_dict[f"grad_norm/{component}_mean"] = stacked.mean().item()
            loss_dict[f"grad_norm/{component}_var"] = stacked.var(unbiased=False).item() if len(values) > 1 else 0.0
        return loss_dict

    # --------------------------------------------------------------------------------------------------------
    # Modes / checkpoints / distributed
    # --------------------------------------------------------------------------------------------------------

    def train_mode(self) -> None:
        self.model.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        self.model.eval()
        if self.rnd:
            self.rnd.eval()

    def get_policy(self) -> EpisodeContextActorView:
        return self._raw_actor  # type: ignore[return-value]

    def save(self) -> dict:
        """``actor_state_dict`` holds the WHOLE joint model; ``critic_state_dict`` aliases its ``critic*`` keys."""
        state = self.model.state_dict()
        saved_dict = {
            "actor_state_dict": state,
            "critic_state_dict": {key: value for key, value in state.items() if key.startswith("critic")},
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd.optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load the joint model from a 5.2 (``actor_state_dict``) or 3.1 (``model_state_dict``) checkpoint.

        ``strict=False`` zero-pads / truncates shape-mismatched tensors and skips the optimizer (its moment
        shapes would not match). ``load_cfg`` selects ``actor`` / ``critic`` by key prefix.
        """
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True, "rnd": True}

        if "model_state_dict" in loaded_dict and "actor_state_dict" not in loaded_dict:
            state = dict(loaded_dict["model_state_dict"])
            print(
                "[EpisodeContextPPO.load] legacy checkpoint: mapping 'model_state_dict' (rsl-rl 3.1"
                f" ActorCriticEpisodeContext, {len(state)} tensors) onto the joint EpisodeContextModel."
            )
        else:
            state = dict(loaded_dict["actor_state_dict"])
            state.update(loaded_dict.get("critic_state_dict", {}))

        load_actor, load_critic = bool(load_cfg.get("actor")), bool(load_cfg.get("critic"))
        if load_actor or load_critic:
            if not (load_actor and load_critic):
                keep_critic = load_critic and not load_actor
                state = {key: value for key, value in state.items() if key.startswith("critic") == keep_critic}
            partial = not (load_actor and load_critic)
            if not strict:
                state = pad_state_dict_to_model(self.model, state, log_prefix="[EpisodeContextPPO.load]")
            self.model.load_state_dict(state, strict=strict and not partial)

        if load_cfg.get("optimizer"):
            if strict:
                self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            else:
                print("[EpisodeContextPPO.load] strict=False: optimizer state not restored.")
        if load_cfg.get("rnd") and self.rnd and "rnd_state_dict" in loaded_dict:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd.optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        return bool(load_cfg.get("iteration", False))

    def broadcast_parameters(self) -> None:
        modules = [self.model]
        if self.rnd:
            modules.append(self.rnd.predictor)
        for module in modules:
            for tensor in chain(module.parameters(), module.buffers()):
                torch.distributed.broadcast(tensor.data, src=0)

    def reduce_parameters(self) -> None:
        all_params = list(chain(self.model.parameters(), self.rnd.parameters() if self.rnd else []))
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
