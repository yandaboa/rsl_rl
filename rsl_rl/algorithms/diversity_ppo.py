# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, SkillDiscriminator


class DiversityPPO(PPO):
    """PPO with a DIAYN-style diversity bonus driven by an env-side reward term.

    The algorithm owns the discriminator ``q(z | s)`` and its optimiser. The discriminator is
    *exposed* on the env (``env.diversity_discriminator``) so a reward term inside the env can
    query it each step. PPO consumes the resulting reward as usual; this class only adds the
    discriminator update step on top of the standard PPO update loop.
    """

    policy: ActorCritic | ActorCriticRecurrent

    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent,
        # Standard PPO args (forwarded to the parent — declared explicitly so that
        # ``sanitize_rsl_rl_cfg`` doesn't strip them based on signature inspection).
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 1.0e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        device: str = "cpu",
        normalize_advantage_per_mini_batch: bool = False,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        weight_decay: float = 0.0,
        # Diversity-specific
        discriminator_cfg: dict | None = None,
        number_of_skills: int = 10,
        skill_obs_key: str = "skill",
        **kwargs,
    ) -> None:
        if kwargs:
            print("Extra kwargs ignored by DiversityPPO:", list(kwargs.keys()))
        super().__init__(
            policy=policy,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
            weight_decay=weight_decay,
        )

        if discriminator_cfg is None:
            raise ValueError("DiversityPPO requires `discriminator_cfg`.")

        self.num_skills = int(number_of_skills)
        self.skill_obs_key = skill_obs_key
        self.disc_cfg = dict(discriminator_cfg)

        # Discriminator + optimiser are constructed lazily in init_storage(), once we know the
        # input dimension from the obs TensorDict.
        self.discriminator: SkillDiscriminator | None = None
        self.discriminator_optimizer: optim.Optimizer | None = None
        self._disc_input_dim: int | None = None

        # Cached reward bonus for logging — populated each env step.
        self.intrinsic_rewards: torch.Tensor | None = None

        # Counter to honour `update_frequency`: skip discriminator update on PPO steps that
        # don't land on a multiple of the configured frequency.
        self._update_step = 0

    # ----------------------------------------------------------------------
    # Construction helpers
    # ----------------------------------------------------------------------

    def _resolve_disc_obs_dim(self, obs: TensorDict) -> int:
        group = self.disc_cfg["obs_group"]
        if group not in obs.keys():
            raise KeyError(
                f"DiversityPPO: obs group '{group}' not found in env observations. "
                f"Available groups: {list(obs.keys())}."
            )
        sample = obs[group]
        if isinstance(sample, TensorDict):
            return sum(sample[k].shape[-1] for k in sample.keys())
        return int(sample.shape[-1])

    def _build_discriminator(self, obs: TensorDict) -> None:
        self._disc_input_dim = self._resolve_disc_obs_dim(obs)
        self.discriminator = SkillDiscriminator(
            input_dim=self._disc_input_dim,
            num_skills=self.num_skills,
            hidden_dims=list(self.disc_cfg["hidden_dims"]),
            activation=self.disc_cfg.get("activation", "elu"),
            obs_normalization=self.disc_cfg.get("obs_normalization", True),
        ).to(self.device)
        self.discriminator_optimizer = optim.AdamW(
            self.discriminator.parameters(),
            lr=float(self.disc_cfg["learning_rate"]),
            weight_decay=float(self.disc_cfg.get("weight_decay", 0.0)),
        )

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        super().init_storage(training_type, num_envs, num_transitions_per_env, obs, actions_shape)
        if self.discriminator is None:
            self._build_discriminator(obs)

    # ----------------------------------------------------------------------
    # Env wiring
    # ----------------------------------------------------------------------

    def attach_env(self, env) -> None:
        """Publish the discriminator on the env so the diversity reward term can call it."""
        unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
        unwrapped.diversity_discriminator = self.discriminator
        unwrapped.diversity_num_skills = self.num_skills
        unwrapped.diversity_reward_scale = float(self.disc_cfg.get("reward_scale", 1.0))
        unwrapped.diversity_use_log_prior = bool(self.disc_cfg.get("use_log_prior", True))

    # ----------------------------------------------------------------------
    # Discriminator helpers
    # ----------------------------------------------------------------------

    def _gather_disc_obs(self, obs_td: TensorDict) -> torch.Tensor:
        """Return s_{disc} as a flat tensor from the storage obs TensorDict slice."""
        group = self.disc_cfg["obs_group"]
        sample = obs_td[group]
        if isinstance(sample, TensorDict):
            return torch.cat([sample[k] for k in sample.keys()], dim=-1)
        return sample

    def _gather_skill_labels(self, policy_obs_td: TensorDict) -> torch.Tensor:
        """Argmax the one-hot skill obs into integer class labels."""
        skill = policy_obs_td[self.skill_obs_key]
        return skill.argmax(dim=-1).long()

    # ----------------------------------------------------------------------
    # Update
    # ----------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        # Standard PPO pass — populates self.storage, runs PPO epochs, clears storage at the end.
        # We need the rollout *before* it gets cleared, so peek at it before delegating.
        disc_metrics = self._update_discriminator_from_storage()
        loss_dict = super().update()

        if disc_metrics is not None:
            loss_dict.update(disc_metrics)
        return loss_dict

    def _update_discriminator_from_storage(self) -> dict[str, float] | None:
        self._update_step += 1
        freq = max(1, int(self.disc_cfg.get("update_frequency", 1)))
        if self._update_step % freq != 0:
            return None
        if self.discriminator is None or self.storage is None:
            return None

        # Pull s_{t+1} and z_{t+1} for every transition. ``policy_observations`` and
        # ``observations`` are stored at step t (i.e. the obs *fed to* the policy at step t,
        # which is s_t). For DIAYN we want s_{t+1}; we approximate it by shifting one step in
        # time and dropping the final transition (which has no recorded s_{t+1}).
        T = self.storage.num_transitions_per_env
        if T < 2:
            return None

        disc_obs = self._gather_disc_obs(self.storage.observations)  # [T, N, D]
        labels = self._gather_skill_labels(self.storage.policy_observations)  # [T, N]
        # Pair (s_t -> s_{t+1}): feed s_{t+1}, label z_{t+1}.
        s_next = disc_obs[1:].reshape(-1, disc_obs.shape[-1])
        z_next = labels[1:].reshape(-1)

        # We mask out two kinds of pairs:
        #  - step t was terminal -> s_{t+1} is a fresh reset state the policy didn't drive,
        #  - the post-success latch was already True at step t -> the policy was free to do
        #    whatever after the task was solved, so those transitions don't reflect skill-
        #    conditioned behaviour.
        # We *do not* drop them — under DDP every rank must call the same number of
        # all-reduces in lockstep, and filtering would give each rank a different batch size
        # and minibatch count. Instead we compute a per-sample weight and use a weighted CE
        # loss; masked samples contribute zero gradient.
        prev_dones = self.storage.dones[:-1].reshape(-1).bool()
        if (
            "diversity_meta" in self.storage.observations.keys()
            and self.storage.observations["diversity_meta"].shape[-1] >= 1
        ):
            prev_latch = (
                self.storage.observations["diversity_meta"][:-1, ..., 0].reshape(-1).bool()
            )
        else:
            prev_latch = torch.zeros_like(prev_dones)
        sample_weight = ((~prev_dones) & (~prev_latch)).float()
        # Note: do NOT early-return when sample_weight is all-zero on this rank; the other
        # ranks will keep all-reducing and the comm will deadlock. The weighted loss handles
        # the all-zero case naturally (grads = 0).

        cfg = self.disc_cfg
        num_epochs = max(1, int(cfg.get("num_learning_epochs", 4)))
        num_minibatches = max(1, int(cfg.get("num_mini_batches", 4)))
        max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
        label_smoothing = float(cfg.get("label_smoothing", 0.0))

        # Update the discriminator's input normalizer with one full pass over the batch.
        # Only feed in unmasked samples so the running mean/var reflects the policy-driven
        # state distribution (skipping reset / post-success states).
        with torch.no_grad():
            unmasked = sample_weight.bool()
            if unmasked.any():
                self.discriminator.update_normalization(s_next[unmasked])

        # Crucially: every rank runs exactly the same number of inner steps and all-reduces,
        # regardless of how many samples are masked out on each rank — otherwise NCCL
        # all-reduce deadlocks.
        batch_size = s_next.shape[0]
        mb_size = max(1, batch_size // num_minibatches)
        n_mb = max(1, batch_size // mb_size)

        total_loss = 0.0
        total_acc = 0.0
        total_weight = 0.0
        for _ in range(num_epochs):
            perm = torch.randperm(batch_size, device=self.device)
            for i in range(n_mb):
                idx = perm[i * mb_size : (i + 1) * mb_size]
                logits = self.discriminator(s_next[idx])
                per_sample_loss = F.cross_entropy(
                    logits, z_next[idx], label_smoothing=label_smoothing, reduction="none"
                )
                w = sample_weight[idx]
                # Weighted CE: gradient direction is the unweighted CE on retained samples;
                # masked samples (w == 0) contribute zero loss and zero grad.
                denom = w.sum().clamp_min(1.0)
                loss = (per_sample_loss * w).sum() / denom

                self.discriminator_optimizer.zero_grad()
                loss.backward()
                if self.is_multi_gpu:
                    self._reduce_discriminator_gradients()
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_grad_norm)
                self.discriminator_optimizer.step()

                with torch.no_grad():
                    correct = (logits.argmax(dim=-1) == z_next[idx]).float()
                    acc_w = (correct * w).sum().item()
                    weight_sum = w.sum().item()
                total_loss += loss.item() * weight_sum
                total_acc += acc_w
                total_weight += weight_sum

        if total_weight == 0:
            return {
                "discriminator_loss": 0.0,
                "discriminator_accuracy": 0.0,
                "discriminator_chance": 1.0 / max(1, self.num_skills),
            }
        return {
            "discriminator_loss": total_loss / total_weight,
            "discriminator_accuracy": total_acc / total_weight,
            "discriminator_chance": 1.0 / max(1, self.num_skills),
        }

    # ----------------------------------------------------------------------
    # Multi-GPU
    # ----------------------------------------------------------------------

    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        if self.discriminator is not None:
            for param in self.discriminator.parameters():
                torch.distributed.broadcast(param.data, src=0)

    def _reduce_discriminator_gradients(self) -> None:
        grads = [p.grad.view(-1) for p in self.discriminator.parameters() if p.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for p in self.discriminator.parameters():
            if p.grad is not None:
                n = p.numel()
                p.grad.data.copy_(all_grads[offset : offset + n].view_as(p.grad.data))
                offset += n
