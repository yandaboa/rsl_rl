# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PolicyFlow (arXiv 2602.01156), ported from the authors' policyflow_torch reference.

The executed action is ``phi(x0) + n``. In the update, the change of the velocity field at a random point of the
act-time interpolation path, ``dv = v_new - v_old``, is the mean of a Gaussian over ``n``; the PPO ratio is
``N(n; dv, std_new) / N(n; 0, std_old)``. A Brownian regularizer keeps the field close to the previous one.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.models.policyflow_model import PolicyFlowModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


def gaussian_kl(mean: torch.Tensor, std: torch.Tensor, old_mean: torch.Tensor, old_std: torch.Tensor) -> torch.Tensor:
    """Reference ``_compute_kl_divergence``: mean over rows of the per-row (summed over dims) Gaussian KL."""
    std_ratio = std / old_std + 1.0e-5
    drift = old_std.square() + (old_mean - mean).square()
    return (std_ratio.log() + drift / (2.0 * std.square()) - 0.5).sum(dim=-1).mean()


class PolicyFlow(PPO):
    """PolicyFlow on a :class:`PolicyFlowModel` actor (critic, storage, eval pool, multi-GPU from PPO)."""

    actor: PolicyFlowModel

    def __init__(
        self,
        actor: PolicyFlowModel,
        critic: MLPModel,
        storage: RolloutStorage,
        brownian_reg_coef: float = 0.002,
        weight_decay: float = 1e-5,
        target_kl: float | None = None,
        adam_betas: tuple[float, float] | list[float] = (0.9, 0.999),
        **kwargs,
    ) -> None:
        assert isinstance(actor, PolicyFlowModel), f"PolicyFlow needs a PolicyFlowModel, got {type(actor).__name__}"
        assert not (actor.is_recurrent or critic.is_recurrent), "PolicyFlow does not support recurrent models"
        assert not kwargs.get("rnd_cfg"), "PolicyFlow does not support RND"
        assert not kwargs.get("symmetry_cfg"), "PolicyFlow does not support symmetry augmentation"
        kwargs.setdefault("schedule", "fixed")
        super().__init__(actor, critic, storage, **kwargs)
        assert self.schedule in ("fixed", "adaptive"), f"unknown schedule {self.schedule!r}"
        if self.schedule == "adaptive":
            assert self.desired_kl is not None and self.desired_kl > 0, "schedule='adaptive' needs desired_kl > 0"

        # AdamW when weight_decay > 0 (reference), over actor + critic together
        params = chain(self.actor.parameters(), self.critic.parameters())
        if weight_decay > 0:
            self.optimizer = optim.AdamW(
                params, lr=self.learning_rate, betas=tuple(adam_betas), weight_decay=weight_decay
            )
        else:
            self.optimizer = optim.Adam(params, lr=self.learning_rate, betas=tuple(adam_betas))

        self.brownian_reg_coef = float(brownian_reg_coef)
        # approx KL = mean((r - 1) - log r) before each step; the step still applies, then the update ends. None = off
        self.target_kl = None if target_kl is None else float(target_kl)

        # Reference ``model_last``: the velocity field of the rollout policy, re-synced at the start of update()
        self.actor_old: PolicyFlowModel = copy.deepcopy(self._raw_actor).requires_grad_(False).eval()

        t_steps, num_envs = storage.num_transitions_per_env, storage.num_envs
        a, dev = actor.num_actions, self.device
        self.pf_latent = torch.zeros(t_steps, num_envs, actor.obs_dim, device=dev)
        self.pf_x0 = torch.zeros(t_steps, num_envs, a, device=dev)
        self.pf_phi = torch.zeros(t_steps, num_envs, a, device=dev)
        self.pf_noise = torch.zeros(t_steps, num_envs, a, device=dev)
        self.pf_old_std = torch.zeros(t_steps, num_envs, a, device=dev)
        self.pf_old_logp = torch.zeros(t_steps, num_envs, device=dev)

    # ------------------------------------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------------------------------------

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample ``phi(x0) + n``, store ``(latent, x0, phi, n, std, log N(n; 0, std))``; eval envs act zero-noise."""
        step = self.storage.step
        num_envs = self.storage.num_envs
        self.transition.hidden_states = (None, None)
        actions = self.actor(obs, stochastic_output=True).detach()
        raw = self._raw_actor
        phi = raw.last_phi
        noise = actions - phi  # exact n for the stored (unclamped) action
        std = raw.std.detach().expand_as(noise)
        old_logp = torch.distributions.Normal(torch.zeros_like(noise), std).log_prob(noise).sum(-1)
        self.pf_latent[step].copy_(raw.last_latent)
        self.pf_x0[step].copy_(raw.last_x0)
        self.pf_phi[step].copy_(phi)
        self.pf_noise[step].copy_(noise)
        self.pf_old_std[step].copy_(std)
        self.pf_old_logp[step].copy_(old_logp)

        if self.eval_env_mask is not None and self.eval_deterministic:
            actions = torch.where(self.eval_env_mask.unsqueeze(-1), raw.output_mean.detach(), actions)
        self.transition.actions = actions
        self.transition.values = self.critic(obs).detach()
        # Storage bookkeeping only (the log-probs live in pf_old_logp)
        self.transition.actions_log_prob = torch.zeros(num_envs, device=self.device)
        self.transition.distribution_params = (torch.zeros(num_envs, 1, device=self.device),)
        self.transition.observations = obs
        return actions

    # ------------------------------------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------------------------------------

    def _mini_batch_generator(self):
        """Flat shuffled mini-batches over the training envs; storage and PolicyFlow buffers share one permutation."""
        st = self.storage
        train = slice(None, st.num_train_envs)
        batch_size = st.num_train_envs * st.num_transitions_per_env
        mini_batch_size = batch_size // self.num_mini_batches
        flat = {
            "observations": st.observations[:, train].flatten(0, 1),
            "values": st.values[:, train].flatten(0, 1),
            "advantages": st.advantages[:, train].flatten(0, 1),
            "returns": st.returns[:, train].flatten(0, 1),
            "latent": self.pf_latent[:, train].flatten(0, 1),
            "x0": self.pf_x0[:, train].flatten(0, 1),
            "phi": self.pf_phi[:, train].flatten(0, 1),
            "noise": self.pf_noise[:, train].flatten(0, 1),
            "old_std": self.pf_old_std[:, train].flatten(0, 1),
            "old_logp": self.pf_old_logp[:, train].flatten(0, 1),
        }
        for _ in range(self.num_learning_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for i in range(self.num_mini_batches):
                idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                yield {k: v[idx] for k, v in flat.items()}

    def update(self) -> dict[str, float]:  # noqa: C901
        stats: dict[str, list[torch.Tensor]] = {
            k: []
            for k in (
                "value",
                "surrogate",
                "entropy",
                "policyflow/brownian",
                "policyflow/kl",
                "policyflow/ratio_mean",
                "policyflow/clip_frac",
                "policyflow/dv_abs_max",
                "policyflow/noise_std_mean",
                "grad_norm/actor",
                "grad_norm/critic",
                "grad_norm/combined",
                "grad_norm/combined_after_clip",
            )
        }
        self.actor_old.load_state_dict(self._raw_actor.state_dict())
        approx_kls: list[float] = []
        stopped = False
        params = list(chain(self.actor.parameters(), self.critic.parameters()))
        grid = self._raw_actor.train_time_grid

        for batch in self._mini_batch_generator():
            advantages = batch["advantages"].squeeze(-1)  # [B]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            latent, x0, phi = batch["latent"], batch["x0"], batch["phi"]
            t = grid[torch.randint(0, grid.shape[0], (latent.shape[0],), device=latent.device)]
            tc = t[:, None]
            x_t = (1.0 - tc) * x0 + tc * phi
            with torch.no_grad():
                v_old = self.actor_old.velocity(latent, x_t, t)
            v = self.actor.velocity(latent, x_t, t)
            dv = v - v_old
            std = self._raw_actor.std.expand_as(dv)
            dist = torch.distributions.Normal(dv, std)
            log_ratio = dist.log_prob(batch["noise"]).sum(-1) - batch["old_logp"]
            ratio = torch.exp(log_ratio)
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                kl_mean = gaussian_kl(dv, std, torch.zeros_like(dv), batch["old_std"])
                if self.is_multi_gpu:
                    torch.distributed.all_reduce(approx_kl, op=torch.distributed.ReduceOp.SUM)
                    approx_kl /= self.gpu_world_size
                    torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                    kl_mean /= self.gpu_world_size
            approx_kls.append(approx_kl.item())

            if self.schedule == "adaptive":
                with torch.inference_mode():
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

            surrogate_loss = torch.max(
                -advantages * ratio, -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            ).mean()
            brownian = F.mse_loss((1.0 - tc) * v, x_t - tc * v_old)
            entropy = dist.entropy().sum(-1).mean()

            values = self.critic(batch["observations"])
            if self.use_clipped_value_loss:
                value_clipped = batch["values"] + (values - batch["values"]).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max(
                    (values - batch["returns"]).pow(2), (value_clipped - batch["returns"]).pow(2)
                ).mean()
            else:
                value_loss = (batch["returns"] - values).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy
                + self.brownian_reg_coef * brownian
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            with torch.no_grad():
                actor_sq = self._grad_norm_sq(self.actor.parameters())
                critic_sq = self._grad_norm_sq(self.critic.parameters())
            # ONE clip over actor + critic together (reference)
            nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            with torch.no_grad():
                after_sq = self._grad_norm_sq(params)
            self.optimizer.step()

            with torch.no_grad():
                stats["value"].append(value_loss.detach())
                stats["surrogate"].append(surrogate_loss.detach())
                stats["entropy"].append(entropy.detach())
                stats["policyflow/brownian"].append(brownian.detach())
                stats["policyflow/kl"].append(kl_mean.detach())
                stats["policyflow/ratio_mean"].append(ratio.detach().mean())
                stats["policyflow/clip_frac"].append(((ratio.detach() - 1.0).abs() > self.clip_param).float().mean())
                stats["policyflow/dv_abs_max"].append(dv.detach().abs().max())
                stats["policyflow/noise_std_mean"].append(std.detach().mean())
                stats["grad_norm/actor"].append(actor_sq.sqrt())
                stats["grad_norm/critic"].append(critic_sq.sqrt())
                stats["grad_norm/combined"].append((actor_sq + critic_sq).sqrt())
                stats["grad_norm/combined_after_clip"].append(after_sq.sqrt())

            if self.target_kl is not None and approx_kls[-1] > self.target_kl:
                stopped = True
                break

        self.storage.clear()

        loss_dict: dict[str, float] = {}
        for key, values in stats.items():
            stacked = torch.stack(values)
            if key == "policyflow/dv_abs_max":
                loss_dict[key] = stacked.max().item()
            elif key.startswith("grad_norm/"):
                loss_dict[f"{key}_mean"] = stacked.mean().item()
            else:
                loss_dict[key] = stacked.mean().item()
        loss_dict["policyflow/approx_kl_mean"] = sum(approx_kls) / len(approx_kls)
        loss_dict["kl_stop/mini_batch_updates"] = float(len(approx_kls))
        loss_dict["kl_stop/stopped"] = float(stopped)
        loss_dict["policyflow/lr"] = float(self.learning_rate)
        return loss_dict

    # ------------------------------------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------------------------------------

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PolicyFlow:
        """Construct PolicyFlow: a ``PolicyFlowModel`` actor (from ``cfg['actor']['class_name']``), critic, storage."""
        alg_class: type[PolicyFlow] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[PolicyFlowModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        cfg["algorithm"].pop("share_cnn_encoders", None)

        actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        storage = RolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
            num_eval_envs=PPO._num_eval_envs(cfg["algorithm"].get("eval_env_fraction", 0.0), env.num_envs),
        )
        alg = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])
        alg.compile(cfg.get("torch_compile_mode"))
        return alg
