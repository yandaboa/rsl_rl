# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FPO++: flow policy optimization (arXiv 2602.02481), ported from the authors' isaaclab_fpo reference.

The likelihood ratio is replaced by per-sample ``exp(old_cfm_loss - new_cfm_loss)`` over ``(eps, t)`` pairs drawn
once at rollout time and reused for every recomputation in the update.
"""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.models.flow_model import FlowMLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


def clamp_ste(x: torch.Tensor, min: float | None = None, max: float | None = None) -> torch.Tensor:
    """Clamp in the forward pass, identity gradient in the backward pass."""
    return x + (x.clamp(min=min, max=max) - x).detach()


class ExponentialMovingAverage:
    """EMA of a module's trainable parameters (``ema = decay * ema + (1 - decay) * param``)."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.model_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.shadow_params = {n: p.detach().clone() for n, p in self.model_params.items()}

    @torch.no_grad()
    def update(self) -> None:
        for name, param in self.model_params.items():
            self.shadow_params[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def reset_to_current(self) -> None:
        for name, param in self.model_params.items():
            self.shadow_params[name].copy_(param.detach())

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow_params:
                param.copy_(self.shadow_params[name])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow_params": {n: p.clone() for n, p in self.shadow_params.items()}}

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = state_dict["decay"]
        for name, value in state_dict["shadow_params"].items():
            self.shadow_params[name].copy_(value)


class FPO(PPO):
    """Flow Policy Optimization on a :class:`FlowMLPModel` actor (critic, storage, eval pool, multi-GPU from PPO)."""

    actor: FlowMLPModel

    def __init__(
        self,
        actor: FlowMLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        n_samples_per_action: int = 16,
        trust_region_mode: str = "aspo",
        cfm_loss_clamp: float = 20.0,
        cfm_loss_clamp_negative_advantages: bool = True,
        cfm_loss_clamp_negative_advantages_max: float = 20.0,
        cfm_diff_clamp_max: float = 10.0,
        advantage_clamp: tuple[float, float] | list[float] = (100.0, 100.0),
        normalize_advantage: bool = True,
        weight_decay: float = 1e-4,
        adam_betas: tuple[float, float] | list[float] = (0.9, 0.999),
        ema_decay: float = 0.95,
        ema_warmup_steps: int = 500,
        knn_entropy_coef: float = 0.0,
        knn_entropy_k: int = 1,
        storage_action_noise_std: float = 0.0,
        target_kl: float | None = None,
        cfm_loss_neg_adv_max_increase: float | None = None,
        lr_decay_final: float = 0.0,
        lr_decay_updates: int = 0,
        **kwargs,
    ) -> None:
        assert isinstance(actor, FlowMLPModel), f"FPO needs a FlowMLPModel actor, got {type(actor).__name__}"
        assert not (actor.is_recurrent or critic.is_recurrent), "FPO does not support recurrent models"
        assert not kwargs.get("rnd_cfg"), "FPO does not support RND"
        assert not kwargs.get("symmetry_cfg"), "FPO does not support symmetry augmentation"
        assert trust_region_mode in ("ppo", "spo", "aspo"), f"unknown trust_region_mode {trust_region_mode!r}"
        kwargs.setdefault("use_clipped_value_loss", False)
        kwargs.setdefault("schedule", "fixed")
        super().__init__(actor, critic, storage, **kwargs)
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

        self.n_samples_per_action = int(n_samples_per_action)
        self.trust_region_mode = trust_region_mode
        # cleanRL / ManiSkill early stop on approx KL = mean((r - 1) - log r) of the per-sample ratios before each
        # step; the step still applies, then the whole update ends above it. None = off (approx KL still logged)
        self.target_kl = None if target_kl is None else float(target_kl)
        # negative-advantage samples stop receiving gradient once their CFM loss exceeds the rollout loss by this
        # (log ratio >= -delta per sample, the mirror of cfm_diff_clamp_max); None = only the absolute clamp
        # linear learning-rate decay (schedule="fixed", on when lr_decay_updates > 0): lr -> lr_decay_final
        self.lr_initial = float(self.learning_rate)
        self.lr_decay_final = float(lr_decay_final)
        self.lr_decay_updates = int(lr_decay_updates)
        self.cfm_loss_neg_adv_max_increase = (
            None if cfm_loss_neg_adv_max_increase is None else float(cfm_loss_neg_adv_max_increase)
        )
        self.cfm_loss_clamp = cfm_loss_clamp
        self.cfm_loss_clamp_negative_advantages = cfm_loss_clamp_negative_advantages
        self.cfm_loss_clamp_negative_advantages_max = cfm_loss_clamp_negative_advantages_max
        self.cfm_diff_clamp_max = cfm_diff_clamp_max
        self.advantage_clamp = tuple(advantage_clamp)
        self.normalize_advantage = normalize_advantage
        self.knn_entropy_coef = knn_entropy_coef
        self.knn_entropy_k = knn_entropy_k
        self.storage_action_noise_std = storage_action_noise_std
        if knn_entropy_coef > 0:
            assert 1 <= knn_entropy_k < self.n_samples_per_action, "knn_entropy_k must be in [1, n_samples)"

        # EMA of the actor, updated once per update() after ema_warmup_steps (reset to current at warmup)
        self.ema_decay = ema_decay
        self.ema_warmup_steps = ema_warmup_steps
        self.ema = ExponentialMovingAverage(self._raw_actor, ema_decay) if ema_decay > 0 else None
        self.tot_timesteps = 0
        self._in_eval_mode = False
        self._ema_actor: FlowMLPModel | None = None

        # FPO-owned rollout buffers [T, N, ...], written at the storage's step index
        t_steps, num_envs = storage.num_transitions_per_env, storage.num_envs
        s, a = self.n_samples_per_action, actor.num_actions
        dev = self.device
        # Normalized actor latent at act time: the update recomputes the CFM loss on exactly the input the old
        # loss saw (the reference stores normalized obs), not on obs renormalized with post-rollout statistics.
        self.fpo_latent = torch.zeros(t_steps, num_envs, actor.obs_dim, device=dev)
        self.fpo_eps = torch.zeros(t_steps, num_envs, s, a, device=dev)
        self.fpo_t = torch.zeros(t_steps, num_envs, s, 1, device=dev)
        self.fpo_old_loss = torch.zeros(t_steps, num_envs, s, device=dev)
        self.fpo_old_x1 = torch.zeros(t_steps, num_envs, s, a, device=dev) if self.schedule == "adaptive" else None

    # ------------------------------------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------------------------------------

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions, draw and store the ``(eps, t)`` pairs and the old CFM loss; eval envs act zero-noise."""
        step = self.storage.step
        num_envs = self.storage.num_envs
        self.transition.hidden_states = (None, None)
        actions = self.actor(obs, stochastic_output=True).detach()
        if self.storage_action_noise_std > 0:
            actions = actions + self.storage_action_noise_std * torch.randn_like(actions)
        if self.eval_env_mask is not None and self.eval_deterministic:
            actions = torch.where(self.eval_env_mask.unsqueeze(-1), self.actor.output_mean.detach(), actions)
        self.transition.actions = actions
        self.transition.values = self.critic(obs).detach()
        # Storage bookkeeping only (FPO has no log-probs / distribution parameters)
        self.transition.actions_log_prob = torch.zeros(num_envs, device=self.device)
        self.transition.distribution_params = (torch.zeros(num_envs, 1, device=self.device),)
        self.transition.observations = obs

        latent = self.actor.get_latent(obs).detach()
        eps = torch.randn(num_envs, self.n_samples_per_action, self.actor.num_actions, device=self.device)
        t = self.actor.sample_cfm_t((num_envs, self.n_samples_per_action, 1), self.device)
        old_loss, old_x1, _ = self.actor.get_cfm_loss_from_latent(latent, actions, eps, t)
        self.fpo_latent[step].copy_(latent)
        self.fpo_eps[step].copy_(eps)
        self.fpo_t[step].copy_(t)
        self.fpo_old_loss[step].copy_(old_loss.detach())
        if self.fpo_old_x1 is not None:
            self.fpo_old_x1[step].copy_(old_x1.detach())
        return actions

    def compute_returns(self, obs: TensorDict) -> None:
        """GAE (PPO); advantages normalized over the training pool unless ``normalize_advantage=False``."""
        super().compute_returns(obs)
        if not self.normalize_advantage:
            self.storage.advantages = self.storage.returns - self.storage.values

    # ------------------------------------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------------------------------------

    def _mini_batch_generator(self):
        """Flat shuffled mini-batches over the training envs; storage and FPO buffers share one permutation."""
        st = self.storage
        train = slice(None, st.num_train_envs)
        batch_size = st.num_train_envs * st.num_transitions_per_env
        mini_batch_size = batch_size // self.num_mini_batches
        flat = {
            "observations": st.observations[:, train].flatten(0, 1),
            "actions": st.actions[:, train].flatten(0, 1),
            "values": st.values[:, train].flatten(0, 1),
            "advantages": st.advantages[:, train].flatten(0, 1),
            "returns": st.returns[:, train].flatten(0, 1),
            "latent": self.fpo_latent[:, train].flatten(0, 1),
            "eps": self.fpo_eps[:, train].flatten(0, 1),
            "t": self.fpo_t[:, train].flatten(0, 1),
            "old_loss": self.fpo_old_loss[:, train].flatten(0, 1),
        }
        if self.fpo_old_x1 is not None:
            flat["old_x1"] = self.fpo_old_x1[:, train].flatten(0, 1)
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
                "fpo/ratio_mean",
                "fpo/clip_frac",
                "fpo/cfm_loss_mean",
                "fpo/log_ratio_abs_max",
                "fpo/neg_adv_clamp_frac",
                "grad_norm/actor",
                "grad_norm/critic",
                "grad_norm/combined",
                "grad_norm/combined_after_clip",
            )
        }
        if self.lr_decay_updates > 0 and self.schedule == "fixed":
            frac = min(self.tot_timesteps / self.lr_decay_updates, 1.0)
            self.learning_rate = self.lr_initial + frac * (self.lr_decay_final - self.lr_initial)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate
        kl_values: list[torch.Tensor] = []
        approx_kls: list[float] = []
        stopped = False
        pos_clamp, neg_clamp = self.advantage_clamp
        params = list(chain(self.actor.parameters(), self.critic.parameters()))

        for batch in self._mini_batch_generator():
            advantages = batch["advantages"]  # [B, 1]
            if self.normalize_advantage and self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = advantages.clamp(-neg_clamp, pos_clamp)

            cfm_loss, x1_pred, x0_pred = self.actor.get_cfm_loss_from_latent(
                batch["latent"], batch["actions"], batch["eps"], batch["t"]
            )
            values = self.critic(batch["observations"])
            entropy_bonus = None
            if self.knn_entropy_coef > 0:
                entropy_bonus = self._compute_knn_entropy(x0_pred, self.knn_entropy_k)

            # KL proxy: mean squared change of the implied x1 prediction (reference)
            if self.schedule == "adaptive":
                with torch.inference_mode():
                    kl_mean = ((x1_pred.detach() - batch["old_x1"]) ** 2).mean()
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
                kl_values.append(kl_mean.detach().clone())

            old_loss = batch["old_loss"]
            if self.cfm_loss_clamp > 0:
                old_loss = old_loss.clamp(max=self.cfm_loss_clamp)
                cfm_loss = cfm_loss.clamp(max=self.cfm_loss_clamp)
            if self.cfm_loss_clamp_negative_advantages:
                cfm_loss = torch.where(
                    advantages < 0, cfm_loss.clamp(max=self.cfm_loss_clamp_negative_advantages_max), cfm_loss
                )
            neg_clamped = None
            if self.cfm_loss_neg_adv_max_increase is not None:
                bound = old_loss + self.cfm_loss_neg_adv_max_increase
                neg_clamped = (advantages < 0) & (cfm_loss > bound)
                cfm_loss = torch.where(advantages < 0, torch.minimum(cfm_loss, bound), cfm_loss)
            log_ratio = clamp_ste(old_loss - cfm_loss, max=self.cfm_diff_clamp_max)
            ratio = torch.exp(log_ratio)  # [B, S]
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                if self.is_multi_gpu:
                    torch.distributed.all_reduce(approx_kl, op=torch.distributed.ReduceOp.SUM)
                    approx_kl /= self.gpu_world_size
            approx_kls.append(approx_kl.item())
            surrogate_loss = self._surrogate_loss(ratio, advantages)

            if self.use_clipped_value_loss:
                value_clipped = batch["values"] + (values - batch["values"]).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max(
                    (values - batch["returns"]).pow(2), (value_clipped - batch["returns"]).pow(2)
                ).mean()
            else:
                value_loss = (batch["returns"] - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss
            if entropy_bonus is not None:
                loss = loss - self.knn_entropy_coef * entropy_bonus

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
                stats["entropy"].append(
                    entropy_bonus.detach() if entropy_bonus is not None else torch.zeros((), device=self.device)
                )
                stats["fpo/ratio_mean"].append(ratio.detach().mean())
                stats["fpo/clip_frac"].append(((ratio.detach() - 1.0).abs() > self.clip_param).float().mean())
                stats["fpo/cfm_loss_mean"].append(cfm_loss.detach().mean())
                stats["fpo/log_ratio_abs_max"].append(log_ratio.detach().abs().max())
                stats["fpo/neg_adv_clamp_frac"].append(
                    neg_clamped.float().mean() if neg_clamped is not None else torch.zeros((), device=self.device)
                )
                stats["grad_norm/actor"].append(actor_sq.sqrt())
                stats["grad_norm/critic"].append(critic_sq.sqrt())
                stats["grad_norm/combined"].append((actor_sq + critic_sq).sqrt())
                stats["grad_norm/combined_after_clip"].append(after_sq.sqrt())

            if self.target_kl is not None and approx_kls[-1] > self.target_kl:
                stopped = True
                break

        self.storage.clear()

        # EMA once per update (reference: reset to current weights at the warmup step, then blend)
        self.tot_timesteps += 1
        if self.ema is not None:
            if self.tot_timesteps == self.ema_warmup_steps:
                self.ema.reset_to_current()
            elif self.tot_timesteps > self.ema_warmup_steps:
                self.ema.update()

        loss_dict: dict[str, float] = {}
        for key, values in stats.items():
            stacked = torch.stack(values)
            if key == "fpo/log_ratio_abs_max":
                loss_dict[key] = stacked.max().item()
            elif key.startswith("grad_norm/"):
                loss_dict[f"{key}_mean"] = stacked.mean().item()
            else:
                loss_dict[key] = stacked.mean().item()
        if kl_values:
            loss_dict["fpo/kl_proxy"] = torch.stack(kl_values).mean().item()
        loss_dict["fpo/approx_kl_first"] = approx_kls[0]
        loss_dict["fpo/approx_kl_mean"] = sum(approx_kls) / len(approx_kls)
        loss_dict["kl_stop/mini_batch_updates"] = float(len(approx_kls))
        loss_dict["kl_stop/stopped"] = float(stopped)
        loss_dict["fpo/lr"] = float(self.learning_rate)
        loss_dict["fpo/ema_active"] = float(self._ema_active())
        return loss_dict

    def _surrogate_loss(self, ratio: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        """Trust-region surrogate over ``[B, S]`` per-sample ratios; ``advantages`` is ``[B, 1]``."""
        ppo_loss = torch.max(
            -advantages * ratio, -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        )
        if self.trust_region_mode == "ppo":
            return ppo_loss.mean()
        spo_loss = -(ratio * advantages - advantages.abs() / (2.0 * self.clip_param) * (ratio - 1.0) ** 2)
        if self.trust_region_mode == "spo":
            return spo_loss.mean()
        return torch.where(advantages > 0, ppo_loss, spo_loss).mean()

    def _compute_knn_entropy(self, x0_pred: torch.Tensor, k: int) -> torch.Tensor:
        """Kozachenko-Leonenko k-NN entropy estimate over the ``S`` samples of each row (reference)."""
        batch, n_samples, action_dim = x0_pred.shape
        dists = torch.cdist(x0_pred, x0_pred, p=2)
        dists = dists + torch.eye(n_samples, device=x0_pred.device).unsqueeze(0) * 1e10
        rho_k = torch.topk(dists, k=k, dim=2, largest=False, sorted=True).values[:, :, -1].clamp(1e-6, 1e9)
        psi_n = torch.digamma(torch.tensor(float(n_samples), device=x0_pred.device))
        psi_k = torch.digamma(torch.tensor(float(k), device=x0_pred.device))
        log_cd = (action_dim / 2) * math.log(math.pi) - math.lgamma(action_dim / 2 + 1)
        entropy = psi_n - psi_k + log_cd + action_dim * torch.log(rho_k).mean(dim=1)
        return entropy.mean()

    # ------------------------------------------------------------------------------------------------
    # Modes, policy, checkpoints
    # ------------------------------------------------------------------------------------------------

    def _ema_active(self) -> bool:
        return self.ema is not None and self.tot_timesteps > self.ema_warmup_steps

    def train_mode(self) -> None:
        super().train_mode()
        self._in_eval_mode = False

    def eval_mode(self) -> None:
        super().eval_mode()
        self._in_eval_mode = True

    def get_policy(self) -> FlowMLPModel:
        """The training actor; in eval mode with an active EMA, a copy carrying the EMA weights."""
        if not (self._in_eval_mode and self._ema_active()):
            return self._raw_actor
        if self._ema_actor is None:
            self._ema_actor = copy.deepcopy(self._raw_actor)
        else:
            self._ema_actor.load_state_dict(self._raw_actor.state_dict())  # normalizer buffers
        self.ema.copy_to(self._ema_actor)  # type: ignore[union-attr]
        return self._ema_actor.eval()

    def save(self) -> dict:
        saved = super().save()
        if self.ema is not None:
            saved["ema_state_dict"] = {**self.ema.state_dict(), "tot_timesteps": self.tot_timesteps}
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if (load_cfg is None or load_cfg.get("actor")) and self.ema is not None and "ema_state_dict" in loaded_dict:
            ema_state = loaded_dict["ema_state_dict"]
            self.ema.load_state_dict(ema_state)
            self.tot_timesteps = int(ema_state.get("tot_timesteps", 0))
        return load_iteration

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> FPO:
        """Construct FPO: a ``FlowMLPModel`` actor (from ``cfg['actor']['class_name']``), PPO critic and storage."""
        alg_class: type[FPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[FlowMLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
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
