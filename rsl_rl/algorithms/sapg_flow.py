# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SAPG with a PolicyFlow leader: block 0 acts with a flow policy, blocks 1..M-1 with Gaussian PPO heads."""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.algorithms.sapg import SAPG
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.models.block_conditioning import block_context
from rsl_rl.models.policyflow_model import PolicyFlowModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups

# Legacy isaaclab_rl model-cfg keys the runner cfg may still carry on the flow-actor dict.
_DEPRECATED_MODEL_KEYS = ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std", "distribution_cfg")
_LEADER_ALGOS = ("policyflow", "fpo", "reinflow")


def clamp_ste(x: torch.Tensor, min: float | None = None, max: float | None = None) -> torch.Tensor:
    """Clamp in the forward pass, identity gradient in the backward pass (:func:`rsl_rl.algorithms.fpo.clamp_ste`)."""
    return x + (x.clamp(min=min, max=max) - x).detach()


def _row_gaussian_kl(
    mean: torch.Tensor, std: torch.Tensor, old_mean: torch.Tensor, old_std: torch.Tensor
) -> torch.Tensor:
    """Per-row (summed over dims) form of :func:`rsl_rl.algorithms.policyflow.gaussian_kl`."""
    std_ratio = std / old_std + 1.0e-5
    drift = old_std.square() + (old_mean - mean).square()
    return (std_ratio.log() + drift / (2.0 * std.square()) - 0.5).sum(dim=-1)


class SAPGFlow(SAPG):
    """Leader-follower SAPG whose leader (block 0) is a :class:`PolicyFlowModel`.

    - ``leader_encoder="first_layer"``: followers are the split-SAPG Gaussian heads (``shared_layers`` trunk, no phi)
      and the flow leader shares the actor's obs normalizer + first Linear + activation as its state encoder.
      ``leader_encoder="trunk"``: the Gaussian SAPG actor as is (fully shared MLP, phi per block); the flow leader
      encodes with the whole hidden stack under the leader's phi[0], so only its velocity head loops when sampling.
    - Leader on-policy rows: the PolicyFlow loss (ratio ``N(n; dv, std_new) / N(n; 0, std_old)``, Brownian reg).
      ``leader_algo="fpo"`` / ``"reinflow"`` swap in FPO++ (CFM-loss ratio) / ReinFlow (per-step Euler noise chain);
      both need ``lambda_off=0`` (AWFM still applies).
    - Relabeled follower rows: ``-lambda_off * mu * min(rho A, clip(rho) A)`` with ``rho`` the marginal new/old
      leader ratio over ``num_latents`` flow latents and ``mu = pi_L_old / pi_F`` (detached, capped at ``mu_max``),
      plus ``fm_coef * w * ||v(x_t, t) - (a - x0)||^2`` with ``w = min(exp(A / fm_beta), fm_w_max)`` (AWFM).
    - One AdamW over actor + critic + flow leader, one joint grad clip.

    Known limitation: ``runner.get_inference_policy`` / export return the Gaussian actor (block 0's head is never
    trained); use :meth:`leader_policy` for the flow leader.
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        flow_cfg: dict,
        num_latents: int = 8,
        lambda_off: float = 1.0,
        mu_max: float = 10.0,
        dual_clip: float | None = None,
        off_detach_encoder: bool = False,
        fm_coef: float = 0.0,
        fm_beta: float = 1.0,
        fm_w_max: float = 10.0,
        brownian_reg_coef: float = 0.002,
        weight_decay: float = 1e-5,
        adam_betas: tuple[float, float] | list[float] = (0.9, 0.999),
        leader_algo: str = "policyflow",
        encoder_grad_minibatches: int = 4,
        fpo_samples: int = 8,
        fpo_cfm_loss_clamp: float = 20.0,
        fpo_cfm_diff_clamp_max: float = 10.0,
        fpo_neg_adv_max_increase: float | None = None,
        fpo_trust_region: str = "ppo",
        reinflow_sigma_min: float = 0.05,
        reinflow_sigma_max: float = 0.3,
        reinflow_noise_hidden: int = 64,
        leader_encoder: str = "first_layer",
        **sapg_kwargs,
    ) -> None:
        """``leader_algo``: the leader's on-policy objective, ``"policyflow"`` (Gaussian noise on phi(x0)), ``"fpo"``
        (FPO++ CFM-loss ratio, ``fpo_*`` knobs) or ``"reinflow"`` (learned per-step Euler noise, ``reinflow_*``).
        ``encoder_grad_minibatches``: the first N minibatch steps of each update log ``grad_enc/*`` (0 = off).
        """
        assert leader_algo in _LEADER_ALGOS, f"leader_algo must be one of {_LEADER_ALGOS}, got {leader_algo!r}"
        # the marginal IS ratio of relabeled rows is PolicyFlow-specific
        assert leader_algo == "policyflow" or lambda_off == 0.0, f"leader_algo={leader_algo!r} needs lambda_off=0."
        assert fpo_trust_region in ("ppo", "spo", "aspo"), f"unknown fpo_trust_region {fpo_trust_region!r}"
        assert fpo_samples >= 1 and encoder_grad_minibatches >= 0
        assert 0.0 < reinflow_sigma_min < reinflow_sigma_max, "need 0 < reinflow_sigma_min < reinflow_sigma_max"
        assert leader_encoder in ("first_layer", "trunk"), f"unknown leader_encoder {leader_encoder!r}"
        if leader_encoder == "first_layer":
            assert sapg_kwargs.get("shared_layers") is not None, "SAPGFlow needs shared_layers (the shared encoder)."
            assert int(sapg_kwargs.get("phi_dim", 32)) == 0, "SAPGFlow needs phi_dim=0."
        else:
            assert sapg_kwargs.get("shared_layers") is None and sapg_kwargs.get("block_layers") is None, (
                "leader_encoder='trunk' needs the fully shared SAPG MLP (no per-block heads)."
            )
        assert int(sapg_kwargs.get("num_blocks", 6)) >= 2, "SAPGFlow needs num_blocks >= 2 (leader + followers)."
        assert num_latents >= 1 and mu_max > 0.0
        super().__init__(actor, critic, storage, **sapg_kwargs)
        # Block 0's Gaussian head exists (attach_block_conditioning) but never acts or trains: the flow leader does.

        mlp = self._raw_actor.mlp
        if leader_encoder == "first_layer":
            assert type(mlp[0]) is nn.Linear, f"the shared first layer must be a plain nn.Linear, got {type(mlp[0]).__name__}."
            hidden = (mlp[0], mlp[1])
        else:
            # every layer but the mean head; a bare (phi-less) input makes the PhiConditionedLinear use phi[0]
            hidden = tuple(list(mlp)[:-1])
        encoder = (self._raw_actor.obs_normalizer, *hidden)
        self._encoder_params: list[nn.Parameter] = [p for m in hidden for p in m.parameters()]
        flow_cfg = dict(flow_cfg)
        flow_class = resolve_callable(flow_cfg.pop("class_name", PolicyFlowModel))
        for key in _DEPRECATED_MODEL_KEYS:
            flow_cfg.pop(key, None)
        obs0 = storage.observations[0]
        obs_groups = {"actor": list(self._raw_actor.obs_groups)}
        self.flow_leader: PolicyFlowModel = flow_class(
            obs0, obs_groups, "actor", storage.actions.shape[-1], encoder=encoder, **flow_cfg
        ).to(self.device)
        assert isinstance(self.flow_leader, PolicyFlowModel)
        assert not self.flow_leader.obs_normalization

        self.num_latents = int(num_latents)
        self.lambda_off = float(lambda_off)
        self.mu_max = float(mu_max)
        # Ye et al. 2020 dual clip: bounds the negative-advantage relabeled term at dual_clip * |A| (the marginal
        # ratio spans e^+-20 while leader and followers are still unrelated policies).
        self.dual_clip = None if dual_clip is None else float(dual_clip)
        # relabeled rows train the flow only, not the shared encoder (follower-diversity control)
        self.off_detach_encoder = bool(off_detach_encoder)
        assert self.dual_clip is None or self.dual_clip > 1.0, f"dual_clip must be > 1, got {dual_clip}"
        # advantage-weighted flow matching toward the relabeled follower actions (needs no support overlap)
        self.fm_coef = float(fm_coef)
        self.fm_beta = float(fm_beta)
        self.fm_w_max = float(fm_w_max)
        assert self.fm_coef >= 0.0 and self.fm_beta > 0.0 and self.fm_w_max > 0.0
        self.brownian_reg_coef = float(brownian_reg_coef)
        self.leader_algo = leader_algo
        self.encoder_grad_minibatches = int(encoder_grad_minibatches)
        self.fpo_samples = int(fpo_samples)
        self.fpo_cfm_loss_clamp = float(fpo_cfm_loss_clamp)
        self.fpo_cfm_diff_clamp_max = float(fpo_cfm_diff_clamp_max)
        self.fpo_neg_adv_max_increase = None if fpo_neg_adv_max_increase is None else float(fpo_neg_adv_max_increase)
        self.fpo_trust_region = fpo_trust_region
        self.reinflow_sigma_min = float(reinflow_sigma_min)
        self.reinflow_sigma_max = float(reinflow_sigma_max)
        self.rf_noise_head: nn.Sequential | None = None
        if leader_algo == "reinflow":
            # the chain likelihood needs explicit per-step means
            assert self.flow_leader.integrator == "euler", "leader_algo='reinflow' needs flow_cfg integrator='euler'."
            hidden, num_actions = int(reinflow_noise_hidden), self.flow_leader.num_actions
            head = nn.Sequential(
                nn.Linear(self.flow_leader.latent_dim + 1, hidden), nn.Mish(), nn.Linear(hidden, num_actions)
            )
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
            self.rf_noise_head = head.to(self.device)

        # ONE AdamW over the deduplicated union (the encoder params live in the actor)
        seen: set[int] = set()
        self._all_params: list[nn.Parameter] = []
        for p in chain(self.actor.parameters(), self.critic.parameters(), self._leader_params()):
            if id(p) not in seen:
                seen.add(id(p))
                self._all_params.append(p)
        if weight_decay > 0:
            self.optimizer = optim.AdamW(
                self._all_params, lr=self.learning_rate, betas=tuple(adam_betas), weight_decay=weight_decay
            )
        else:
            self.optimizer = optim.Adam(self._all_params, lr=self.learning_rate, betas=tuple(adam_betas))
        self._sync_flow_old()

        n_lead = self.num_leader_envs
        assert bool(self.env_block_ids[:n_lead].eq(0).all()), "the leader block must be env ids 0..n_leader-1."
        t_steps, a, dev = storage.num_transitions_per_env, self.flow_leader.num_actions, self.device
        if leader_algo == "policyflow":
            self.pf_x0 = torch.zeros(t_steps, n_lead, a, device=dev)
            self.pf_phi = torch.zeros(t_steps, n_lead, a, device=dev)
            self.pf_noise = torch.zeros(t_steps, n_lead, a, device=dev)
            self.pf_old_std = torch.zeros(t_steps, n_lead, a, device=dev)
            self.pf_old_logp = torch.zeros(t_steps, n_lead, device=dev)
            self.pf_latent = torch.zeros(t_steps, n_lead, self.flow_leader.latent_dim, device=dev)
        elif leader_algo == "fpo":
            s_ = self.fpo_samples
            self.fpo_eps = torch.zeros(t_steps, n_lead, s_, a, device=dev)
            self.fpo_t = torch.zeros(t_steps, n_lead, s_, device=dev)
            self.fpo_old_loss = torch.zeros(t_steps, n_lead, s_, device=dev)
        else:
            k_ = self.flow_leader.sampling_steps
            self.rf_chain = torch.zeros(t_steps, n_lead, k_ + 1, a, device=dev)
            self.rf_old_mean = torch.zeros(t_steps, n_lead, k_, a, device=dev)
            self.rf_old_sigma = torch.zeros(t_steps, n_lead, k_, a, device=dev)
            self.rf_old_logp = torch.zeros(t_steps, n_lead, device=dev)
        self._flow_stats: dict[str, float] = {}

    def _leader_params(self) -> list[nn.Parameter]:
        """Flow-leader parameters (+ the ReinFlow noise head); the shared encoder lives in the actor."""
        params = list(self.flow_leader.parameters())
        if self.rf_noise_head is not None:
            params += list(self.rf_noise_head.parameters())
        return params

    def _leader_row(self, s: torch.Tensor) -> torch.Tensor:
        """Flat storage row ``[T * num_envs]`` of a leader env -> flat leader-buffer row ``[T * n_leader]``."""
        num_envs = self.storage.num_envs
        return (s // num_envs) * self.num_leader_envs + (s % num_envs)

    def _rf_sigma(self, latent: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """``latent [B, D], t [B] -> sigma [B, A]`` in ``[sigma_min, sigma_max]``."""
        assert self.rf_noise_head is not None
        out = self.rf_noise_head(torch.cat([latent, t[:, None]], dim=-1))
        return self.reinflow_sigma_min + (self.reinflow_sigma_max - self.reinflow_sigma_min) * torch.sigmoid(out)

    def _rf_chain_stats(self, latent: torch.Tensor, chain_x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Teacher-forced Euler chain ``[B, K+1, A]`` -> per-step means / sigmas ``[B, K, A]`` and log-lik ``[B]``."""
        flow = self.flow_leader
        b, k1, a = chain_x.shape
        k = k1 - 1
        x = chain_x[:, :k].reshape(b * k, a)
        t = (torch.arange(k, device=latent.device, dtype=latent.dtype) / k).repeat(b)
        lat = latent.repeat_interleave(k, 0)
        mean = x + flow.velocity(lat, x, t) / k
        sigma = self._rf_sigma(lat, t)
        logp = torch.distributions.Normal(mean, sigma).log_prob(chain_x[:, 1:].reshape(b * k, a)).sum(-1)
        return mean.view(b, k, a), sigma.view(b, k, a), logp.view(b, k).sum(-1)

    def _fpo_cfm_loss(
        self, latent: torch.Tensor, actions: torch.Tensor, eps: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Per-sample CFM loss ``[B, S]`` of ``actions [B, A]`` with noises ``eps [B, S, A]`` at times ``t [B, S]``."""
        b, s_, a = eps.shape
        tc = t.unsqueeze(-1)
        target = actions.unsqueeze(1)
        x_t = (1.0 - tc) * eps + tc * target
        v = self.flow_leader.velocity(latent.repeat_interleave(s_, 0), x_t.reshape(b * s_, a), t.reshape(-1))
        return (v.view(b, s_, a) - (target - eps)).square().mean(-1)

    def _fpo_row_surrogate(self, ratio: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
        """FPO's trust-region surrogate (``ratio [B, S]``, ``adv [B, 1]``), averaged over samples -> ``[B]``."""
        ppo = torch.max(-adv * ratio, -adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param))
        if self.fpo_trust_region == "ppo":
            return ppo.mean(1)
        spo = -(ratio * adv - adv.abs() / (2.0 * self.clip_param) * (ratio - 1.0) ** 2)
        if self.fpo_trust_region == "spo":
            return spo.mean(1)
        return torch.where(adv > 0, ppo, spo).mean(1)

    def _sync_flow_old(self) -> None:
        """Frozen copy of the flow leader incl. its encoder (deepcopy: the encoder is not in the state dict)."""
        with torch.inference_mode(False), torch.no_grad():
            old = copy.deepcopy(self.flow_leader).requires_grad_(False).eval()
            for module in old.encoder or ():
                module.requires_grad_(False)
        self.flow_old: PolicyFlowModel = old

    # ------------------------------------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------------------------------------

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Gaussian heads for every env, then the leader envs' actions are replaced by the flow leader's."""
        step = self.storage.step
        super().act(obs)
        n = self.num_leader_envs
        flow = self.flow_leader
        if self.leader_algo != "policyflow":
            return self._act_leader_noise_free(obs, step, n)
        actions = flow(obs[:n], stochastic_output=True).detach()
        phi = flow.last_phi
        noise = actions - phi
        std = flow.std.detach().expand_as(noise)
        old_logp = torch.distributions.Normal(torch.zeros_like(noise), std).log_prob(noise).sum(-1)
        self.pf_latent[step].copy_(flow.last_latent)
        self.pf_x0[step].copy_(flow.last_x0)
        self.pf_phi[step].copy_(phi)
        self.pf_noise[step].copy_(noise)
        self.pf_old_std[step].copy_(std)
        self.pf_old_logp[step].copy_(old_logp)
        self.transition.actions[:n] = actions
        # Bookkeeping; the leader rows' Gaussian distribution params stay as placeholders.
        self.transition.actions_log_prob[:n] = old_logp
        return self.transition.actions  # type: ignore[return-value]

    def _act_leader_noise_free(self, obs: TensorDict, step: int, n: int) -> torch.Tensor:
        """FPO / ReinFlow leader envs: no output noise (FPO: x0 only; ReinFlow: per-step chain noise)."""
        flow = self.flow_leader
        latent = flow.get_latent(obs[:n])
        if self.leader_algo == "fpo":
            actions = flow.integrate(latent, torch.randn(n, flow.num_actions, device=self.device))
            grid = flow.train_time_grid
            eps = torch.randn(n, self.fpo_samples, flow.num_actions, device=self.device)
            t = grid[torch.randint(0, grid.shape[0], (n, self.fpo_samples), device=self.device)]
            self.fpo_eps[step].copy_(eps)
            self.fpo_t[step].copy_(t)
            self.fpo_old_loss[step].copy_(self._fpo_cfm_loss(latent, actions, eps, t))
            log_prob = torch.zeros(n, device=self.device)
        else:
            k = flow.sampling_steps
            x = torch.randn(n, flow.num_actions, device=self.device)
            xs, means, sigmas = [x], [], []
            for i in range(k):
                t = torch.full((n,), i / k, device=self.device)
                mean = x + flow.velocity(latent, x, t) / k
                sigma = self._rf_sigma(latent, t)
                x = mean + sigma * torch.randn_like(mean)
                xs.append(x)
                means.append(mean)
                sigmas.append(sigma)
            chain_x, mean_t, sigma_t = torch.stack(xs, 1), torch.stack(means, 1), torch.stack(sigmas, 1)
            log_prob = torch.distributions.Normal(mean_t, sigma_t).log_prob(chain_x[:, 1:]).sum((-1, -2))
            actions = x
            self.rf_chain[step].copy_(chain_x)
            self.rf_old_mean[step].copy_(mean_t)
            self.rf_old_sigma[step].copy_(sigma_t)
            self.rf_old_logp[step].copy_(log_prob)
        self.transition.actions[:n] = actions.detach()
        self.transition.actions_log_prob[:n] = log_prob.detach()
        return self.transition.actions  # type: ignore[return-value]

    @torch.no_grad()
    def _refresh_leader_old(self) -> None:
        """Recompute the FPO old CFM losses / ReinFlow old chain stats under the post-rollout obs normalizer.

        The update recomputes the leader latent with the current normalizer, so this makes ratio == 1 before the first
        step (the act-time values drift with the normalizer updates of the rollout).
        """
        st, n = self.storage, self.num_leader_envs
        obs = st.observations[:, :n].flatten(0, 1)
        actions = st.actions[:, :n].flatten(0, 1)
        rows = actions.shape[0]
        flow = self.flow_leader
        if self.leader_algo == "fpo":
            eps, t = self.fpo_eps.flatten(0, 1), self.fpo_t.flatten(0, 1)
            old_loss = self.fpo_old_loss.view(rows, -1)
            chunk = max(1, self.relabel_chunk_size // self.fpo_samples)
            for i0 in range(0, rows, chunk):
                sl = slice(i0, min(i0 + chunk, rows))
                old_loss[sl] = self._fpo_cfm_loss(flow.get_latent(obs[sl]), actions[sl], eps[sl], t[sl])
        else:
            chain_x = self.rf_chain.flatten(0, 1)
            k = flow.sampling_steps
            old_mean, old_sigma = self.rf_old_mean.view(rows, k, -1), self.rf_old_sigma.view(rows, k, -1)
            old_logp = self.rf_old_logp.view(rows)
            chunk = max(1, self.relabel_chunk_size // k)
            for i0 in range(0, rows, chunk):
                sl = slice(i0, min(i0 + chunk, rows))
                old_mean[sl], old_sigma[sl], old_logp[sl] = self._rf_chain_stats(flow.get_latent(obs[sl]), chain_x[sl])

    @torch.no_grad()
    def leader_policy(self, obs: TensorDict, stochastic: bool = False) -> torch.Tensor:
        """Flow-leader action without the Gaussian noise: ``phi(0)``, or ``phi(x0)`` with ``x0 ~ N(0, I)``."""
        flow = self.flow_leader
        latent = flow.get_latent(obs)
        x0 = latent.new_zeros(latent.shape[0], flow.num_actions)
        if stochastic:
            x0 = torch.randn_like(x0)
        return flow.integrate(latent, x0)

    # ------------------------------------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------------------------------------

    def compute_returns(self, obs: TensorDict) -> None:
        """SAPG targets, then the no-grad old-leader cache of the relabeled rows (latents, phi, ell, mu)."""
        self._sync_flow_old()
        super().compute_returns(obs)
        if self.leader_algo != "policyflow":
            with torch.inference_mode(False):
                self._refresh_leader_old()
        rows = self._rows
        assert rows is not None
        st = self.storage
        k = self.num_latents
        a = self.flow_leader.num_actions
        with torch.inference_mode(False), torch.no_grad():
            off = rows["off"].clone()
            num_rows = off.shape[0]
            src = rows["src"][off].clone()
            r = src.shape[0]
            off_pos = torch.full((num_rows,), -1, dtype=torch.long, device=self.device)
            off_pos[off] = torch.arange(r, device=self.device)
            rows["off_pos"] = off_pos
            rows["off_z"] = torch.empty(r, k, a, device=self.device)
            rows["off_phi"] = torch.empty(r, k, a, device=self.device)
            rows["off_ell"] = torch.empty(r, k, device=self.device)
            rows["off_log_mu"] = torch.empty(r, device=self.device)
            self._flow_stats = {}
            if r == 0:
                return
            flat_obs = st.observations.flatten(0, 1)
            flat_actions = st.actions.flatten(0, 1)
            behavior = st.actions_log_prob.flatten(0, 1).view(-1)
            old = self.flow_old
            std_old = old.std
            log_mu_raw = torch.empty(r, device=self.device)
            eff = torch.empty(r, device=self.device)
            log_pi_old = torch.empty(r, device=self.device)
            phi_spread = torch.empty(r, device=self.device)
            nearest = torch.empty(r, device=self.device)
            # the integrate batch is chunk x K rows
            chunk = max(1, self.relabel_chunk_size // k)
            for i0 in range(0, r, chunk):
                sl = slice(i0, min(i0 + chunk, r))
                s = src[sl]
                z, phi = old.sample_phi(old.get_latent(flat_obs[s]), k)
                ell = torch.distributions.Normal(phi, std_old).log_prob(flat_actions[s].unsqueeze(1)).sum(-1)
                rows["off_z"][sl] = z
                rows["off_phi"][sl] = phi
                rows["off_ell"][sl] = ell
                log_pi_old[sl] = torch.logsumexp(ell, dim=1) - math.log(k)
                log_mu_raw[sl] = log_pi_old[sl] - behavior[s]
                w = torch.softmax(ell, dim=1)
                eff[sl] = 1.0 / w.square().sum(dim=1)
                phi_spread[sl] = phi.std(dim=1).mean(dim=-1) if k > 1 else phi.new_zeros(phi.shape[0])
                nearest[sl] = (phi - flat_actions[s].unsqueeze(1)).norm(dim=-1).min(dim=1).values
            log_cap = math.log(self.mu_max)
            rows["off_log_mu"] = log_mu_raw.clamp(max=log_cap)
            mu = rows["off_log_mu"].exp()
            self._flow_stats = {
                "SAPGFlow/log_mu_mean": log_mu_raw.mean().item(),
                "SAPGFlow/log_mu_std": log_mu_raw.std().item() if r > 1 else 0.0,
                "SAPGFlow/mu_ess": (mu.sum().square() / mu.square().sum() / r).item(),
                "SAPGFlow/mu_clip_frac": (log_mu_raw > log_cap).float().mean().item(),
                "SAPGFlow/eff_latents": eff.mean().item(),
                "SAPGFlow/phi_spread": phi_spread.mean().item(),
                "SAPGFlow/off_nearest_dist": nearest.mean().item(),
                "SAPGFlow/log_pi_old_mean": log_pi_old.mean().item(),
                "SAPGFlow/behavior_logp_mean": behavior[src].mean().item(),
            }

    # ------------------------------------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------------------------------------

    def minibatch_losses(self, idx: torch.Tensor, flat: dict, old_params: tuple[torch.Tensor, ...]) -> dict:
        """Gaussian PPO rows (followers), PolicyFlow rows (leader on-policy), marginal-ratio rows (relabeled)."""
        rows = self._rows
        assert rows is not None
        src = rows["src"][idx]
        block = rows["block"][idx]
        off = rows["off"][idx]
        gauss = block >= 1
        lead_on = (block == 0) & ~off
        obs = flat["observations"][src]
        actions = flat["actions"][src]
        old_log_prob = flat["actions_log_prob"][src].view(-1)
        old_values = rows["values"][idx]
        returns = rows["returns"][idx]
        advantages = rows["advantages"][idx].view(-1)
        if self.normalize_advantage_per_mini_batch:
            with torch.no_grad():
                if self.normalize_advantage_per_block:
                    # follower blocks, leader on-policy rows, relabeled rows: each its own group
                    group = torch.where(off, torch.full_like(block, self.num_blocks), block)
                    normalized = torch.empty_like(advantages)
                    for j in range(self.num_blocks + 1):
                        mask = group == j
                        if bool(mask.any()):
                            adv_j = advantages[mask]
                            std_j = adv_j.std() if adv_j.numel() > 1 else adv_j.new_zeros(())
                            normalized[mask] = (adv_j - adv_j.mean()) / (std_j + 1e-8)
                    advantages = normalized
                else:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Gaussian path for every row (leader rows are placeholders, overwritten below)
        with block_context(block, self.actor, self.critic):
            self.actor(obs, stochastic_output=True)
            log_prob = self.actor.get_output_log_prob(actions)  # type: ignore
            values = self.critic(obs)
        entropy = self.actor.output_entropy
        params = tuple(p for p in self.actor.output_distribution_params)
        ratio_g = torch.exp(torch.clamp(log_prob - old_log_prob, -20.0, 20.0))
        surrogate = torch.max(
            -advantages * ratio_g, -advantages * torch.clamp(ratio_g, 1.0 - self.clip_param, 1.0 + self.clip_param)
        )
        surrogate = torch.where(gauss, surrogate, torch.zeros_like(surrogate))
        entropy_coef = rows["entropy_coef"][idx]
        entropy_rows = torch.where(gauss, entropy_coef * entropy, torch.zeros_like(entropy))
        entropy_raw = torch.where(gauss, entropy, torch.zeros_like(entropy)).detach()
        with torch.no_grad():
            ratio = ratio_g.detach().clone()
            kl = self.actor.get_kl_divergence(old_params, tuple(p.detach() for p in params))  # type: ignore
            kl = torch.where(gauss, kl, torch.zeros_like(kl))

        flow, old = self.flow_leader, self.flow_old
        grid = flow.train_time_grid
        std_new = flow.std
        brownian = surrogate.new_zeros(())
        fm_loss = surrogate.new_zeros(())
        fm_w_mean = surrogate.new_zeros(())
        off_log_rho = surrogate.new_zeros(0)
        off_mu = surrogate.new_zeros(0)
        leader_stats: dict[str, torch.Tensor] = {}

        # Leader on-policy rows: FPO++ / ReinFlow
        if bool(lead_on.any()) and self.leader_algo != "policyflow":
            lead_idx = torch.nonzero(lead_on).view(-1)
            lead_row = self._leader_row(src[lead_idx])
            latent = flow.get_latent(obs[lead_idx])
            adv = advantages[lead_idx]
            if self.leader_algo == "fpo":
                eps = self.fpo_eps.flatten(0, 1)[lead_row]
                t = self.fpo_t.flatten(0, 1)[lead_row]
                old_loss = self.fpo_old_loss.flatten(0, 1)[lead_row]
                new_loss = self._fpo_cfm_loss(latent, actions[lead_idx], eps, t)
                leader_stats["fpo_cfm_loss"] = new_loss.detach().mean()
                if self.fpo_cfm_loss_clamp > 0:
                    old_loss = old_loss.clamp(max=self.fpo_cfm_loss_clamp)
                    new_loss = new_loss.clamp(max=self.fpo_cfm_loss_clamp)
                adv_col = adv[:, None]
                if self.fpo_neg_adv_max_increase is not None:
                    bound = old_loss + self.fpo_neg_adv_max_increase
                    leader_stats["fpo_neg_clamp_frac"] = ((adv_col < 0) & (new_loss > bound)).float().mean()
                    new_loss = torch.where(adv_col < 0, torch.minimum(new_loss, bound), new_loss)
                log_ratio = clamp_ste(old_loss - new_loss, max=self.fpo_cfm_diff_clamp_max)
                r_lead = torch.exp(log_ratio)
                surrogate = surrogate.index_put((lead_idx,), self._fpo_row_surrogate(r_lead, adv_col))
                with torch.no_grad():
                    ratio[lead_idx] = r_lead.detach().mean(1)
                    kl[lead_idx] = ((r_lead.detach() - 1.0) - log_ratio.detach()).mean(1)
            else:
                chain_x = self.rf_chain.flatten(0, 1)[lead_row]
                old_mean = self.rf_old_mean.flatten(0, 1)[lead_row]
                old_sigma = self.rf_old_sigma.flatten(0, 1)[lead_row]
                old_logp = self.rf_old_logp.flatten(0, 1)[lead_row]
                mean, sigma, logp = self._rf_chain_stats(latent, chain_x)
                r_lead = torch.exp(torch.clamp(logp - old_logp, -20.0, 20.0))
                sur_lead = torch.max(
                    -adv * r_lead, -adv * torch.clamp(r_lead, 1.0 - self.clip_param, 1.0 + self.clip_param)
                )
                surrogate = surrogate.index_put((lead_idx,), sur_lead)
                ent_lead = (0.5 + 0.5 * math.log(2 * math.pi) + sigma.log()).sum(dim=(1, 2))
                entropy_rows = entropy_rows.index_put((lead_idx,), entropy_coef[lead_idx] * ent_lead)
                entropy_raw = entropy_raw.index_put((lead_idx,), ent_lead.detach())
                with torch.no_grad():
                    ratio[lead_idx] = r_lead.detach()
                    kl[lead_idx] = _row_gaussian_kl(mean.detach(), sigma.detach(), old_mean, old_sigma).sum(-1)

        # Leader on-policy rows: PolicyFlow
        if bool(lead_on.any()) and self.leader_algo == "policyflow":
            lead_idx = torch.nonzero(lead_on).view(-1)
            pf_row = self._leader_row(src[lead_idx])
            x0 = self.pf_x0.flatten(0, 1)[pf_row]
            phi = self.pf_phi.flatten(0, 1)[pf_row]
            noise = self.pf_noise.flatten(0, 1)[pf_row]
            old_std = self.pf_old_std.flatten(0, 1)[pf_row]
            old_logp = self.pf_old_logp.flatten(0, 1)[pf_row]
            lead_obs = obs[lead_idx]
            latent = flow.get_latent(lead_obs)
            t = grid[torch.randint(0, grid.shape[0], (lead_idx.numel(),), device=self.device)]
            tc = t[:, None]
            x_t = (1.0 - tc) * x0 + tc * phi
            with torch.no_grad():
                v_old = old.velocity(old.get_latent(lead_obs), x_t, t)
            v = flow.velocity(latent, x_t, t)
            dv = v - v_old
            std = std_new.expand_as(dv)
            log_ratio = torch.distributions.Normal(dv, std).log_prob(noise).sum(-1) - old_logp
            r_lead = torch.exp(log_ratio)
            adv = advantages[lead_idx]
            sur_lead = torch.max(
                -adv * r_lead, -adv * torch.clamp(r_lead, 1.0 - self.clip_param, 1.0 + self.clip_param)
            )
            surrogate = surrogate.index_put((lead_idx,), sur_lead)
            brownian = F.mse_loss((1.0 - tc) * v, x_t - tc * v_old)
            ent_lead = (0.5 + 0.5 * math.log(2 * math.pi) + std_new.log()).sum().expand(lead_idx.numel())
            entropy_rows = entropy_rows.index_put((lead_idx,), entropy_coef[lead_idx] * ent_lead)
            entropy_raw = entropy_raw.index_put((lead_idx,), ent_lead.detach())
            with torch.no_grad():
                ratio[lead_idx] = r_lead.detach()
                kl[lead_idx] = _row_gaussian_kl(dv.detach(), std.detach(), torch.zeros_like(dv), old_std)

        # Relabeled rows: marginal ratio over K old-leader latents, weighted by mu (skipped at lambda_off = 0)
        if bool(off.any()):
            off_idx = torch.nonzero(off).view(-1)
            r = off_idx.numel()
            off_obs = obs[off_idx]
            adv = advantages[off_idx]
            with torch.no_grad():
                ratio[off_idx] = 1.0  # kl stays 0; both overwritten by the marginal ratio below
            if self.lambda_off > 0.0 or self.fm_coef > 0.0:
                latent = flow.get_latent(off_obs)
                if self.off_detach_encoder:
                    latent = latent.detach()
        if bool(off.any()) and self.lambda_off > 0.0:
            pos = rows["off_pos"][idx[off_idx]]
            z = rows["off_z"][pos]
            phi_old = rows["off_phi"][pos]
            ell = rows["off_ell"][pos]
            mu = rows["off_log_mu"][pos].exp()
            _, k, a = z.shape
            t = grid[torch.randint(0, grid.shape[0], (r,), device=self.device)]
            x_t = (1.0 - t.view(r, 1, 1)) * z + t.view(r, 1, 1) * phi_old
            x_flat = x_t.reshape(r * k, a)
            t_rep = t.repeat_interleave(k)
            with torch.no_grad():
                v_old = old.velocity(old.get_latent(off_obs).repeat_interleave(k, 0), x_flat, t_rep)
            dv = (flow.velocity(latent.repeat_interleave(k, 0), x_flat, t_rep) - v_old).view(r, k, a)
            target = actions[off_idx].unsqueeze(1) - phi_old
            log_num = torch.distributions.Normal(dv, std_new.expand_as(dv)).log_prob(target).sum(-1)
            log_rho = torch.logsumexp(log_num, dim=1) - torch.logsumexp(ell, dim=1)
            rho = torch.exp(torch.clamp(log_rho, -20.0, 20.0))
            clipped = torch.clamp(rho, 1.0 - self.clip_param, 1.0 + self.clip_param)
            sur_off = torch.max(-adv * rho, -adv * clipped)
            if self.dual_clip is not None:
                sur_off = torch.where(adv < 0, torch.min(sur_off, -self.dual_clip * adv), sur_off)
            sur_off = self.lambda_off * mu * sur_off
            surrogate = surrogate.index_put((off_idx,), sur_off)
            entropy_rows = entropy_rows.index_put((off_idx,), torch.zeros_like(sur_off))
            off_log_rho, off_mu = log_rho.detach(), mu
            with torch.no_grad():
                ratio[off_idx] = rho.detach()
                kl[off_idx] = (rho.detach() - 1.0) - log_rho.detach()
        if bool(off.any()) and self.fm_coef > 0.0:
            # fresh x0 (not the cached off_z): regress the velocity onto the straight path x0 -> follower action
            a_off = actions[off_idx]
            x0 = torch.randn_like(a_off)
            t_fm = grid[torch.randint(0, grid.shape[0], (r,), device=self.device)]
            tc = t_fm[:, None]
            x_t_fm = (1.0 - tc) * x0 + tc * a_off
            v_fm = flow.velocity(latent, x_t_fm, t_fm)
            w_fm = torch.exp(adv / self.fm_beta).clamp(max=self.fm_w_max).detach()
            fm_loss = (w_fm[:, None] * (v_fm - (a_off - x0)).square()).mean()
            fm_w_mean = w_fm.mean()

        if self.use_clipped_value_loss:
            value_clipped = old_values + (values - old_values).clamp(-self.clip_param, self.clip_param)
            value_rows = torch.max((values - returns).pow(2), (value_clipped - returns).pow(2)).view(-1)
        else:
            value_rows = (returns - values).pow(2).view(-1)
        loss = (
            surrogate.mean()
            + self.value_loss_coef * value_rows.mean()
            - entropy_rows.mean()
            + self.brownian_reg_coef * brownian
            + self.fm_coef * fm_loss
        )
        # per-component shares of the actor part of ``loss`` (encoder-gradient metric)
        num_rows = surrogate.shape[0]
        loss_follower = (surrogate[gauss].sum() - entropy_rows[gauss].sum()) / num_rows
        loss_leader_on = (
            (surrogate[lead_on].sum() - entropy_rows[lead_on].sum()) / num_rows + self.brownian_reg_coef * brownian
        )
        loss_leader_off = surrogate[off].sum() / num_rows + self.fm_coef * fm_loss
        with torch.no_grad():
            inside = ((ratio - 1.0).abs() <= self.clip_param).float()
            on = ~off
        return {
            "loss": loss,
            "loss_follower": loss_follower,
            "loss_leader_on": loss_leader_on,
            "loss_leader_off": loss_leader_off,
            "leader_stats": leader_stats,
            "surrogate": surrogate.mean(),
            "value": value_rows.mean(),
            "entropy": entropy_raw[on].mean() if bool(on.any()) else entropy_raw.new_zeros(()),
            "surrogate_rows": surrogate,
            "value_rows": value_rows,
            "entropy_rows": entropy_rows,
            "entropy_raw": entropy_raw,
            "kl": kl,
            "inside_clip": inside,
            "ratio": ratio,
            "params": tuple(p.detach() for p in params),
            "block": block,
            "off": off,
            "brownian": brownian,
            "fm_loss": fm_loss.detach(),
            "fm_w_mean": fm_w_mean.detach(),
            "off_log_rho": off_log_rho,
            "off_mu": off_mu,
        }

    def _grad_diagnostic(self, terms: dict) -> dict[str, torch.Tensor]:
        raise NotImplementedError("SAPGFlow does not support log_off_policy_grads_every.")

    def update(self) -> dict[str, float]:
        """SAPG.update with one joint clip over actor + critic + flow leader and the flow-leader stats."""
        assert self._rows is not None, "compute_returns() must run before update()."
        assert self.log_off_policy_grads_every <= 0, "SAPGFlow does not support log_off_policy_grads_every."
        st = self.storage
        rows = self._rows
        flat = {
            "observations": st.observations.flatten(0, 1),
            "actions": st.actions.flatten(0, 1),
            "actions_log_prob": st.actions_log_prob.flatten(0, 1),
        }
        old_params = tuple(p.flatten(0, 1)[rows["src"]].clone() for p in st.distribution_params)  # type: ignore

        num_rows = rows["src"].shape[0]
        mini_batch_size = num_rows // self.num_mini_batches
        indices = torch.randperm(self.num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        sums: dict[str, torch.Tensor] = {}

        def acc(key: str, value: torch.Tensor | float) -> None:
            value = torch.as_tensor(value, device=self.device, dtype=torch.float32).detach()
            sums[key] = sums[key] + value if key in sums else value

        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        flow_params = self._leader_params()
        grad_norms: dict[str, list[torch.Tensor]] = {"actor": [], "critic": [], "flow": [], "combined": []}
        epoch_kls: list[torch.Tensor] = []
        self.last_minibatch_debug = None
        num_steps_run, stop, approx_kl_first, approx_kl_last = 0, False, 0.0, 0.0

        for epoch in range(self.num_learning_epochs):
            if stop:
                break
            num_epoch_steps = 0
            kl_acc = torch.zeros(5, device=self.device)
            for i in range(self.num_mini_batches):
                idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                if epoch == 0 and i < self.div_minibatches:
                    self._accumulate_block_diversity(flat["observations"][rows["src"][idx]], sums)
                terms = self.minibatch_losses(idx, flat, tuple(p[idx] for p in old_params))
                if epoch == 0 and i == 0:
                    self.last_minibatch_debug = {
                        k: v for k, v in terms.items() if isinstance(v, torch.Tensor) and not k.startswith("loss")
                    }
                    self.last_minibatch_debug["idx"] = idx

                off, kl = terms["off"], terms["kl"]
                leader_on = (~off) & (terms["block"] == 0)
                kl_acc += torch.stack([
                    kl.mean(),
                    kl[leader_on].sum(),
                    leader_on.sum().float(),
                    kl[off].sum(),
                    off.sum().float(),
                ])
                if self.kl_reference == "previous_epoch":
                    for stored, new in zip(old_params, terms["params"]):
                        stored[idx] = new

                measure_encoder = num_steps_run < self.encoder_grad_minibatches
                if measure_encoder:
                    enc_grads = [
                        self._encoder_grad(terms[key])
                        for key in ("loss_follower", "loss_leader_on", "loss_leader_off")
                    ]
                self.optimizer.zero_grad()
                terms["loss"].backward()
                if measure_encoder:
                    # per rank, before the all-reduce and the clip
                    with torch.no_grad():
                        g_follower, g_on, g_off = enc_grads
                        g_leader = g_on + g_off
                        g_total = torch.cat([
                            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                            for p in self._encoder_params
                        ])
                        n_f, n_l = g_follower.norm(), g_leader.norm()
                        acc("_genc/follower", n_f)
                        acc("_genc/leader_on", g_on.norm())
                        acc("_genc/leader_off", g_off.norm())
                        acc("_genc/leader_total", n_l)
                        acc("_genc/leader_over_follower", n_l / (n_f + 1e-12))
                        acc("_genc/cos_leader_follower", (g_leader * g_follower).sum() / (n_l * n_f).clamp_min(1e-12))
                        acc("_genc/total", g_total.norm())
                        acc("_genc_count", 1.0)
                if self.is_multi_gpu:
                    self.reduce_parameters()
                with torch.no_grad():
                    actor_sq = self._safe_grad_norm_sq(actor_params)
                    critic_sq = self._safe_grad_norm_sq(critic_params)
                    flow_sq = self._safe_grad_norm_sq(flow_params)
                    grad_norms["actor"].append(actor_sq.sqrt())
                    grad_norms["critic"].append(critic_sq.sqrt())
                    grad_norms["flow"].append(flow_sq.sqrt())
                    grad_norms["combined"].append(self._safe_grad_norm_sq(self._all_params).sqrt())
                nn.utils.clip_grad_norm_(self._all_params, self.max_grad_norm)
                self.optimizer.step()
                num_steps_run += 1
                num_epoch_steps += 1

                if self.target_kl is not None:
                    with torch.no_grad():
                        on_ratio = terms["ratio"][~off].clamp_min(1e-12)
                        approx_kl = ((on_ratio - 1.0) - on_ratio.log()).mean()
                        if self.is_multi_gpu:
                            torch.distributed.all_reduce(approx_kl, op=torch.distributed.ReduceOp.SUM)
                            approx_kl /= self.gpu_world_size
                    approx_kl_last = approx_kl.item()
                    if num_steps_run == 1:
                        approx_kl_first = approx_kl_last
                    stop = approx_kl_last > self.target_kl

                acc("value", terms["value"])
                acc("surrogate", terms["surrogate"])
                acc("entropy", terms["entropy"])
                acc("_brownian", terms["brownian"])
                acc("_fm_loss", terms["fm_loss"])
                acc("_fm_w_mean", terms["fm_w_mean"])
                for key, value in terms["leader_stats"].items():
                    acc(f"_ls/{key}", value)
                    acc(f"_ls_count/{key}", 1.0)
                with torch.no_grad():
                    inside = terms["inside_clip"]
                    acc("_inside_on", inside[~off].sum())
                    acc("_count_on", (~off).sum())
                    acc("_inside_off", inside[off].sum())
                    acc("_ratio_off", terms["ratio"][off].sum())
                    acc("_count_off", off.sum())
                    acc("_leader_kl", kl[leader_on].sum())
                    acc("_count_leader", leader_on.sum())
                if stop:
                    break

            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_acc, op=torch.distributed.ReduceOp.SUM)
                kl_acc[0] /= self.gpu_world_size
            kl_all = kl_acc[0] / max(num_epoch_steps, 1)
            kl_leader = kl_acc[1] / kl_acc[2].clamp(min=1.0)
            kl_off = kl_acc[3] / kl_acc[4].clamp(min=1.0)
            epoch_kls.append(torch.stack([kl_all, kl_leader, kl_off]))
            self._adapt_learning_rate(kl_all if self.kl_rows == "all" else kl_leader)

        num_updates = num_steps_run
        out = {k: v.item() for k, v in sums.items()}
        loss_dict: dict[str, float] = {
            "value": out["value"] / num_updates,
            "surrogate": out["surrogate"] / num_updates,
            "entropy": out["entropy"] / num_updates,
            "SAPGFlow/brownian": out["_brownian"] / num_updates,
            "SAPGFlow/fm_loss": out["_fm_loss"] / num_updates,
            "SAPGFlow/fm_w_mean": out["_fm_w_mean"] / num_updates,
        }
        kls = torch.stack(epoch_kls).mean(dim=0).tolist()
        loss_dict["SAPG/kl_all"], loss_dict["SAPG/kl_leader"] = kls[0], kls[1]
        if out["_count_off"] > 0 and self.lambda_off > 0.0:
            loss_dict["SAPG/kl_relabeled"] = kls[2]
            loss_dict["SAPG/clip_inside_off"] = out["_inside_off"] / out["_count_off"]
            loss_dict["SAPG/ratio_off_mean"] = out["_ratio_off"] / out["_count_off"]
            loss_dict["SAPGFlow/off_rho_mean"] = out["_ratio_off"] / out["_count_off"]
            loss_dict["SAPGFlow/off_clip_inside"] = out["_inside_off"] / out["_count_off"]
        loss_dict["SAPG/clip_inside_on"] = out["_inside_on"] / max(out["_count_on"], 1.0)
        loss_dict["SAPGFlow/leader_kl"] = out["_leader_kl"] / max(out["_count_leader"], 1.0)
        for key in out:
            if key.startswith("_ls/"):
                name = key[len("_ls/") :]
                loss_dict[f"SAPGFlow/{name}"] = out[key] / out[f"_ls_count/{name}"]
        if "_genc_count" in out:
            for key in out:
                if key.startswith("_genc/"):
                    loss_dict[f"grad_enc/{key[len('_genc/') :]}"] = out[key] / out["_genc_count"]
        with torch.no_grad():
            if self.leader_algo == "policyflow":
                leader_std = self.flow_leader.std.mean().item()
                loss_dict["SAPGFlow/leader_noise_std"] = leader_std
                loss_dict["SAPG/block0_action_std"] = leader_std
            elif self.leader_algo == "reinflow":
                sigma = self.rf_old_sigma.mean().item()
                loss_dict["SAPGFlow/reinflow_sigma_mean"] = sigma
                loss_dict["SAPG/block0_action_std"] = sigma
            std = st.distribution_params[1].mean(dim=-1).mean(dim=0).view(-1)  # type: ignore[index]
            for j in range(1, self.num_blocks):
                loss_dict[f"SAPG/block{j}_action_std"] = std[self.env_block_ids == j].mean().item()
        if self.target_kl is not None:
            loss_dict["kl_stop/mini_batch_updates"] = float(num_steps_run)
            loss_dict["kl_stop/stopped"] = float(num_steps_run < self.num_learning_epochs * self.num_mini_batches)
            loss_dict["kl_stop/approx_kl_first"] = approx_kl_first
            loss_dict["kl_stop/approx_kl_last"] = approx_kl_last
        loss_dict.update(self._block_diversity_logs(out))
        loss_dict.update(self._relabel_stats)
        loss_dict.update(self._flow_stats)
        loss_dict.update(self.episode_metrics())
        for component, values in grad_norms.items():
            stacked = torch.stack(values)
            loss_dict[f"grad_norm/{component}_mean"] = stacked.mean().item()
            loss_dict[f"grad_norm/{component}_var"] = stacked.var(unbiased=False).item() if len(values) > 1 else 0.0

        self.storage.clear()
        self._rows = None
        self._relabel_stats = {}
        self._flow_stats = {}
        self._num_updates += 1
        return loss_dict

    def _encoder_grad(self, component: torch.Tensor) -> torch.Tensor:
        """Flat gradient of one loss component w.r.t. the shared encoder (zeros where it does not reach)."""
        if not component.requires_grad:
            return torch.cat([torch.zeros_like(p).reshape(-1) for p in self._encoder_params])
        grads = torch.autograd.grad(component, self._encoder_params, retain_graph=True, allow_unused=True)
        return torch.cat([
            (g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, self._encoder_params)
        ])

    @staticmethod
    def _safe_grad_norm_sq(params: list[nn.Parameter]) -> torch.Tensor:
        sq = [p.grad.detach().pow(2).sum() for p in params if p.grad is not None]
        return torch.stack(sq).sum() if sq else torch.zeros(())

    # ------------------------------------------------------------------------------------------------
    # Modes, multi-GPU, checkpoints
    # ------------------------------------------------------------------------------------------------

    def train_mode(self) -> None:
        super().train_mode()
        self.flow_leader.train()
        if self.rf_noise_head is not None:
            self.rf_noise_head.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.flow_leader.eval()
        if self.rf_noise_head is not None:
            self.rf_noise_head.eval()

    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        for tensor in chain(self._leader_params(), self.flow_leader.buffers()):
            torch.distributed.broadcast(tensor.data, src=0)

    def reduce_parameters(self) -> None:
        """All-reduce every gradient of the deduplicated actor + critic + flow-leader parameter list."""
        # None grads become zeros so every rank flattens the same layout
        for p in self._all_params:
            if p.grad is None and p.requires_grad:
                p.grad = torch.zeros_like(p)
        params = [p for p in self._all_params if p.grad is not None]
        all_grads = torch.cat([p.grad.view(-1) for p in params])  # type: ignore[union-attr]
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for p in params:
            numel = p.numel()
            p.grad.data.copy_(all_grads[offset : offset + numel].view_as(p.grad.data))  # type: ignore[union-attr]
            offset += numel

    def save(self) -> dict:
        saved = super().save()
        saved["flow_leader_state_dict"] = self.flow_leader.state_dict()
        saved["sapg"]["num_latents"] = self.num_latents
        saved["sapg"]["fm_coef"] = self.fm_coef
        saved["sapg"]["fm_beta"] = self.fm_beta
        saved["sapg"]["fm_w_max"] = self.fm_w_max
        saved["sapg"]["flow_leader"] = True
        saved["sapg"]["leader_algo"] = self.leader_algo
        if self.rf_noise_head is not None:
            saved["reinflow_noise_head_state_dict"] = self.rf_noise_head.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        assert "flow_leader_state_dict" in loaded_dict, "not a SAPGFlow checkpoint (no flow_leader_state_dict)."
        result = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("actor"):
            self.flow_leader.load_state_dict(loaded_dict["flow_leader_state_dict"], strict=strict)
            if self.rf_noise_head is not None:
                if "reinflow_noise_head_state_dict" in loaded_dict:
                    self.rf_noise_head.load_state_dict(loaded_dict["reinflow_noise_head_state_dict"], strict=strict)
                else:
                    assert not strict, "checkpoint has no reinflow_noise_head_state_dict (load with strict=False)."
        self._sync_flow_old()
        return result

    # ------------------------------------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------------------------------------

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> SAPGFlow:
        """Gaussian actor / critic from ``cfg['actor'/'critic']``, the flow leader from ``cfg['flow_actor']``."""
        alg_class: type[SAPGFlow] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore
        assert cfg.get("flow_actor"), "SAPGFlow needs a 'flow_actor' model cfg."
        flow_cfg = dict(cfg["flow_actor"])
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        # SAPG rejects RND / symmetry; resolving them keeps the keys the runner reads
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)
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
        alg = alg_class(
            actor, critic, storage, flow_cfg=flow_cfg, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"]
        )
        print(f"Flow Leader Model: {alg.flow_leader}")
        alg.compile(cfg.get("torch_compile_mode"))
        return alg
