# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SAPG: Split and Aggregate Policy Gradients (Singla et al. 2024, arXiv 2407.20230), leader-follower variant.

Port of play2perfect's rl_games implementation (``use_others_experience: lf``, ``expl_type: mixed_expl_learn_param``,
``expl_reward_type: entropy``, ``fixed_sigma: coef_cond``) onto rsl_rl PPO with MLP / MLPEncoder models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl.models.block_conditioning import attach_block_conditioning, block_context
from rsl_rl.storage import RolloutStorage


class SAPG(PPO):
    """PPO over ``num_blocks`` contiguous env blocks per rank: block 0 = leader, 1..M-1 = followers.

    - Actor and critic are conditioned on a learned per-block ``phi_j`` (separate tables) and the actor's
      exploration std is per block. Deployment / inference / export = the leader.
    - Each iteration ``off_policy_ratio`` whole follower block(s), drawn at random (same draw on every rank), are
      relabeled as leader data: 1-step TD target ``y = r + gamma V_L(s') (1 - done)``, advantage ``y - V_L(s)``,
      old value ``V_L(s)``, behavior log-prob / distribution = the follower's. They are appended to the
      on-policy rows and shuffled into the same minibatches.
    - One loss = mean over every row (all blocks + relabeled): PPO clip surrogate, clipped value loss, and a per-row
      entropy bonus (leader and relabeled rows: ``leader_entropy_coef``; follower j: ``sapg_entropy_scale * 0.5 *
      j / (M - 1)``). PPO's ``entropy_coef`` is not used.
    - Advantages are normalized jointly over all rows (across ranks); the adaptive LR reads the epoch-mean KL and
      steps once per epoch.
    """

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_blocks: int = 6,
        phi_dim: int = 32,
        per_block_std: bool = True,
        off_policy_ratio: float = 1.0,
        sapg_entropy_scale: float = 0.002,
        leader_entropy_coef: float = 0.0,
        kl_rows: str = "all",
        kl_reference: str = "previous_epoch",
        log_off_policy_grads_every: int = 0,
        relabel_chunk_size: int = 65536,
        target_kl: float | None = None,
        off_policy_sampling: str = "block",
        shared_layers: int | None = None,
        normalize_advantage_per_block: bool = False,
        follower_entropy_coef: float | None = None,
        **kwargs: object,
    ) -> None:
        """``num_blocks`` M, ``phi_dim`` d; everything else is PPO's (``kwargs``).

        ``kl_rows``: ``"all"`` (every row of the minibatch, play2perfect) or ``"leader"`` (leader on-policy rows)
        drive the adaptive LR. ``kl_reference``: ``"previous_epoch"`` (play2perfect: a row's stored distribution is
        overwritten with the one it was just evaluated at, so epoch k>0 measures the step since epoch k-1) or
        ``"rollout"`` (rsl_rl: always against the behavior distribution).

        ``shared_layers``: actor and critic share only their first ``shared_layers`` Linears (and the obs normalizer);
        the rest are per-block heads (``None`` = one MLP for every block). ``follower_entropy_coef``: one constant
        entropy coef for every block (incl. block 0), replacing the leader / ladder schedule.
        """
        assert num_blocks >= 1, f"num_blocks must be >= 1, got {num_blocks}."
        assert phi_dim >= 0, f"phi_dim must be >= 0, got {phi_dim}."
        assert off_policy_ratio >= 0.0, f"off_policy_ratio must be >= 0, got {off_policy_ratio}."
        assert kl_rows in ("all", "leader"), f"kl_rows must be 'all' or 'leader', got {kl_rows!r}."
        assert kl_reference in ("previous_epoch", "rollout"), f"unknown kl_reference {kl_reference!r}."
        assert off_policy_sampling in ("block", "uniform"), f"unknown off_policy_sampling {off_policy_sampling!r}."
        assert float(kwargs.get("eval_env_fraction", 0.0) or 0.0) <= 0.0, (
            "SAPG grades / deploys on the leader block; the eval env pool (eval_env_fraction > 0) is not supported."
        )
        assert not kwargs.get("symmetry_cfg"), "SAPG does not support symmetry augmentation."
        assert not kwargs.get("rnd_cfg"), "SAPG does not support RND."
        assert not (actor.is_recurrent or critic.is_recurrent), "SAPG supports feedforward models only."
        # Conditioning parameters must exist before PPO builds the optimizer over the model parameters.
        attach_block_conditioning(actor, num_blocks, phi_dim, per_block_std=per_block_std, shared_layers=shared_layers)
        attach_block_conditioning(critic, num_blocks, phi_dim, shared_layers=shared_layers)
        super().__init__(actor, critic, storage, **kwargs)
        assert self.noise_scale_tracker is None, "SAPG does not support the gradient-noise-scale tracker."
        if self.entropy_coef:
            print(
                f"[SAPG] entropy_coef={self.entropy_coef} is ignored: leader entropy coef = {leader_entropy_coef},"
                f" followers = sapg_entropy_scale schedule."
            )

        self.num_blocks = int(num_blocks)
        self.phi_dim = int(phi_dim)
        self.per_block_std = bool(per_block_std)
        self.shared_layers = None if shared_layers is None else int(shared_layers)
        self.normalize_advantage_per_block = bool(normalize_advantage_per_block)
        self.follower_entropy_coef = None if follower_entropy_coef is None else float(follower_entropy_coef)
        self.num_off_policy_blocks = min(self.num_blocks - 1, int(off_policy_ratio))
        # "block": whole follower block(s) per iteration (play2perfect); "uniform": the same number of rows drawn
        # uniformly over every follower transition (SAPG paper, Algorithm 1)
        self.off_policy_sampling = off_policy_sampling
        self.sapg_entropy_scale = float(sapg_entropy_scale)
        self.leader_entropy_coef = float(leader_entropy_coef)
        self.kl_rows = kl_rows
        self.kl_reference = kl_reference
        self.log_off_policy_grads_every = int(log_off_policy_grads_every)
        self.relabel_chunk_size = int(relabel_chunk_size)
        # cleanRL / ManiSkill early stop: approx KL of the on-policy rows before each step; above it the update ends
        self.target_kl = None if target_kl is None else float(target_kl)

        num_envs = storage.num_envs
        assert num_envs % self.num_blocks == 0, (
            f"num_envs={num_envs} is not a multiple of num_blocks={self.num_blocks}."
        )
        # Contiguous equal blocks (the formula would also tolerate uneven ones).
        self.env_block_ids = torch.arange(num_envs, device=self.device) * self.num_blocks // num_envs
        self.leader_env_mask = self.env_block_ids == 0
        self.num_leader_envs = int(self.leader_env_mask.sum())
        # play2perfect: linspace(0.5, 0, M)[b] * scale with its leader last; ours is block 0, so follower j gets
        # 0.5 * j / (M - 1) * scale (the largest bonus on block M-1).
        if self.follower_entropy_coef is not None:
            self.block_entropy_coefs = [self.follower_entropy_coef] * self.num_blocks
        else:
            self.block_entropy_coefs = [self.leader_entropy_coef] + [
                self.sapg_entropy_scale * 0.5 * j / (self.num_blocks - 1) for j in range(1, self.num_blocks)
            ]
        self._env_entropy_coef = torch.tensor(self.block_entropy_coefs, device=self.device)[self.env_block_ids]

        num_steps = storage.num_transitions_per_env
        self._time_outs = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self._rows: dict[str, torch.Tensor] | None = None  # training rows (on-policy + relabeled), see compute_returns
        self._relabel_stats: dict[str, float] = {}
        self._num_updates = 0

        # Per-block episode statistics, accumulated over one iteration (see ``episode_metrics``).
        self.episode_success_fn = None  # optional callable -> bool [num_envs] success flag of the step just taken
        self._ep_return = torch.zeros(num_envs, device=self.device)
        self._ep_length = torch.zeros(num_envs, device=self.device)
        self._blk_return_sum = torch.zeros(self.num_blocks, device=self.device)
        self._blk_length_sum = torch.zeros(self.num_blocks, device=self.device)
        self._blk_success_sum = torch.zeros(self.num_blocks, device=self.device)
        self._blk_count = torch.zeros(self.num_blocks, device=self.device)

        self.last_minibatch_debug: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------------------------------------

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Every env acts with its own block's phi and std."""
        with block_context(self.env_block_ids, self.actor, self.critic):
            return super().act(obs)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record the step (plus time-outs and per-block episode stats)."""
        step = self.storage.step
        if "time_outs" in extras:
            self._time_outs[step].copy_(extras["time_outs"].view(-1, 1).to(self.device).float())
        else:
            self._time_outs[step].zero_()
        self._track_episodes(rewards, dones)
        super().process_env_step(obs, rewards, dones, extras)

    def _track_episodes(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        self._ep_return += rewards.view(-1).to(self.device).float()
        self._ep_length += 1
        done = dones.view(-1).to(self.device) > 0
        if not bool(done.any()):
            return
        blocks = self.env_block_ids[done]
        self._blk_return_sum.index_add_(0, blocks, self._ep_return[done])
        self._blk_length_sum.index_add_(0, blocks, self._ep_length[done])
        self._blk_count.index_add_(0, blocks, torch.ones_like(self._ep_return[done]))
        if self.episode_success_fn is not None:
            success = self.episode_success_fn().view(-1).to(self.device).float()
            self._blk_success_sum.index_add_(0, blocks, success[done])
        self._ep_return[done] = 0.0
        self._ep_length[done] = 0.0

    def episode_metrics(self) -> dict[str, float]:
        """``SAPG/*`` episode statistics of the episodes finished since the last call (then reset)."""
        stats = torch.stack([self._blk_return_sum, self._blk_length_sum, self._blk_success_sum, self._blk_count])
        if self.is_multi_gpu:
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        for buf in (self._blk_return_sum, self._blk_length_sum, self._blk_success_sum, self._blk_count):
            buf.zero_()
        ret, length, success, count = stats.cpu()
        metrics: dict[str, float] = {}
        if count[0] > 0:
            metrics["SAPG/leader_return"] = float(ret[0] / count[0])
            metrics["SAPG/leader_episode_length"] = float(length[0] / count[0])
            if self.episode_success_fn is not None:
                metrics["SAPG/leader_success"] = float(success[0] / count[0])
        if self.num_blocks > 1 and count[1:].sum() > 0 and self.episode_success_fn is not None:
            metrics["SAPG/follower_success"] = float(success[1:].sum() / count[1:].sum())
        for j in range(self.num_blocks):
            if count[j] > 0:
                metrics[f"SAPG/block{j}_return"] = float(ret[j] / count[j])
                if self.episode_success_fn is not None:
                    metrics[f"SAPG/block{j}_success"] = float(success[j] / count[j])
        return metrics

    # ------------------------------------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------------------------------------

    def compute_returns(self, obs: TensorDict) -> None:
        """Per-block GAE, the relabeled follower block(s), then one joint advantage normalization."""
        st = self.storage
        with block_context(self.env_block_ids, self.actor, self.critic):
            last_values = self.critic(obs).detach()
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values

        num_on = st.num_transitions_per_env * st.num_envs
        rows = {
            "src": torch.arange(num_on, device=self.device),
            "block": self.env_block_ids.repeat(st.num_transitions_per_env),
            "off": torch.zeros(num_on, dtype=torch.bool, device=self.device),
            "values": st.values.flatten(0, 1),
            "returns": st.returns.flatten(0, 1),
            "advantages": st.advantages.flatten(0, 1),
        }
        off_blocks = self._draw_off_policy_blocks()
        if off_blocks:
            rel = self._relabel(obs, off_blocks)
            if self.off_policy_sampling == "uniform":
                num_keep = self.num_off_policy_blocks * self.num_leader_envs * st.num_transitions_per_env
                keep = torch.randperm(rel["src"].shape[0], device=self.device)[:num_keep]
                rel = {key: value[keep] for key, value in rel.items()}
                self._relabel_stats["SAPG/relabel_rows"] = float(num_keep)
                self._relabel_stats["SAPG/relabel_adv_mean"] = rel["advantages"].mean().item()
            for key, value in rel.items():
                rows[key] = torch.cat([rows[key], value], dim=0)
        rows["entropy_coef"] = torch.cat([
            self._env_entropy_coef.repeat(st.num_transitions_per_env),
            torch.full((rows["src"].shape[0] - num_on,), self.block_entropy_coefs[0], device=self.device),
        ])
        if not self.normalize_advantage_per_mini_batch:
            rows["advantages"] = self._normalize(rows["advantages"])
            st.advantages = rows["advantages"][:num_on].view_as(st.advantages).clone()
        self._rows = rows

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """``(x - mean) / (std + 1e-8)`` with the moments over every rank's rows (play2perfect dist_mean_var_count)."""
        if not self.is_multi_gpu:
            return (x - x.mean()) / (x.std() + 1e-8)
        stats = torch.stack([x.sum(), torch.tensor(float(x.numel()), device=x.device)])
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        mean = stats[0] / stats[1]
        sq = (x - mean).pow(2).sum()
        torch.distributed.all_reduce(sq, op=torch.distributed.ReduceOp.SUM)
        std = torch.sqrt(sq / (stats[1] - 1.0))
        return (x - mean) / (std + 1e-8)

    def _draw_off_policy_blocks(self) -> list[int]:
        """``num_off_policy_blocks`` distinct follower blocks, drawn on rank 0 and broadcast (same on every rank)."""
        if self.num_off_policy_blocks <= 0:
            return []
        if self.off_policy_sampling == "uniform":
            return list(range(1, self.num_blocks))
        draw = torch.randperm(self.num_blocks - 1, device=self.device)[: self.num_off_policy_blocks] + 1
        if self.is_multi_gpu:
            torch.distributed.broadcast(draw, src=0)
        return sorted(int(b) for b in draw.tolist())

    @torch.no_grad()
    def _relabel(self, last_obs: TensorDict, blocks: list[int]) -> dict[str, torch.Tensor]:
        """Every transition of ``blocks`` as a leader row, evaluated with the pre-update leader critic."""
        st = self.storage
        num_steps, num_envs = st.num_transitions_per_env, st.num_envs
        env_ids = torch.nonzero(torch.isin(self.env_block_ids, torch.tensor(blocks, device=self.device))).view(-1)
        n = env_ids.numel()

        v_lead = torch.empty(num_steps, n, 1, device=self.device)
        steps_per_chunk = max(1, self.relabel_chunk_size // n)
        with block_context(None, self.critic):
            for t0 in range(0, num_steps, steps_per_chunk):
                t = slice(t0, min(t0 + steps_per_chunk, num_steps))
                chunk = st.observations[t][:, env_ids].flatten(0, 1)
                v_lead[t] = self.critic(chunk).view(-1, n, 1)
            v_last = self.critic(last_obs[env_ids]).view(1, n, 1)

        # Stored rewards carry the behavior critic's time-out bootstrap (gamma * V_j(s_t)); swap in the leader's.
        rewards = st.rewards[:, env_ids] + self.gamma * self._time_outs[:, env_ids] * (v_lead - st.values[:, env_ids])
        next_values = torch.cat([v_lead[1:], v_last], dim=0)
        not_done = 1.0 - st.dones[:, env_ids].float()
        returns = rewards + self.gamma * not_done * next_values
        advantages = returns - v_lead

        src = (torch.arange(num_steps, device=self.device).unsqueeze(1) * num_envs + env_ids.unsqueeze(0)).flatten()
        self._relabel_stats = {
            "SAPG/relabel_block": float(blocks[0]) if len(blocks) == 1 else -1.0,
            "SAPG/relabel_rows": float(src.numel()),
            "SAPG/relabel_adv_mean": advantages.mean().item(),
            "SAPG/relabel_value_mean": v_lead.mean().item(),
        }
        return {
            "src": src,
            "block": torch.zeros(src.numel(), dtype=torch.long, device=self.device),
            "off": torch.ones(src.numel(), dtype=torch.bool, device=self.device),
            "values": v_lead.flatten(0, 1),
            "returns": returns.flatten(0, 1),
            "advantages": advantages.flatten(0, 1),
        }

    # ------------------------------------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------------------------------------

    def minibatch_losses(self, idx: torch.Tensor, flat: dict, old_params: tuple[torch.Tensor, ...]) -> dict:
        """Per-row loss terms of the rows ``idx`` of the training set (``loss`` is what gets back-propagated)."""
        rows = self._rows
        assert rows is not None
        src = rows["src"][idx]
        block = rows["block"][idx]
        off = rows["off"][idx]
        obs = flat["observations"][src]
        actions = flat["actions"][src]
        old_log_prob = flat["actions_log_prob"][src].view(-1)
        old_values = rows["values"][idx]
        returns = rows["returns"][idx]
        advantages = rows["advantages"][idx].view(-1)
        if self.normalize_advantage_per_mini_batch:
            with torch.no_grad():
                if self.normalize_advantage_per_block:
                    normalized = torch.empty_like(advantages)
                    for j in range(self.num_blocks):
                        mask = block == j
                        if bool(mask.any()):
                            adv_j = advantages[mask]
                            std_j = adv_j.std() if adv_j.numel() > 1 else adv_j.new_zeros(())
                            normalized[mask] = (adv_j - adv_j.mean()) / (std_j + 1e-8)
                    advantages = normalized
                else:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with block_context(block, self.actor, self.critic):
            self.actor(obs, stochastic_output=True)
            log_prob = self.actor.get_output_log_prob(actions)  # type: ignore
            values = self.critic(obs)
        entropy = self.actor.output_entropy
        params = tuple(p for p in self.actor.output_distribution_params)

        # Plain PPO clip on every row; relabeled rows' ratio is pi_L / pi_follower (play2perfect: no mu).
        ratio = torch.exp(torch.clamp(log_prob - old_log_prob, -20.0, 20.0))
        surrogate = torch.max(
            -advantages * ratio, -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        )
        if self.use_clipped_value_loss:
            value_clipped = old_values + (values - old_values).clamp(-self.clip_param, self.clip_param)
            value_rows = torch.max((values - returns).pow(2), (value_clipped - returns).pow(2)).view(-1)
        else:
            value_rows = (returns - values).pow(2).view(-1)
        entropy_rows = rows["entropy_coef"][idx] * entropy
        loss = surrogate.mean() + self.value_loss_coef * value_rows.mean() - entropy_rows.mean()

        with torch.no_grad():
            kl = self.actor.get_kl_divergence(old_params, tuple(p.detach() for p in params))  # type: ignore
            inside = ((ratio - 1.0).abs() <= self.clip_param).float()
        return {
            "loss": loss,
            "surrogate": surrogate.mean(),
            "value": value_rows.mean(),
            "entropy": entropy.mean(),
            "surrogate_rows": surrogate,
            "value_rows": value_rows,
            "entropy_rows": entropy_rows,
            "entropy_raw": entropy.detach(),
            "kl": kl,
            "inside_clip": inside,
            "ratio": ratio.detach(),
            "params": tuple(p.detach() for p in params),
            "block": block,
            "off": off,
        }

    def _grad_diagnostic(self, terms: dict) -> dict[str, torch.Tensor]:
        """play2perfect LOG_OFF_POLICY_GRADS: gradients of the relabeled vs on-policy parts of the loss."""
        per_row = terms["surrogate_rows"] + self.value_loss_coef * terms["value_rows"] - terms["entropy_rows"]
        n = per_row.shape[0]
        off = terms["off"]
        leader_on = (~off) & (terms["block"] == 0)
        params = [p for p in chain(self.actor.parameters(), self.critic.parameters()) if p.requires_grad]

        def flat_grad(mask: torch.Tensor) -> torch.Tensor:
            grads = torch.autograd.grad(per_row[mask].sum() / n, params, retain_graph=True, allow_unused=True)
            return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g, p in zip(grads, params)])

        g_off, g_on, g_lead = flat_grad(off), flat_grad(~off), flat_grad(leader_on)
        cos = nn.functional.cosine_similarity
        return {
            "SAPG/grad_cos_off_on": cos(g_off, g_on, dim=0),
            "SAPG/grad_cos_off_leader_on": cos(g_off, g_lead, dim=0),
            "SAPG/grad_norm_off_over_on": g_off.norm() / (g_on.norm() + 1e-12),
        }

    def update(self) -> dict[str, float]:
        """PPO epochs over the on-policy + relabeled rows; LR adapted once per epoch."""
        assert self._rows is not None, "compute_returns() must run before update()."
        st = self.storage
        rows = self._rows
        flat = {
            "observations": st.observations.flatten(0, 1),
            "actions": st.actions.flatten(0, 1),
            "actions_log_prob": st.actions_log_prob.flatten(0, 1),
        }
        # Old distribution per row (relabeled rows: the follower's); overwritten per minibatch for "previous_epoch".
        old_params = tuple(p.flatten(0, 1)[rows["src"]].clone() for p in st.distribution_params)  # type: ignore

        num_rows = rows["src"].shape[0]
        mini_batch_size = num_rows // self.num_mini_batches
        indices = torch.randperm(self.num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
        log_grads = self.log_off_policy_grads_every > 0 and self._num_updates % self.log_off_policy_grads_every == 0
        log_grads = log_grads and bool(rows["off"].any())

        sums: dict[str, torch.Tensor] = {}

        def acc(key: str, value: torch.Tensor | float) -> None:
            value = torch.as_tensor(value, device=self.device, dtype=torch.float32).detach()
            sums[key] = sums[key] + value if key in sums else value

        grad_norms: dict[str, list[torch.Tensor]] = {"actor": [], "critic": [], "combined": []}
        epoch_kls: list[torch.Tensor] = []
        self.last_minibatch_debug = None
        num_steps_run, stop, approx_kl_first, approx_kl_last = 0, False, 0.0, 0.0

        for epoch in range(self.num_learning_epochs):
            if stop:
                break
            num_epoch_steps = 0
            # [sum over minibatch means (all rows), leader-on KL sum, leader-on count, relabeled KL sum, count]
            kl_acc = torch.zeros(5, device=self.device)
            for i in range(self.num_mini_batches):
                idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                terms = self.minibatch_losses(idx, flat, tuple(p[idx] for p in old_params))
                if epoch == 0 and i == 0:
                    self.last_minibatch_debug = {
                        k: v for k, v in terms.items() if isinstance(v, torch.Tensor) and k != "loss"
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

                if log_grads:
                    for key, value in self._grad_diagnostic(terms).items():
                        acc(key, value)
                    acc("SAPG/grad_diag_count", 1.0)

                self.optimizer.zero_grad()
                terms["loss"].backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                with torch.no_grad():
                    actor_sq = self._grad_norm_sq(self.actor.parameters())
                    critic_sq = self._grad_norm_sq(self.critic.parameters())
                    grad_norms["actor"].append(actor_sq.sqrt())
                    grad_norms["critic"].append(critic_sq.sqrt())
                    grad_norms["combined"].append((actor_sq + critic_sq).sqrt())
                # Same two calls as PPO.update (uwlab's joint-clip patch recognizes them by parameter set).
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
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
                    # the step above still applies (ManiSkill order), then the whole update ends
                    stop = approx_kl_last > self.target_kl

                acc("value", terms["value"])
                acc("surrogate", terms["surrogate"])
                acc("entropy", terms["entropy"])
                with torch.no_grad():
                    inside = terms["inside_clip"]
                    acc("_inside_on", inside[~off].sum())
                    acc("_count_on", (~off).sum())
                    acc("_inside_off", inside[off].sum())
                    acc("_ratio_off", terms["ratio"][off].sum())
                    acc("_count_off", off.sum())
                if stop:
                    break

            # Epoch KL: mean over minibatches (all rows) or over leader on-policy rows; rank-averaged.
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
        }
        kls = torch.stack(epoch_kls).mean(dim=0).tolist()
        loss_dict["SAPG/kl_all"], loss_dict["SAPG/kl_leader"] = kls[0], kls[1]
        if out["_count_off"] > 0:
            loss_dict["SAPG/kl_relabeled"] = kls[2]
            loss_dict["SAPG/clip_inside_off"] = out["_inside_off"] / out["_count_off"]
            loss_dict["SAPG/ratio_off_mean"] = out["_ratio_off"] / out["_count_off"]
        loss_dict["SAPG/clip_inside_on"] = out["_inside_on"] / max(out["_count_on"], 1.0)
        if log_grads and out.get("SAPG/grad_diag_count", 0.0) > 0:
            count = out["SAPG/grad_diag_count"]
            for key in ("SAPG/grad_cos_off_on", "SAPG/grad_cos_off_leader_on", "SAPG/grad_norm_off_over_on"):
                loss_dict[key] = out[key] / count
        with torch.no_grad():
            std = st.distribution_params[1].mean(dim=-1).mean(dim=0).view(-1)  # type: ignore[index]
            for j in range(self.num_blocks):
                loss_dict[f"SAPG/block{j}_action_std"] = std[self.env_block_ids == j].mean().item()
        if self.target_kl is not None:
            loss_dict["kl_stop/mini_batch_updates"] = float(num_steps_run)
            loss_dict["kl_stop/stopped"] = float(num_steps_run < self.num_learning_epochs * self.num_mini_batches)
            loss_dict["kl_stop/approx_kl_first"] = approx_kl_first
            loss_dict["kl_stop/approx_kl_last"] = approx_kl_last
        loss_dict.update(self._relabel_stats)
        # Episode stats ride on the loss dict (logged as ``Loss/SAPG/*``): the runner stays stock.
        loss_dict.update(self.episode_metrics())
        for component, values in grad_norms.items():
            stacked = torch.stack(values)
            loss_dict[f"grad_norm/{component}_mean"] = stacked.mean().item()
            loss_dict[f"grad_norm/{component}_var"] = stacked.var(unbiased=False).item() if len(values) > 1 else 0.0

        self.storage.clear()
        self._rows = None
        self._relabel_stats = {}
        self._num_updates += 1
        return loss_dict

    def _adapt_learning_rate(self, kl: torch.Tensor) -> None:
        """rsl_rl's adaptive rule, applied once per epoch (play2perfect ``schedule_type: standard``)."""
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        if self.gpu_global_rank == 0:
            kl_value = kl.item()
            if kl_value > self.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_value < self.desired_kl / 2.0 and kl_value > 0.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def save(self) -> dict:
        """PPO checkpoint (phi / per-block std live in the actor/critic state dicts) plus the block layout."""
        saved = super().save()
        saved["sapg"] = {
            "num_blocks": self.num_blocks,
            "phi_dim": self.phi_dim,
            "per_block_std": self.per_block_std,
            "shared_layers": self.shared_layers,
        }
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load a SAPG checkpoint with the same block layout (SAPG runs train from scratch: no plain-PPO warm start)."""
        assert "sapg" in loaded_dict, (
            "not a SAPG checkpoint (warm starting SAPG from a plain PPO checkpoint is unsupported)."
        )
        saved = loaded_dict["sapg"]
        assert saved["num_blocks"] == self.num_blocks and saved["phi_dim"] == self.phi_dim, (
            f"SAPG checkpoint has num_blocks={saved['num_blocks']}, phi_dim={saved['phi_dim']};"
            f" this run has {self.num_blocks}, {self.phi_dim}."
        )
        assert saved.get("shared_layers") == self.shared_layers, (
            f"SAPG checkpoint has shared_layers={saved.get('shared_layers')}; this run has {self.shared_layers}."
        )
        return super().load(loaded_dict, load_cfg, strict)
