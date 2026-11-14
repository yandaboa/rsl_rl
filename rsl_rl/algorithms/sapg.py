# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SAPG: Split and Aggregate Policy Gradients (Singla et al. 2024, arXiv 2407.20230), leader-follower variant.

Port of play2perfect's rl_games implementation (``use_others_experience: lf``, ``expl_type: mixed_expl_learn_param``,
``expl_reward_type: entropy``, ``fixed_sigma: coef_cond``) onto rsl_rl PPO with MLP / MLPEncoder models.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl.models.block_conditioning import (
    BlockLinear,
    SplitBlockLinear,
    _BlockStdMixin,
    attach_block_conditioning,
    block_context,
    block_parameters,
)
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_optimizer


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
    - ``off_policy_update="awr"``: relabeled rows train the leader actor with ``-clamp(exp(A / beta), w_max) log pi_L``
      instead of the clipped ratio surrogate (their value / entropy terms are unchanged).
    - ``independent_blocks``: M fully separate PPO learners that only exchange the relabeled rows: per-block
      parameters, observation normalizers, loss means, advantage normalization, grad clip and adaptive LR.
    """

    allow_uneven_blocks: bool = False
    """Inference-only escape hatch (rollout probes): num_envs need not be a multiple of num_blocks."""

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
        block_layers: list[int] | None = None,
        div_minibatches: int = 2,
        freeze_shared_layers: bool = False,
        block_init_std: list[float] | None = None,
        critic_shared_layers: int | None = None,
        critic_phi_dim: int | None = None,
        off_policy_update: str = "is",
        awr_beta: float = 1.0,
        awr_w_max: float = 10.0,
        awr_advantage: str = "leader",
        off_policy_coef: float = 1.0,
        independent_blocks: bool = False,
        **kwargs: object,
    ) -> None:
        """``num_blocks`` M, ``phi_dim`` d; everything else is PPO's (``kwargs``).

        ``kl_rows``: ``"all"`` (every row of the minibatch, play2perfect) or ``"leader"`` (leader on-policy rows)
        drive the adaptive LR. ``kl_reference``: ``"previous_epoch"`` (play2perfect: a row's stored distribution is
        overwritten with the one it was just evaluated at, so epoch k>0 measures the step since epoch k-1) or
        ``"rollout"`` (rsl_rl: always against the behavior distribution).

        ``shared_layers``: actor and critic share only their first ``shared_layers`` Linears (and the obs normalizer);
        the rest are per-block heads (``None`` = one MLP for every block). ``block_layers``: instead, exactly these
        Linear indices (0 = first, negative from the end) are per-block. ``follower_entropy_coef``: one constant
        entropy coef for every block (incl. block 0), replacing the leader / ladder schedule.
        ``div_minibatches``: the first N minibatches of epoch 0 log ``block_div/*`` (pairwise block action-mean
        distances; 0 = off). ``critic_shared_layers`` / ``critic_phi_dim``: the critic's layout when it differs from
        the actor's (``None`` = same as the actor; ``block_layers`` always applies to both).

        ``off_policy_update``: ``"is"`` (clipped ratio surrogate on relabeled rows) or ``"awr"``. ``off_policy_coef``
        scales the relabeled rows' loss. ``independent_blocks`` needs the fully separate layout (no shared Linear,
        phi or std). ``awr_advantage``: the AWR weight's advantage on a relabeled row, ``"leader"`` (1-step TD
        under the leader critic) or ``"follower"`` (the source follower's own GAE advantage, normalized per follower).
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
        assert shared_layers is None or block_layers is None, "set either shared_layers or block_layers, not both."
        assert div_minibatches >= 0, f"div_minibatches must be >= 0, got {div_minibatches}."
        assert off_policy_update in ("is", "awr"), f"off_policy_update must be 'is' / 'awr', got {off_policy_update!r}."
        assert awr_beta > 0.0 and awr_w_max > 0.0, f"awr_beta / awr_w_max must be > 0, got {awr_beta}, {awr_w_max}."
        assert off_policy_coef >= 0.0, f"off_policy_coef must be >= 0, got {off_policy_coef}."
        assert awr_advantage in ("leader", "follower"), f"awr_advantage: 'leader' / 'follower', got {awr_advantage!r}."
        assert awr_advantage == "leader" or off_policy_update == "awr", "awr_advantage='follower' needs AWR updates."
        if independent_blocks:
            assert shared_layers == 0 and block_layers is None and phi_dim == 0, (
                "independent_blocks needs shared_layers=0, phi_dim=0 (and no block_layers)."
            )
            assert critic_shared_layers in (None, 0) and critic_phi_dim in (None, 0), (
                "independent_blocks needs critic_shared_layers=0 and critic_phi_dim=0."
            )
            assert per_block_std and not freeze_shared_layers, "independent_blocks needs per_block_std (no shared std)."
            assert normalize_advantage_per_block, "independent_blocks needs normalize_advantage_per_block=True."
            assert target_kl is None, "independent_blocks: target_kl would stop every block at once; use desired_kl."
        # Conditioning parameters must exist before PPO builds the optimizer over the model parameters.
        indep = bool(independent_blocks)
        layout = {"shared_layers": shared_layers, "block_layers": block_layers, "independent": indep}
        attach_block_conditioning(actor, num_blocks, phi_dim, per_block_std=per_block_std, **layout)
        if critic_shared_layers is not None:
            assert block_layers is None, "critic_shared_layers needs the shared_layers layout, not block_layers."
            layout = {"shared_layers": critic_shared_layers, "block_layers": None, "independent": indep}
        c_phi_dim = phi_dim if critic_phi_dim is None else int(critic_phi_dim)
        attach_block_conditioning(critic, num_blocks, c_phi_dim, **layout)
        if freeze_shared_layers:  # shared actor Linears stay at their random init (per-block ones are BlockLinear)
            frozen = [m for m in actor.mlp if type(m) is nn.Linear]
            assert frozen, "freeze_shared_layers: the actor has no shared Linear."
            for m in frozen:
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)
            print(f"[SAPG] froze {len(frozen)} shared actor Linear(s)")
        if block_init_std is not None:
            assert per_block_std and len(block_init_std) == num_blocks, "block_init_std needs one std per block."
            dist = actor.distribution
            assert isinstance(dist, _BlockStdMixin), "block_init_std needs a per-block std distribution."
            with torch.no_grad():
                vals = torch.tensor([float(v) for v in block_init_std])
                if dist.sapg_std_name == "log_std_param":
                    vals = vals.log()
                if dist.sapg_std_split:
                    for param, value in zip(dist.sapg_std_params(), vals.tolist()):
                        param.fill_(value)
                else:
                    getattr(dist, dist.sapg_std_name).fill_(vals[0].item())
                    followers = dist.sapg_follower_std
                    shape = (-1, *([1] * (followers.dim() - 1)))
                    followers.copy_(vals[1:].to(followers.device).view(shape).expand_as(followers))
            print(f"[SAPG] per-block init std: {block_init_std}")
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
        self.critic_shared_layers = None if critic_shared_layers is None else int(critic_shared_layers)
        self.critic_phi_dim = None if critic_phi_dim is None else int(critic_phi_dim)
        num_linears = sum(isinstance(m, (nn.Linear, BlockLinear, SplitBlockLinear)) for m in actor.mlp)
        # resolved by attach_block_heads (None = no per-block Linears configured)
        self.block_layers: list[int] | None = getattr(actor, "sapg_block_layers", None)
        per_block = self.block_layers or []
        shared = [k for k in range(num_linears) if k not in per_block]
        print(f"[SAPG] per-block actor Linears: {per_block} (shared: {shared})")
        if self.critic_shared_layers is not None or self.critic_phi_dim is not None:
            c_block = getattr(critic, "sapg_block_layers", None) or []
            print(f"[SAPG] critic layout: per-block Linears {c_block}, phi_dim {c_phi_dim}")
        self.div_minibatches = int(div_minibatches)
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
        self.off_policy_update = off_policy_update
        self.awr_beta = float(awr_beta)
        self.awr_w_max = float(awr_w_max)
        self.awr_advantage = awr_advantage
        if off_policy_update == "awr":
            print(f"[SAPG] awr_advantage={awr_advantage}")
        self.off_policy_coef = float(off_policy_coef)
        self.independent_blocks = bool(independent_blocks)
        # relabeled rows form their own advantage-normalization group (else they join the leader / all rows)
        self._off_own_adv_group = self.independent_blocks or self.off_policy_update == "awr"
        self.block_params: list[list[nn.Parameter]] = []
        self.block_learning_rates = [float(self.learning_rate)] * self.num_blocks
        if self.independent_blocks:
            self._build_block_optimizer(str(kwargs.get("optimizer", "adam")))

        num_envs = storage.num_envs
        assert self.allow_uneven_blocks or num_envs % self.num_blocks == 0, (
            f"num_envs={num_envs} is not a multiple of num_blocks={self.num_blocks}."
        )
        # Contiguous equal blocks (the formula would also tolerate uneven ones).
        self.env_block_ids = torch.arange(num_envs, device=self.device) * self.num_blocks // num_envs
        self.leader_env_mask = self.env_block_ids == 0
        self.num_leader_envs = int(self.leader_env_mask.sum())
        # play2perfect: linspace(0.5, 0, M)[b] * scale with its leader last; ours is block 0, so follower j gets
        # 0.5 * j / (M - 1) * scale (the largest bonus on block M-1).
        if self.follower_entropy_coef is not None:
            self.block_entropy_coefs = [self.leader_entropy_coef] + [self.follower_entropy_coef] * (self.num_blocks - 1)
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

    def _build_block_optimizer(self, optimizer: str) -> None:
        """Group every trainable parameter by its block (asserting no sharing); one optimizer param group per block."""
        actor_groups = block_parameters(self.actor, self.num_blocks)
        critic_groups = block_parameters(self.critic, self.num_blocks)
        self.block_params = [a + c for a, c in zip(actor_groups, critic_groups)]
        owned = [id(p) for group in self.block_params for p in group]
        assert len(owned) == len(set(owned)), "independent_blocks: a parameter is owned by more than one block."
        trainable = [p for p in chain(self.actor.parameters(), self.critic.parameters()) if p.requires_grad]
        unowned = [tuple(p.shape) for p in trainable if id(p) not in set(owned)]
        assert not unowned, f"independent_blocks: parameters shared by every block: {unowned}."
        assert len(owned) == len(trainable), "independent_blocks: a block owns a frozen / foreign parameter."
        sizes = [sum(p.numel() for p in group) for group in self.block_params]
        assert len(set(sizes)) == 1, f"independent_blocks: blocks differ in size: {sizes}."
        self.optimizer = resolve_optimizer(optimizer)(
            [{"params": group} for group in self.block_params], lr=self.learning_rate
        )  # type: ignore
        print(f"[SAPG] independent_blocks: {self.num_blocks} blocks, {sizes[0]} params each, 0 shared")

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
        # per-block observation normalizers update from their own envs only
        with block_context(self.env_block_ids, self.actor, self.critic):
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
        # behavior block of every row (relabeled rows: their source follower)
        rows["src_block"] = self.env_block_ids.repeat(st.num_transitions_per_env)[rows["src"]]
        if self.awr_advantage == "follower":
            # the source follower's own raw GAE advantage (normalized per source follower below / per minibatch)
            rows["awr_adv"] = st.advantages.flatten(0, 1)[rows["src"]].clone()
        rows["entropy_coef"] = torch.cat([
            self._env_entropy_coef.repeat(st.num_transitions_per_env),
            torch.full((rows["src"].shape[0] - num_on,), self.block_entropy_coefs[0], device=self.device),
        ])
        # advantage-normalization group: block j's on-policy rows (per-block / independent) or 0, relabeled rows M
        per_block = self.normalize_advantage_per_block or self.independent_blocks
        on_group = rows["block"] if per_block else torch.zeros_like(rows["block"])
        rows["adv_group"] = torch.where(rows["off"] & self._off_own_adv_group, self.num_blocks, on_group)
        if not self.normalize_advantage_per_mini_batch:
            if self._off_own_adv_group:
                # the same group list on every rank (collectives inside _normalize)
                groups = list(range(self.num_blocks)) if per_block else [0]
                groups += [self.num_blocks] if off_blocks else []
                normalized = torch.empty_like(rows["advantages"])
                for g in groups:
                    mask = rows["adv_group"] == g
                    normalized[mask] = self._normalize(rows["advantages"][mask])
                rows["advantages"] = normalized
            else:
                rows["advantages"] = self._normalize(rows["advantages"])
            st.advantages = rows["advantages"][:num_on].view_as(st.advantages).clone()
            if self.awr_advantage == "follower":
                # one group per drawn follower: off_blocks is identical on every rank (collectives inside _normalize)
                awr_adv = rows["awr_adv"].clone()
                for j in off_blocks:
                    mask = rows["off"] & (rows["src_block"] == j)
                    awr_adv[mask] = self._normalize(rows["awr_adv"][mask])
                rows["awr_adv"] = awr_adv
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
                if self.normalize_advantage_per_block or self._off_own_adv_group:
                    groups = rows["adv_group"][idx]
                    normalized = torch.empty_like(advantages)
                    for j in range(self.num_blocks + 1):
                        mask = groups == j
                        if bool(mask.any()):
                            adv_j = advantages[mask]
                            std_j = adv_j.std() if adv_j.numel() > 1 else adv_j.new_zeros(())
                            normalized[mask] = (adv_j - adv_j.mean()) / (std_j + 1e-8)
                    advantages = normalized
                else:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        awr_adv = advantages
        if self.awr_advantage == "follower":
            src_block = rows["src_block"][idx]
            awr_adv = rows["awr_adv"][idx].view(-1)
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    normalized = awr_adv.clone()
                    for j in range(1, self.num_blocks):
                        mask = off & (src_block == j)
                        if bool(mask.any()):
                            adv_j = awr_adv[mask]
                            std_j = adv_j.std() if adv_j.numel() > 1 else adv_j.new_zeros(())
                            normalized[mask] = (adv_j - adv_j.mean()) / (std_j + 1e-8)
                    awr_adv = normalized
            awr_adv = torch.where(off, awr_adv, advantages)

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
        awr_w = awr_clamped = None
        actor_rows = surrogate
        if self.off_policy_update == "awr":
            with torch.no_grad():
                log_w = awr_adv / self.awr_beta
                awr_clamped = log_w > math.log(self.awr_w_max)
                awr_w = torch.exp(log_w.clamp(max=math.log(self.awr_w_max)))
            actor_rows = torch.where(off, -awr_w * log_prob, surrogate)
        if self.use_clipped_value_loss:
            value_clipped = old_values + (values - old_values).clamp(-self.clip_param, self.clip_param)
            value_rows = torch.max((values - returns).pow(2), (value_clipped - returns).pow(2)).view(-1)
        else:
            value_rows = (returns - values).pow(2).view(-1)
        entropy_rows = rows["entropy_coef"][idx] * entropy
        if self.independent_blocks:
            # each block: mean over its own on-policy rows; the relabeled rows: their own mean, x off_policy_coef
            per_row = actor_rows + self.value_loss_coef * value_rows - entropy_rows
            group = torch.where(off, self.num_blocks, block)
            sums = per_row.new_zeros(self.num_blocks + 1).index_add(0, group, per_row)
            counts = torch.bincount(group, minlength=self.num_blocks + 1).clamp(min=1).to(per_row.dtype)
            means = sums / counts
            loss = means[: self.num_blocks].sum() + self.off_policy_coef * means[self.num_blocks]
        elif self.off_policy_coef != 1.0 or self.off_policy_update != "is":
            row_w = torch.where(off, self.off_policy_coef, 1.0)
            loss = ((actor_rows + self.value_loss_coef * value_rows - entropy_rows) * row_w).mean()
        else:
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
            "actor_rows": actor_rows,
            "awr_w": awr_w,
            "awr_clamped": awr_clamped,
            "value_rows": value_rows,
            "entropy_rows": entropy_rows,
            "entropy_raw": entropy.detach(),
            "kl": kl,
            "inside_clip": inside,
            "ratio": ratio.detach(),
            "params": tuple(p.detach() for p in params),
            "block": block,
            "src_block": rows["src_block"][idx],
            "off": off,
        }

    def _grad_diagnostic(self, terms: dict) -> dict[str, torch.Tensor]:
        """play2perfect LOG_OFF_POLICY_GRADS: gradients of the relabeled vs on-policy parts of the loss."""
        per_row = terms["actor_rows"] + self.value_loss_coef * terms["value_rows"] - terms["entropy_rows"]
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

    def _div_blocks(self) -> list[int]:
        """Blocks whose Gaussian heads enter ``block_div/*`` (SAPGFlow: block 0's head is unused)."""
        return list(range(1 if hasattr(self, "flow_leader") else 0, self.num_blocks))

    @torch.no_grad()
    def _block_diversity(self, obs: TensorDict) -> dict[str, torch.Tensor] | None:
        """Pairwise distances between the blocks' deterministic action means on the same ``obs`` rows."""
        blocks = self._div_blocks()
        if len(blocks) < 2:
            return None
        n = obs.batch_size[0]
        mus, stds = [], []
        for j in blocks:
            with block_context(torch.full((n,), j, dtype=torch.long, device=self.device), self.actor):
                mu = self.actor(obs)
                # Gaussian / gSDE: the mean is the raw MLP output; update() applies block j's std
                self.actor.distribution.update(mu)  # type: ignore[union-attr]
                std = self.actor.output_std.expand_as(mu)
            mus.append(mu)
            stds.append(std)
        dist_std, kl, dist_raw, agree = [], [], [], []
        for a in range(len(blocks)):
            for b in range(a + 1, len(blocks)):
                diff = mus[a] - mus[b]
                var_a, var_b = stds[a].pow(2), stds[b].pow(2)
                dist_std.append((diff / (0.5 * (stds[a] + stds[b]))).pow(2).mean().sqrt())
                # symmetric KL of diag Gaussians, summed over action dims
                kl_rows = 0.5 * ((var_a + diff.pow(2)) / var_b + (var_b + diff.pow(2)) / var_a - 2.0).sum(dim=-1)
                kl.append(kl_rows.mean())
                dist_raw.append(diff.pow(2).mean().sqrt())
                agree.append((torch.sign(mus[a]) == torch.sign(mus[b])).all(dim=-1).float().mean())
        dist_std_t = torch.stack(dist_std)
        return {
            "block_div/pair_dist_std_mean": dist_std_t.mean(),
            "block_div/pair_dist_std_min": dist_std_t.min(),
            "block_div/pair_kl_mean": torch.stack(kl).mean(),
            "block_div/pair_dist_raw_mean": torch.stack(dist_raw).mean(),
            "block_div/argmax_agree": torch.stack(agree).mean(),
        }

    def _accumulate_block_diversity(self, obs: TensorDict, sums: dict[str, torch.Tensor]) -> None:
        """Add one minibatch's ``block_div/*`` values to ``sums`` (plus the ``_div_count`` counter)."""
        metrics = self._block_diversity(obs)
        if metrics is None:
            return
        for key, value in metrics.items():
            sums[key] = sums[key] + value if key in sums else value
        sums["_div_count"] = sums.get("_div_count", torch.zeros((), device=self.device)) + 1.0

    @staticmethod
    def _block_diversity_logs(out: dict[str, float]) -> dict[str, float]:
        count = out.get("_div_count", 0.0)
        if count <= 0:
            return {}
        return {k: v / count for k, v in out.items() if k.startswith("block_div/")}

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
        epoch_block_kls: list[torch.Tensor] = []
        # per source follower [sum w, sum w^2, count] of the AWR weights on relabeled rows
        awr_f = torch.zeros(3, self.num_blocks, device=self.device)
        self.last_minibatch_debug = None
        num_steps_run, stop, approx_kl_first, approx_kl_last = 0, False, 0.0, 0.0

        for epoch in range(self.num_learning_epochs):
            if stop:
                break
            num_epoch_steps = 0
            # [sum over minibatch means (all rows), leader-on KL sum, leader-on count, relabeled KL sum, count]
            kl_acc = torch.zeros(5, device=self.device)
            # per-block [KL sum, row count] over each block's own on-policy rows
            block_kl_acc = torch.zeros(2, self.num_blocks, device=self.device)
            for i in range(self.num_mini_batches):
                idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                if epoch == 0 and i < self.div_minibatches:
                    self._accumulate_block_diversity(flat["observations"][rows["src"][idx]], sums)
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
                on_block = terms["block"][~off]
                block_kl_acc[0].index_add_(0, on_block, kl[~off].float())
                block_kl_acc[1].index_add_(0, on_block, torch.ones_like(kl[~off], dtype=torch.float32))
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
                if self.independent_blocks:
                    for params in self.block_params:
                        nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                else:
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
                    acc("_ratio_sq_off", terms["ratio"][off].pow(2).sum())
                    acc("_actor_off", terms["actor_rows"][off].detach().sum())
                    acc("_count_off", off.sum())
                    if terms["awr_w"] is not None:
                        acc("_awr_w", terms["awr_w"][off].sum())
                        acc("_awr_w_sq", terms["awr_w"][off].pow(2).sum())
                        acc("_awr_clamped", terms["awr_clamped"][off].sum())
                        w_off, b_off = terms["awr_w"][off].float(), terms["src_block"][off]
                        awr_f[0].index_add_(0, b_off, w_off)
                        awr_f[1].index_add_(0, b_off, w_off.pow(2))
                        awr_f[2].index_add_(0, b_off, torch.ones_like(w_off))
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
            if self.independent_blocks:
                if self.is_multi_gpu:
                    torch.distributed.all_reduce(block_kl_acc, op=torch.distributed.ReduceOp.SUM)
                block_kl = block_kl_acc[0] / block_kl_acc[1].clamp(min=1.0)
                epoch_block_kls.append(block_kl)
                self._adapt_block_learning_rates(block_kl)
            else:
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
            n_off = out["_count_off"]
            loss_dict["SAPG/kl_relabeled"] = kls[2]
            loss_dict["SAPG/clip_inside_off"] = out["_inside_off"] / n_off
            loss_dict["SAPG/clip_frac_off"] = 1.0 - out["_inside_off"] / n_off
            loss_dict["SAPG/ratio_off_mean"] = out["_ratio_off"] / n_off
            # ESS fraction (sum w)^2 / (n sum w^2), pooled over every minibatch of the update
            loss_dict["SAPG/ratio_off_ess"] = out["_ratio_off"] ** 2 / max(n_off * out["_ratio_sq_off"], 1e-12)
            loss_dict["SAPG/off_actor_loss"] = out["_actor_off"] / n_off
            if "_awr_w" in out:
                loss_dict["SAPG/awr_w_mean"] = out["_awr_w"] / n_off
                loss_dict["SAPG/awr_w_max_frac"] = out["_awr_clamped"] / n_off
                loss_dict["SAPG/awr_w_ess"] = out["_awr_w"] ** 2 / max(n_off * out["_awr_w_sq"], 1e-12)
                w_sum, w_sq, w_n = awr_f.tolist()
                for j in range(1, self.num_blocks):
                    if w_n[j] > 0:
                        loss_dict[f"SAPG/awr_w_mean_f{j}"] = w_sum[j] / w_n[j]
                        loss_dict[f"SAPG/awr_w_ess_f{j}"] = w_sum[j] ** 2 / max(w_n[j] * w_sq[j], 1e-12)
        if epoch_block_kls:
            block_kls = torch.stack(epoch_block_kls).mean(dim=0).tolist()
            for j in range(self.num_blocks):
                loss_dict[f"SAPG/block{j}_kl"] = block_kls[j]
                loss_dict[f"SAPG/block{j}_lr"] = self.block_learning_rates[j]
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
        loss_dict.update(self._block_diversity_logs(out))
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

    def _adapt_block_learning_rates(self, kl: torch.Tensor) -> None:
        """The adaptive rule per block on its own KL; rank 0 decides, every rank gets its LRs."""
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        lrs = list(self.block_learning_rates)
        if self.gpu_global_rank == 0:
            for j, kl_value in enumerate(kl.tolist()):
                if kl_value > self.desired_kl * 2.0:
                    lrs[j] = max(1e-5, lrs[j] / 1.5)
                elif kl_value < self.desired_kl / 2.0 and kl_value > 0.0:
                    lrs[j] = min(1e-2, lrs[j] * 1.5)
        if self.is_multi_gpu:
            lr_tensor = torch.tensor(lrs, device=self.device, dtype=torch.float64)
            torch.distributed.broadcast(lr_tensor, src=0)
            lrs = lr_tensor.tolist()
        self._set_block_learning_rates(lrs)

    def _set_block_learning_rates(self, lrs: list[float]) -> None:
        assert len(self.optimizer.param_groups) == len(lrs) == self.num_blocks, "one optimizer group per block."
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr
        self.block_learning_rates = list(lrs)
        self.learning_rate = lrs[0]  # the runner logs the leader's

    def save(self) -> dict:
        """PPO checkpoint (phi / per-block std live in the actor/critic state dicts) plus the block layout."""
        saved = super().save()
        saved["sapg"] = {
            "num_blocks": self.num_blocks,
            "phi_dim": self.phi_dim,
            "per_block_std": self.per_block_std,
            "shared_layers": self.shared_layers,
            "block_layers": self.block_layers,
            "critic_shared_layers": self.critic_shared_layers,
            "critic_phi_dim": self.critic_phi_dim,
            "independent_blocks": self.independent_blocks,
            "off_policy_update": self.off_policy_update,
            "awr_beta": self.awr_beta,
            "awr_w_max": self.awr_w_max,
            "awr_advantage": self.awr_advantage,
            "off_policy_coef": self.off_policy_coef,
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
        for k in ("critic_shared_layers", "critic_phi_dim"):
            assert saved.get(k) == getattr(self, k), f"SAPG checkpoint has {k}={saved.get(k)}; this run has {getattr(self, k)}."
        if "block_layers" in saved:
            assert saved["block_layers"] == self.block_layers, (
                f"SAPG checkpoint has block_layers={saved['block_layers']}; this run has {self.block_layers}."
            )
        assert bool(saved.get("independent_blocks", False)) == self.independent_blocks, (
            f"SAPG checkpoint has independent_blocks={saved.get('independent_blocks', False)};"
            f" this run has {self.independent_blocks}."
        )
        if self.independent_blocks:  # per-block normalizers / heads / stds must all be in the checkpoint
            for key, model in (("actor_state_dict", self._raw_actor), ("critic_state_dict", self._raw_critic)):
                missing = set(model.state_dict()) ^ set(loaded_dict[key])
                assert not missing, f"SAPG independent_blocks: {key} key mismatch {sorted(missing)[:8]}."
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("optimizer"):
            lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
            if self.independent_blocks:
                self._set_block_learning_rates(lrs)
            else:
                self.learning_rate = lrs[0]
            print(f"[SAPG] restored learning rate(s) from the optimizer state: {lrs}")
        return load_iteration
