# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SAPG ``independent_blocks`` (zero coupling between blocks) and the ``off_policy_update`` IS / AWR switch; CPU.

source /home/yandabao/miniforge3/etc/profile.d/conda.sh && conda activate patlab3
PYTHONPATH=. python -m pytest tests/test_sapg_independent.py -q
"""

from __future__ import annotations

import copy
import math
import pytest
import torch

from rsl_rl.algorithms import SAPG
from rsl_rl.models import BlockNormalization, SplitBlockLinear, block_context
from rsl_rl.runners import OnPolicyRunner
from tests.runners.test_on_policy_runner import DummyEnv
from tests.test_sapg import GAUSSIAN, GSDE, NUM_ENVS, _alg_kwargs, _collect, _models, _obs, _storage

NUM_BLOCKS = 3


def _indep(update: str = "is", dist: dict = GSDE, seed: int = 0, **overrides) -> SAPG:
    actor, critic = _models(dist=dist, seed=seed)
    kw = dict(
        num_blocks=NUM_BLOCKS,
        phi_dim=0,
        shared_layers=0,
        per_block_std=True,
        independent_blocks=True,
        normalize_advantage_per_block=True,
        off_policy_ratio=1.0,
        off_policy_sampling="uniform",
        off_policy_update=update,
        sapg_entropy_scale=0.0,
        leader_entropy_coef=0.006,
        follower_entropy_coef=0.006,
    )
    for key in list(overrides):
        if key in kw or key in ("awr_beta", "awr_w_max", "off_policy_coef", "kl_reference", "critic_shared_layers"):
            kw[key] = overrides.pop(key)
    return SAPG(actor, critic, _storage(), **kw, **_alg_kwargs(normalize_advantage_per_mini_batch=True, **overrides))


def _grads(alg: SAPG) -> list[float]:
    """Total |grad| per block."""
    return [sum(float(p.grad.abs().sum()) for p in group if p.grad is not None) for group in alg.block_params]


def test_layout_every_parameter_has_one_owner(capsys) -> None:
    alg = _indep()
    assert "[SAPG] independent_blocks: 3 blocks" in capsys.readouterr().out
    assert all(isinstance(m, SplitBlockLinear) for m in alg.actor.mlp if not isinstance(m, torch.nn.ELU))
    assert isinstance(alg.actor.obs_normalizer, BlockNormalization)
    assert isinstance(alg.critic.obs_normalizer, BlockNormalization)
    assert len(alg.optimizer.param_groups) == NUM_BLOCKS
    owned = {id(p) for g in alg.block_params for p in g}
    assert owned == {id(p) for p in list(alg.actor.parameters()) + list(alg.critic.parameters())}
    with pytest.raises(AssertionError):  # shared first Linear
        _indep(shared_layers=1)
    with pytest.raises(AssertionError):  # shared critic Linear
        _indep(critic_shared_layers=1)


@pytest.mark.parametrize("update", ["is", "awr"])
def test_block_rows_only_touch_their_block(update: str) -> None:
    alg = _indep(update)
    _collect(alg)
    rows = alg._rows
    flat = {
        "observations": alg.storage.observations.flatten(0, 1),
        "actions": alg.storage.actions.flatten(0, 1),
        "actions_log_prob": alg.storage.actions_log_prob.flatten(0, 1),
    }
    old = tuple(p.flatten(0, 1)[rows["src"]] for p in alg.storage.distribution_params)
    assert bool(rows["off"].any())
    for k in range(NUM_BLOCKS):
        idx = torch.nonzero((rows["block"] == k) & ~rows["off"]).view(-1)
        alg.optimizer.zero_grad()
        alg.minibatch_losses(idx, flat, tuple(p[idx] for p in old))["loss"].backward()
        grads = _grads(alg)
        assert grads[k] > 0 and all(g == 0 for j, g in enumerate(grads) if j != k), (k, grads)
    idx = torch.nonzero(rows["off"]).view(-1)
    alg.optimizer.zero_grad()
    alg.minibatch_losses(idx, flat, tuple(p[idx] for p in old))["loss"].backward()
    grads = _grads(alg)
    assert grads[0] > 0 and grads[1:] == [0.0] * (NUM_BLOCKS - 1), grads


def test_loss_is_sum_of_per_block_means() -> None:
    alg = _indep("awr", off_policy_coef=0.5)
    _collect(alg)
    rows = alg._rows
    flat = {
        "observations": alg.storage.observations.flatten(0, 1),
        "actions": alg.storage.actions.flatten(0, 1),
        "actions_log_prob": alg.storage.actions_log_prob.flatten(0, 1),
    }
    old = tuple(p.flatten(0, 1)[rows["src"]] for p in alg.storage.distribution_params)
    idx = torch.randperm(rows["src"].shape[0])
    terms = alg.minibatch_losses(idx, flat, tuple(p[idx] for p in old))
    per_row = terms["actor_rows"] + alg.value_loss_coef * terms["value_rows"] - terms["entropy_rows"]
    off, block = terms["off"], terms["block"]
    expected = sum(per_row[~off & (block == j)].mean() for j in range(NUM_BLOCKS)) + 0.5 * per_row[off].mean()
    assert torch.allclose(terms["loss"], expected, atol=1e-5)
    # AWR rows: -clamp(exp(A / beta), w_max) * log pi_L, weights never above w_max
    assert float(terms["awr_w"][off].max()) <= alg.awr_w_max + 1e-6


def test_per_block_normalizers_see_only_their_envs() -> None:
    alg = _indep()
    shift = torch.repeat_interleave(torch.arange(NUM_BLOCKS, dtype=torch.float32) * 10.0, NUM_ENVS // NUM_BLOCKS)
    obs = _obs()
    obs["policy"] = obs["policy"] + shift.unsqueeze(1)
    with torch.no_grad():
        for _ in range(3):
            alg.act(obs)
            alg.process_env_step(obs, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), {})
    for model in (alg.actor, alg.critic):
        for j, norm in enumerate(model.obs_normalizer.norms):
            assert norm.count == 3 * NUM_ENVS // NUM_BLOCKS
            assert abs(float(norm.mean[0]) - 10.0 * j) < 3.0
    # A bare forward (export / relabeling) uses the leader's statistics and does not update them.
    count = alg.actor.obs_normalizer.norms[0].count.clone()
    x = torch.randn(4, alg.actor.obs_dim)
    assert torch.equal(alg.actor.obs_normalizer(x), alg.actor.obs_normalizer.norms[0](x))
    assert torch.equal(alg.actor.obs_normalizer.norms[0].count, count)


@pytest.mark.parametrize("update", ["is", "awr"])
def test_update_logs_and_per_block_lr(update: str) -> None:
    alg = _indep(update)
    _collect(alg)
    losses = alg.update()
    keys = [f"SAPG/block{j}_{k}" for j in range(NUM_BLOCKS) for k in ("lr", "kl")]
    keys += ["SAPG/off_actor_loss", "SAPG/ratio_off_mean", "SAPG/ratio_off_ess", "SAPG/clip_frac_off"]
    if update == "awr":
        keys += ["SAPG/awr_w_mean", "SAPG/awr_w_max_frac", "SAPG/awr_w_ess"]
    for key in keys:
        assert key in losses and math.isfinite(losses[key]), key
    assert [g["lr"] for g in alg.optimizer.param_groups] == alg.block_learning_rates
    assert alg.learning_rate == alg.block_learning_rates[0]


def test_per_block_lr_follows_its_own_kl() -> None:
    alg = _indep()
    alg._adapt_block_learning_rates(torch.tensor([1.0, 0.0001, 0.01]))
    lr = 1e-3
    assert alg.block_learning_rates == pytest.approx([lr / 1.5, lr * 1.5, lr])


def test_per_block_grad_clip(monkeypatch) -> None:
    alg = _indep(max_grad_norm=1e-3)
    _collect(alg)
    calls: list[set[int]] = []
    original = torch.nn.utils.clip_grad_norm_

    def record(parameters, max_norm, *args, **kwargs):
        params = list(parameters)
        calls.append({id(p) for p in params})
        total = original(params, max_norm, *args, **kwargs)
        norm = torch.sqrt(sum(p.grad.pow(2).sum() for p in params))
        assert float(norm) <= max_norm * (1 + 1e-4)
        return total

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record)
    alg.update()
    groups = [{id(p) for p in g} for g in alg.block_params]
    steps = alg.num_learning_epochs * alg.num_mini_batches
    assert calls == groups * steps


def test_checkpoint_restores_blocks_normalizers_and_lrs(tmp_path) -> None:
    alg = _indep("awr")
    for _ in range(2):
        _collect(alg)
        alg.update()
    alg._set_block_learning_rates([2e-4, 3e-4, 4e-4])
    torch.save(alg.save(), tmp_path / "m.pt")
    other = _indep("awr", seed=7)
    other.load(torch.load(tmp_path / "m.pt", weights_only=False), None, True)
    assert other.block_learning_rates == pytest.approx([2e-4, 3e-4, 4e-4])
    assert [g["lr"] for g in other.optimizer.param_groups] == pytest.approx([2e-4, 3e-4, 4e-4])
    for a, b in ((alg.actor, other.actor), (alg.critic, other.critic)):
        for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
            assert ka == kb and torch.equal(va, vb), ka
    assert any("norms.2._mean" in k for k in other.actor.state_dict())
    # a non-independent checkpoint does not load into an independent run (and vice versa)
    with pytest.raises(AssertionError):
        actor, critic = _models(dist=GSDE)
        SAPG(actor, critic, _storage(), num_blocks=NUM_BLOCKS, phi_dim=0, shared_layers=0, **_alg_kwargs()).load(
            torch.load(tmp_path / "m.pt", weights_only=False), None, True
        )


def test_leader_export_and_block_forward() -> None:
    alg = _indep(dist=GAUSSIAN)
    _collect(alg)
    alg.update()
    policy = alg.get_policy()
    alg.eval_mode()
    obs = _obs(generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        leader = policy(obs)
        with block_context(torch.zeros(NUM_ENVS, dtype=torch.long), policy):
            assert torch.equal(policy(obs), leader)
        with block_context(torch.ones(NUM_ENVS, dtype=torch.long), policy):
            assert not torch.allclose(policy(obs), leader)
        jit = torch.jit.script(policy.as_jit())
        assert torch.allclose(jit(torch.cat([obs[g] for g in policy.obs_groups], dim=-1)), leader, atol=1e-6)


def test_default_is_mode_matches_previous_loss() -> None:
    """off_policy_update='is', off_policy_coef=1, not independent: the loss is the old all-row mean."""
    actor, critic = _models(dist=GAUSSIAN)
    alg = SAPG(actor, critic, _storage(), num_blocks=NUM_BLOCKS, phi_dim=4, **_alg_kwargs())
    _collect(alg)
    rows = alg._rows
    flat = {
        "observations": alg.storage.observations.flatten(0, 1),
        "actions": alg.storage.actions.flatten(0, 1),
        "actions_log_prob": alg.storage.actions_log_prob.flatten(0, 1),
    }
    old = tuple(p.flatten(0, 1)[rows["src"]] for p in alg.storage.distribution_params)
    idx = torch.arange(rows["src"].shape[0])
    terms = alg.minibatch_losses(idx, flat, tuple(p[idx] for p in old))
    expected = terms["surrogate_rows"].mean() + terms["value_rows"].mean() - terms["entropy_rows"].mean()
    assert torch.equal(terms["loss"], expected)


@pytest.mark.parametrize("update", ["is", "awr"])
def test_runner_learns_saves_and_resumes(tmp_path, update: str) -> None:
    cfg = {
        "num_steps_per_env": 8,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "SAPG",
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "num_blocks": 2,
            "phi_dim": 0,
            "shared_layers": 0,
            "independent_blocks": True,
            "normalize_advantage_per_block": True,
            "normalize_advantage_per_mini_batch": True,
            "off_policy_sampling": "uniform",
            "off_policy_update": update,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [16, 16],
            "obs_normalization": True,
            "distribution_cfg": {"class_name": "GsdeDistribution"},
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [16], "obs_normalization": True},
    }
    runner = OnPolicyRunner(DummyEnv(), copy.deepcopy(cfg), log_dir=None, device="cpu")
    runner.learn(num_learning_iterations=2)
    path = str(tmp_path / "model.pt")
    runner.save(path)
    other = OnPolicyRunner(DummyEnv(), copy.deepcopy(cfg), log_dir=None, device="cpu")
    other.load(path)
    assert other.current_learning_iteration == runner.current_learning_iteration
    assert other.alg.block_learning_rates == runner.alg.block_learning_rates
    other.learn(num_learning_iterations=1)
    runner.export_policy_to_jit(str(tmp_path), "policy.pt")
