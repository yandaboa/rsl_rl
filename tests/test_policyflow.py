# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PolicyFlow (``rsl_rl.algorithms.PolicyFlow`` + ``rsl_rl.models.PolicyFlowModel``): CPU, tiny dims, no Isaac.

    conda activate patlab3 && PYTHONPATH=. python -m pytest tests/test_policyflow.py -q
"""

from __future__ import annotations

import math
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PolicyFlow
from rsl_rl.models import MLPModel, PolicyFlowModel
from rsl_rl.storage import RolloutStorage

OBS_DIM = 5
ACT_DIM = 3
NUM_ENVS = 8
STEPS = 4
OBS_GROUPS = {"actor": ["policy"], "critic": ["policy"]}


def _obs(num_envs: int = NUM_ENVS, obs_dim: int = OBS_DIM, scale: float = 1.0) -> TensorDict:
    return TensorDict({"policy": scale * torch.randn(num_envs, obs_dim)}, batch_size=[num_envs])


def _actor(obs: TensorDict, act_dim: int = ACT_DIM, **kwargs) -> PolicyFlowModel:
    kw = dict(hidden_dims=[16, 16], obs_normalization=True, embed_dim=16, sampling_steps=4)
    kw.update(kwargs)
    return PolicyFlowModel(obs, OBS_GROUPS, "actor", act_dim, **kw)


def _alg(num_envs: int = NUM_ENVS, steps: int = STEPS, act_dim: int = ACT_DIM, actor_kw=None, **kwargs) -> PolicyFlow:
    obs = _obs(num_envs)
    actor = _actor(obs, act_dim, **(actor_kw or {}))
    critic = MLPModel(obs, OBS_GROUPS, "critic", 1, hidden_dims=[16, 16], obs_normalization=True)
    storage = RolloutStorage("rl", num_envs, steps, obs, [act_dim], "cpu")
    kw = dict(num_learning_epochs=2, num_mini_batches=2, learning_rate=1e-3, entropy_coef=0.0)
    kw.update(kwargs)
    return PolicyFlow(actor, critic, storage, **kw)


def _rollout(alg: PolicyFlow, steps: int = STEPS, obs_scale: float = 1.0) -> None:
    n = alg.storage.num_envs
    obs = _obs(n, alg.actor.obs_dim, obs_scale)
    for _ in range(steps):
        alg.act(obs)
        obs = _obs(n, alg.actor.obs_dim, obs_scale)
        alg.process_env_step(obs, torch.randn(n), torch.zeros(n), {})
    alg.compute_returns(obs)


def test_time_grid_and_init():
    obs = _obs()
    actor = _actor(obs, sampling_steps=4)
    torch.testing.assert_close(
        actor.train_time_grid, torch.tensor([0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0])
    )
    torch.testing.assert_close(actor.std, torch.ones(ACT_DIM))
    assert "time_embed.freqs" in actor.state_dict()
    assert not any(p is actor.time_embed.freqs for p in actor.parameters())


def test_ratio_is_one_before_any_step():
    # lr=0: every minibatch sees the rollout policy; dv must be exactly 0 and the ratio exactly 1.
    torch.manual_seed(1)
    alg = _alg(learning_rate=0.0)
    _rollout(alg, obs_scale=5.0)
    out = alg.update()
    assert out["policyflow/dv_abs_max"] == 0.0
    assert out["policyflow/ratio_mean"] == 1.0
    assert out["policyflow/clip_frac"] == 0.0
    assert out["policyflow/approx_kl_mean"] == 0.0
    assert math.isfinite(out["policyflow/brownian"])


def test_first_minibatch_ratio_exactly_one():
    torch.manual_seed(2)
    alg = _alg(learning_rate=1e-2, num_learning_epochs=1, num_mini_batches=1)
    _rollout(alg)
    out = alg.update()
    assert out["policyflow/ratio_mean"] == 1.0 and out["policyflow/dv_abs_max"] == 0.0
    # next update starts from the re-synced actor_old: ratio 1 again on its first (only) minibatch
    _rollout(alg)
    out = alg.update()
    assert out["policyflow/ratio_mean"] == 1.0 and out["policyflow/dv_abs_max"] == 0.0


def test_stored_noise_is_exact():
    torch.manual_seed(3)
    alg = _alg()
    _rollout(alg)
    torch.testing.assert_close(alg.pf_phi + alg.pf_noise, alg.storage.actions, rtol=0, atol=1e-6)
    expected = torch.distributions.Normal(0.0, alg.pf_old_std).log_prob(alg.pf_noise).sum(-1)
    torch.testing.assert_close(alg.pf_old_logp, expected)


def test_jit_export_matches_zero_noise_forward():
    torch.manual_seed(4)
    obs = _obs()
    actor = _actor(obs)
    actor.update_normalization(_obs(64, scale=3.0))
    with torch.no_grad():
        for p in actor.mlp.parameters():
            p.add_(0.1 * torch.randn_like(p))  # make the field non-trivial
    actor.eval()
    scripted = torch.jit.script(actor.as_jit())
    torch.testing.assert_close(scripted(obs["policy"]), actor(obs))
    # stochastic forward also yields the same zero-noise rows as output_mean
    actor(obs, stochastic_output=True)
    torch.testing.assert_close(actor.output_mean, actor(obs))


def test_update_is_finite_and_moves_params():
    torch.manual_seed(5)
    alg = _alg(brownian_reg_coef=0.01, entropy_coef=0.01)
    before = [p.detach().clone() for p in alg.actor.parameters()]
    _rollout(alg)
    out = alg.update()
    assert all(math.isfinite(v) for v in out.values())
    for k in ("value", "surrogate", "entropy", "policyflow/brownian", "policyflow/kl", "policyflow/lr"):
        assert k in out
    assert out["policyflow/brownian"] > 0.0
    assert any(not torch.equal(b, p) for b, p in zip(before, alg.actor.parameters()))
    assert alg.storage.step == 0


def test_target_kl_early_stop():
    for target, expect_all in ((None, True), (1e6, True), (1e-12, False)):
        torch.manual_seed(6)
        alg = _alg(target_kl=target, learning_rate=1e-2, num_learning_epochs=3, num_mini_batches=2)
        _rollout(alg)
        out = alg.update()
        planned = alg.num_learning_epochs * alg.num_mini_batches
        assert math.isfinite(out["policyflow/approx_kl_mean"]) and out["policyflow/approx_kl_mean"] >= 0.0
        if expect_all:
            assert out["kl_stop/mini_batch_updates"] == planned and out["kl_stop/stopped"] == 0.0
        else:
            # first minibatch has approx KL 0 (ratio 1), the second sees the step and stops
            assert out["kl_stop/mini_batch_updates"] == 2 and out["kl_stop/stopped"] == 1.0


def test_adaptive_schedule_changes_lr():
    torch.manual_seed(7)
    alg = _alg(schedule="adaptive", desired_kl=1e-9, learning_rate=1e-3)
    _rollout(alg)
    out = alg.update()
    assert out["policyflow/lr"] < 1e-3


def test_eval_pool_executes_zero_noise_action():
    torch.manual_seed(8)
    obs = _obs()
    actor = _actor(obs)
    critic = MLPModel(obs, OBS_GROUPS, "critic", 1, hidden_dims=[16])
    storage = RolloutStorage("rl", NUM_ENVS, STEPS, obs, [ACT_DIM], "cpu", num_eval_envs=2)
    alg = PolicyFlow(actor, critic, storage, eval_env_fraction=0.25, learning_rate=1e-3)
    alg.train_mode()
    actions = alg.act(obs)
    torch.testing.assert_close(actions[-2:], actor.output_mean[-2:])
    assert not torch.allclose(actions[:-2], actor.output_mean[:-2])


def test_construct_algorithm():
    class _Env:
        num_envs = NUM_ENVS
        num_actions = ACT_DIM

    obs = _obs()
    cfg = {
        "algorithm": {"class_name": "rsl_rl.algorithms:PolicyFlow", "learning_rate": 1e-4, "target_kl": 0.1},
        "actor": {"class_name": "rsl_rl.models:PolicyFlowModel", "hidden_dims": [16], "embed_dim": 16},
        "critic": {"class_name": "rsl_rl.models:MLPModel", "hidden_dims": [16]},
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": STEPS,
        "multi_gpu": None,
    }
    alg = PolicyFlow.construct_algorithm(obs, _Env(), cfg, "cpu")
    assert isinstance(alg, PolicyFlow) and isinstance(alg.optimizer, torch.optim.AdamW)
    assert alg.pf_noise.shape == (STEPS, NUM_ENVS, ACT_DIM)
    saved = alg.save()
    alg.load(saved, None, strict=True)


def test_bandit_moves_zero_noise_action_to_target():
    # One-step bandit, reward -(a - 0.5)^2 on a 1-D action (converges in ~25 updates at this lr).
    torch.manual_seed(9)
    n, steps, target = 256, 4, 0.5
    alg = _alg(
        num_envs=n,
        steps=steps,
        act_dim=1,
        actor_kw=dict(obs_normalization=False, hidden_dims=[32, 32], init_std=0.5),
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=1e-3,
        normalize_advantage_per_mini_batch=True,
    )
    obs = TensorDict({"policy": torch.ones(n, OBS_DIM)}, batch_size=[n])
    alg.train_mode()

    def zero_noise() -> float:
        with torch.no_grad():
            return alg.actor(obs[:1]).item()

    start = zero_noise()
    for _ in range(30):
        with torch.inference_mode():
            for _ in range(steps):
                a = alg.act(obs)
                alg.process_env_step(obs, -((a[:, 0] - target) ** 2), torch.ones(n), {})
            alg.compute_returns(obs)
        alg.update()
    end = zero_noise()
    assert abs(end - target) < 0.1, (start, end)
    assert abs(end - target) < abs(start - target)
