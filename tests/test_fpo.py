# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FPO++ (``rsl_rl.algorithms.FPO`` + ``rsl_rl.models.FlowMLPModel``): CPU, tiny dims, no Isaac.

    conda activate patlab3 && PYTHONPATH=. python -m pytest tests/test_fpo.py -q
"""

from __future__ import annotations

import math
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import FPO
from rsl_rl.algorithms.fpo import clamp_ste
from rsl_rl.models import FlowMLPModel, MLPModel
from rsl_rl.models.flow_model import embed_timestep
from rsl_rl.storage import RolloutStorage

OBS_DIM = 5
ACT_DIM = 3
NUM_ENVS = 8
STEPS = 4
OBS_GROUPS = {"actor": ["policy"], "critic": ["policy"]}


def _obs(num_envs: int = NUM_ENVS, obs_dim: int = OBS_DIM, scale: float = 1.0) -> TensorDict:
    return TensorDict({"policy": scale * torch.randn(num_envs, obs_dim)}, batch_size=[num_envs])


def _actor(obs: TensorDict, act_dim: int = ACT_DIM, **kwargs) -> FlowMLPModel:
    kw = dict(hidden_dims=[16, 16], obs_normalization=True, sampling_steps=8)
    kw.update(kwargs)
    return FlowMLPModel(obs, OBS_GROUPS, "actor", act_dim, **kw)


def _fpo(num_envs: int = NUM_ENVS, steps: int = STEPS, act_dim: int = ACT_DIM, actor_kw=None, **kwargs) -> FPO:
    obs = _obs(num_envs)
    actor = _actor(obs, act_dim, **(actor_kw or {}))
    critic = MLPModel(obs, OBS_GROUPS, "critic", 1, hidden_dims=[16, 16], obs_normalization=True)
    storage = RolloutStorage("rl", num_envs, steps, obs, [act_dim], "cpu")
    kw = dict(
        n_samples_per_action=4,
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1e-3,
        clip_param=0.05,
        ema_decay=0.9,
        ema_warmup_steps=1,
        entropy_coef=0.0,
    )
    kw.update(kwargs)
    return FPO(actor, critic, storage, **kw)


def _rollout(alg: FPO, steps: int = STEPS, obs_scale: float = 1.0) -> None:
    n = alg.storage.num_envs
    obs = _obs(n, alg.actor.obs_dim, obs_scale)
    for _ in range(steps):
        alg.act(obs)
        obs = _obs(n, alg.actor.obs_dim, obs_scale)
        alg.process_env_step(obs, torch.randn(n), torch.zeros(n), {})
    alg.compute_returns(obs)


def test_cfm_loss_matches_hand_computation():
    torch.manual_seed(0)
    obs = _obs(2)
    actor = _actor(obs, actor_scale=2.0)
    actions = torch.randn(2, ACT_DIM)
    eps = torch.randn(2, 3, ACT_DIM)
    t = torch.rand(2, 3, 1)
    loss, x1, x0 = actor.get_cfm_loss(obs, actions, eps, t)

    latent = actor.obs_normalizer(obs["policy"])
    a = actions / 2.0
    for b in range(2):
        for s in range(3):
            tt = t[b, s]
            x_t = tt * eps[b, s] + (1 - tt) * a[b]
            freqs = torch.tensor([1.0, 2.0, 4.0, 8.0])
            emb = torch.cat([torch.cos(tt * freqs), torch.sin(tt * freqs)])
            v = actor.mlp(torch.cat([latent[b], emb, x_t]))
            expected = ((v - (eps[b, s] - a[b])) ** 2).sum() / math.sqrt(ACT_DIM)
            torch.testing.assert_close(loss[b, s], expected)
            torch.testing.assert_close(x0[b, s], x_t - tt * v)
            torch.testing.assert_close(x1[b, s], x_t - tt * v + v)
    assert embed_timestep(torch.zeros(1, 1), 8).tolist() == [[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]


def test_cfm_t_sampling_range():
    actor = _actor(_obs(), cfm_loss_t_inverse_cdf_beta=3.0)
    t = actor.sample_cfm_t((1000, 4, 1), "cpu")
    assert t.min() >= 0.005 and t.max() <= 0.995
    assert t.mean() < 0.5  # beta > 1 favours t near 0 (the action end)


def test_ratio_is_one_before_any_step():
    # lr=0: every minibatch of every epoch sees the rollout policy. Obs normalization is on and the stats move
    # during the rollout, so this also checks that the update reuses the act-time latent.
    torch.manual_seed(1)
    alg = _fpo(learning_rate=0.0)
    _rollout(alg, obs_scale=5.0)
    flat = lambda x: x.flatten(0, 1)  # noqa: E731
    new_loss, _, _ = alg.actor.get_cfm_loss_from_latent(
        flat(alg.fpo_latent), flat(alg.storage.actions), flat(alg.fpo_eps), flat(alg.fpo_t)
    )
    torch.testing.assert_close(new_loss, flat(alg.fpo_old_loss))
    out = alg.update()
    assert math.isclose(out["fpo/ratio_mean"], 1.0, abs_tol=1e-6)
    assert out["fpo/clip_frac"] == 0.0
    assert out["fpo/log_ratio_abs_max"] < 1e-5  # batched-GEMM rounding only


def test_surrogate_modes_hand_example():
    alg = _fpo(clip_param=0.1)
    ratio = torch.tensor([[1.2, 0.8]])
    pos, neg = torch.tensor([[2.0]]), torch.tensor([[-2.0]])
    expected = {
        ("ppo", 1): -1.9, ("spo", 1): -1.6, ("aspo", 1): -1.9,
        ("ppo", -1): 2.1, ("spo", -1): 2.4, ("aspo", -1): 2.4,
    }  # fmt: skip
    for (mode, sign), value in expected.items():
        alg.trust_region_mode = mode
        got = alg._surrogate_loss(ratio, pos if sign > 0 else neg)
        assert math.isclose(got.item(), value, rel_tol=1e-6), (mode, sign, got.item(), value)


def test_clamp_ste():
    x = torch.tensor([0.5, 3.0, 12.0], requires_grad=True)
    y = clamp_ste(x, max=10.0)
    assert y.tolist() == [0.5, 3.0, 10.0]
    y.sum().backward()
    assert x.grad.tolist() == [1.0, 1.0, 1.0]


def test_zero_noise_action_is_deterministic():
    torch.manual_seed(2)
    obs = _obs()
    actor = _actor(obs)
    a1 = actor(obs)
    a2 = actor(obs)
    torch.testing.assert_close(a1, a2)
    actor.train()
    s1 = actor(obs, stochastic_output=True)
    torch.testing.assert_close(actor.output_mean, a1)
    assert not torch.allclose(s1, a1)
    assert actor.output_std.shape == (ACT_DIM,) and (actor.output_std > 0).all()


def test_jit_export_matches_zero_noise_forward():
    torch.manual_seed(3)
    obs = _obs()
    actor = _actor(obs)
    actor.update_normalization(_obs(64, scale=3.0))
    actor.eval()
    scripted = torch.jit.script(actor.as_jit())
    torch.testing.assert_close(scripted(obs["policy"]), actor(obs))


def test_update_changes_params_and_is_finite():
    torch.manual_seed(4)
    alg = _fpo()
    before = [p.detach().clone() for p in alg.actor.parameters()]
    _rollout(alg)
    out = alg.update()
    assert all(math.isfinite(v) for v in out.values())
    for k in ("value", "surrogate", "entropy", "fpo/ratio_mean", "fpo/clip_frac", "fpo/cfm_loss_mean"):
        assert k in out
    assert any(not torch.equal(b, p) for b, p in zip(before, alg.actor.parameters()))


def test_knn_entropy_runs():
    torch.manual_seed(5)
    alg = _fpo(knn_entropy_coef=0.01, knn_entropy_k=1)
    _rollout(alg)
    out = alg.update()
    assert math.isfinite(out["entropy"]) and out["entropy"] != 0.0


def test_adaptive_schedule_uses_x1_proxy():
    torch.manual_seed(6)
    alg = _fpo(schedule="adaptive", desired_kl=1e-12)
    _rollout(alg)
    out = alg.update()
    assert "fpo/kl_proxy" in out and out["fpo/kl_proxy"] > 0
    assert alg.learning_rate < 1e-3  # the proxy far exceeds 2 * desired_kl -> lr decays


def test_eval_pool_executes_zero_noise_action():
    torch.manual_seed(7)
    obs = _obs()
    actor = _actor(obs)
    critic = MLPModel(obs, OBS_GROUPS, "critic", 1, hidden_dims=[16])
    storage = RolloutStorage("rl", NUM_ENVS, STEPS, obs, [ACT_DIM], "cpu", num_eval_envs=2)
    alg = FPO(actor, critic, storage, n_samples_per_action=4, eval_env_fraction=0.25, learning_rate=1e-3)
    alg.train_mode()
    actions = alg.act(obs)
    torch.testing.assert_close(actions[-2:], actor.output_mean[-2:])
    assert not torch.allclose(actions[:-2], actor.output_mean[:-2])


def test_save_load_roundtrip_with_ema():
    torch.manual_seed(8)
    alg = _fpo(ema_warmup_steps=1)
    for _ in range(3):
        _rollout(alg)
        alg.update()
    assert alg._ema_active()
    saved = alg.save()
    assert "ema_state_dict" in saved and saved["ema_state_dict"]["tot_timesteps"] == 3

    fresh = _fpo(ema_warmup_steps=1)
    fresh.load(saved, None, strict=True)
    assert fresh.tot_timesteps == 3
    for name, value in alg.ema.shadow_params.items():
        torch.testing.assert_close(fresh.ema.shadow_params[name], value)
    for p, q in zip(alg.actor.parameters(), fresh.actor.parameters()):
        torch.testing.assert_close(p, q)

    # eval-mode policy carries the EMA weights, the training actor keeps its own
    fresh.eval_mode()
    policy = fresh.get_policy()
    assert policy is not fresh.actor
    for name, p in policy.named_parameters():
        torch.testing.assert_close(p, alg.ema.shadow_params[name])
    assert any(not torch.equal(p, q) for p, q in zip(policy.parameters(), fresh.actor.parameters()))
    fresh.train_mode()
    assert fresh.get_policy() is fresh.actor


def test_construct_algorithm():
    class _Env:
        num_envs = NUM_ENVS
        num_actions = ACT_DIM

    obs = _obs()
    cfg = {
        "algorithm": {"class_name": "rsl_rl.algorithms:FPO", "n_samples_per_action": 4, "learning_rate": 1e-4},
        "actor": {"class_name": "rsl_rl.models:FlowMLPModel", "hidden_dims": [16], "sampling_steps": 4},
        "critic": {"class_name": "rsl_rl.models:MLPModel", "hidden_dims": [16]},
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": STEPS,
        "multi_gpu": None,
    }
    alg = FPO.construct_algorithm(obs, _Env(), cfg, "cpu")
    assert isinstance(alg, FPO) and isinstance(alg.optimizer, torch.optim.AdamW)
    assert alg.fpo_eps.shape == (STEPS, NUM_ENVS, 4, ACT_DIM)


def test_bandit_moves_zero_noise_action_to_target():
    # One-step bandit, reward -(a - 0.7)^2 on a 1-D action.
    torch.manual_seed(9)
    n, steps, target = 64, 4, 0.7
    alg = _fpo(
        num_envs=n,
        steps=steps,
        act_dim=1,
        actor_kw=dict(obs_normalization=False, hidden_dims=[32, 32]),
        n_samples_per_action=8,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=3e-3,
        clip_param=0.05,
        ema_decay=0.0,
    )
    obs = TensorDict({"policy": torch.ones(n, OBS_DIM)}, batch_size=[n])
    alg.train_mode()

    def zero_noise() -> float:
        with torch.no_grad():
            return alg.actor(obs[:1]).item()

    start = zero_noise()
    for _ in range(300):
        with torch.inference_mode():
            for _ in range(steps):
                a = alg.act(obs)
                alg.process_env_step(obs, -((a[:, 0] - target) ** 2), torch.ones(n), {})
            alg.compute_returns(obs)
        alg.update()
    end = zero_noise()
    assert abs(end - target) < 0.1, (start, end)
    assert abs(end - target) < abs(start - target)
