# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SAPGFlow (SAPG with a PolicyFlow leader sharing the followers' encoder); CPU.

source /home/yandabao/miniforge3/etc/profile.d/conda.sh && conda activate patlab3
PYTHONPATH=. python -m pytest tests/test_sapg_flow.py -q
"""

from __future__ import annotations

import copy
import torch
from tensordict import TensorDict

import pytest

from rsl_rl.algorithms import SAPG, SAPGFlow
from rsl_rl.models import BlockLinear, MLPModel
from rsl_rl.storage import RolloutStorage

NUM_ENVS = 10
NUM_STEPS = 4
OBS_DIM = 6
NUM_ACTIONS = 3
K = 4
OBS_GROUPS = {"actor": ["policy"], "critic": ["policy"]}
FLOW_CFG = {
    "class_name": "PolicyFlowModel",
    "hidden_dims": [16, 16],
    "embed_dim": 8,
    "sampling_steps": 3,
    "init_std": 1.0,
    "obs_normalization": False,
}


def _obs(num_envs: int = NUM_ENVS, generator: torch.Generator | None = None) -> TensorDict:
    return TensorDict({"policy": torch.randn(num_envs, OBS_DIM, generator=generator)}, batch_size=[num_envs])


def _models(seed: int = 0) -> tuple[MLPModel, MLPModel]:
    torch.manual_seed(seed)
    obs = _obs()
    dist = {"class_name": "GaussianDistribution", "init_std": 0.8, "std_type": "log"}
    actor = MLPModel(
        obs, OBS_GROUPS, "actor", NUM_ACTIONS, hidden_dims=[16, 16], obs_normalization=True, distribution_cfg=dist
    )
    critic = MLPModel(obs, OBS_GROUPS, "critic", 1, hidden_dims=[16, 16], obs_normalization=True)
    for model, out_gain in ((actor, 0.01), (critic, 1.0)):
        linears = [m for m in model.mlp if isinstance(m, torch.nn.Linear)]
        for i, layer in enumerate(linears):
            torch.nn.init.orthogonal_(layer.weight, out_gain if i == len(linears) - 1 else 2.0**0.5)
            torch.nn.init.zeros_(layer.bias)
    return actor, critic


def _alg(num_blocks: int = 2, off_policy_ratio: float = 1.0, flow_cfg: dict | None = None, **kw) -> SAPGFlow:
    kw.setdefault("target_kl", None)
    actor, critic = _models()
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, _obs(), [NUM_ACTIONS])
    kw.setdefault("num_latents", K)
    kw.setdefault("num_learning_epochs", 2)
    kw.setdefault("num_mini_batches", 2)
    return SAPGFlow(
        actor,
        critic,
        storage,
        flow_cfg=copy.deepcopy(flow_cfg or FLOW_CFG),
        num_blocks=num_blocks,
        phi_dim=0,
        shared_layers=1,
        per_block_std=True,
        off_policy_ratio=off_policy_ratio,
        off_policy_sampling="uniform",
        normalize_advantage_per_block=True,
        normalize_advantage_per_mini_batch=True,
        follower_entropy_coef=0.0,
        learning_rate=1e-3,
        schedule="fixed",
        desired_kl=None,
        entropy_coef=0.0,
        **kw,
    )


def _collect(alg: SAPGFlow, seed: int = 1) -> None:
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed + 100)
    obs = _obs(generator=generator)
    with torch.inference_mode():
        for _ in range(NUM_STEPS):
            alg.act(obs)
            obs = _obs(generator=generator)
            rewards = torch.randn(NUM_ENVS, generator=generator)
            dones = (torch.rand(NUM_ENVS, generator=generator) < 0.25).float()
            time_outs = dones * (torch.rand(NUM_ENVS, generator=generator) < 0.5).float()
            alg.process_env_step(obs, rewards, dones, {"time_outs": time_outs})
        alg.compute_returns(obs)


def _flat(alg: SAPGFlow) -> dict:
    st = alg.storage
    return {
        "observations": st.observations.flatten(0, 1),
        "actions": st.actions.flatten(0, 1),
        "actions_log_prob": st.actions_log_prob.flatten(0, 1),
    }


def _losses(alg: SAPGFlow, idx: torch.Tensor) -> dict:
    old = tuple(p.flatten(0, 1)[alg._rows["src"]][idx] for p in alg.storage.distribution_params)
    return alg.minibatch_losses(idx, _flat(alg), old)


def _heads(alg: SAPGFlow) -> list[BlockLinear]:
    return [m for m in alg.actor.mlp if isinstance(m, BlockLinear)]


def _follower_head_grad(alg: SAPGFlow) -> float:
    total = 0.0
    for head in _heads(alg):
        for p in (head.weight, head.bias):
            if p.grad is not None:
                total += float(p.grad[1:].abs().sum())
    dist = alg.actor.distribution
    if dist.sapg_follower_std.grad is not None:
        total += float(dist.sapg_follower_std.grad.abs().sum())
    return total


def _nonzero(p: torch.Tensor) -> bool:
    return p.grad is not None and float(p.grad.abs().sum()) > 0


def test_encoder_is_shared_and_not_in_flow_params() -> None:
    alg = _alg()
    shared = alg.actor.mlp[0]
    assert alg.flow_leader.encoder[1] is shared and alg.flow_leader.encoder[0] is alg.actor.obs_normalizer
    flow_ids = {id(p) for p in alg.flow_leader.parameters()}
    assert id(shared.weight) not in flow_ids and "encoder" not in str(list(alg.flow_leader.state_dict()))
    opt_ids = [id(p) for g in alg.optimizer.param_groups for p in g["params"]]
    assert len(opt_ids) == len(set(opt_ids)) and id(shared.weight) in opt_ids and flow_ids <= set(opt_ids)
    # the frozen copy owns its own encoder copy
    assert alg.flow_old.encoder[1] is not shared
    assert torch.equal(alg.flow_old.encoder[1].weight, shared.weight)


def test_leader_rows_only_touch_flow_and_shared_encoder() -> None:
    """(a) off_policy_ratio 0: empty off cache, finite loss; leader rows train flow + shared Linear, not followers."""
    alg = _alg(num_blocks=2, off_policy_ratio=0.0)
    _collect(alg)
    rows = alg._rows
    assert rows["off_z"].shape[0] == 0 and not bool(rows["off"].any()) and bool((rows["off_pos"] == -1).all())
    lead = torch.nonzero(rows["block"] == 0).view(-1)
    terms = _losses(alg, lead)
    assert torch.isfinite(terms["loss"])
    alg.optimizer.zero_grad()
    terms["loss"].backward()
    assert _follower_head_grad(alg) == 0.0
    assert _nonzero(alg.actor.mlp[0].weight)
    assert any(_nonzero(p) for p in alg.flow_leader.parameters())
    # the whole-batch loss is finite too
    terms = _losses(alg, torch.arange(rows["src"].shape[0]))
    assert torch.isfinite(terms["loss"]) and terms["brownian"].item() >= 0.0


def test_off_rows_gradients_and_no_grad_cache() -> None:
    """(b) flow_old / cache carry no grad; off rows train the flow leader + shared encoder only."""
    alg = _alg(num_blocks=5, off_policy_ratio=1.0)
    _collect(alg)
    rows = alg._rows
    for key in ("off_z", "off_phi", "off_ell", "off_log_mu"):
        assert not rows[key].requires_grad
    assert rows["off_z"].shape == (rows["off"].sum().item(), K, NUM_ACTIONS)
    assert all(not p.requires_grad for p in alg.flow_old.parameters())
    assert all(not p.requires_grad for m in alg.flow_old.encoder for p in m.parameters())
    # perturb the leader so the ratio differs from 1 and the advantage sign matters
    with torch.no_grad():
        for p in alg.flow_leader.parameters():
            p.add_(0.05 * torch.randn_like(p))
    off = torch.nonzero(rows["off"]).view(-1)
    terms = _losses(alg, off)
    alg.optimizer.zero_grad()
    terms["loss"].backward()
    assert _follower_head_grad(alg) == 0.0
    assert _nonzero(alg.actor.mlp[0].weight)
    assert any(_nonzero(p) for p in alg.flow_leader.parameters())
    assert all(p.grad is None for p in alg.flow_old.parameters())
    assert all(p.grad is None for m in alg.flow_old.encoder for p in m.parameters())


def test_mu_stats_sane() -> None:
    """(c) large leader std: finite log mu, ESS in (0, 1], effective latents in [1, K]."""
    alg = _alg(num_blocks=5, flow_cfg={**FLOW_CFG, "init_std": 3.0})
    _collect(alg)
    assert torch.isfinite(alg._rows["off_log_mu"]).all()
    assert bool((alg._rows["off_log_mu"] <= torch.log(torch.tensor(alg.mu_max)) + 1e-6).all())
    stats = alg._flow_stats
    assert 0.0 < stats["SAPGFlow/mu_ess"] <= 1.0 + 1e-6
    assert 1.0 - 1e-6 <= stats["SAPGFlow/eff_latents"] <= K + 1e-6
    for key in ("log_mu_mean", "log_mu_std", "mu_clip_frac", "log_pi_old_mean", "behavior_logp_mean"):
        assert f"SAPGFlow/{key}" in stats


def test_marginal_ratio_is_one_right_after_sync() -> None:
    """(d) flow_leader == flow_old: log rho = 0 on every relabeled row, flow ratio = 1 on leader rows."""
    alg = _alg(num_blocks=5)
    _collect(alg)
    with torch.no_grad():
        terms = _losses(alg, torch.arange(alg._rows["src"].shape[0]))
    assert terms["off_log_rho"].numel() > 0
    assert terms["off_log_rho"].abs().max() < 1e-5
    lead_on = (terms["block"] == 0) & ~terms["off"]
    assert (terms["ratio"][lead_on] - 1.0).abs().max() < 1e-5


def test_act_writes_flow_actions_for_leader_envs() -> None:
    """(e) leader envs execute the flow sample (pf buffers filled); followers the Gaussian sample."""
    alg = _alg(num_blocks=2)
    n = alg.num_leader_envs
    obs = _obs()
    with torch.inference_mode():
        torch.manual_seed(5)
        actions = alg.act(obs).clone()
    assert torch.equal(alg.pf_phi[0] + alg.pf_noise[0], actions[:n])
    assert alg.pf_old_logp[0].abs().sum() > 0 and alg.pf_old_std[0].min() > 0
    assert torch.allclose(alg.transition.actions_log_prob[:n], alg.pf_old_logp[0])
    # the leader rows are not the Gaussian head's: replaying the follower path reproduces only rows >= n
    with torch.inference_mode():
        torch.manual_seed(5)
        gauss = SAPG.act(alg, obs).clone()
    assert torch.equal(gauss[n:], actions[n:])
    assert not torch.allclose(gauss[:n], actions[:n])


def test_update_runs_and_reports() -> None:
    alg = _alg(num_blocks=5, target_kl=10.0)
    _collect(alg)
    loss = alg.update()
    for key in (
        "SAPGFlow/brownian",
        "SAPGFlow/off_rho_mean",
        "SAPGFlow/off_clip_inside",
        "SAPGFlow/leader_noise_std",
        "SAPGFlow/leader_kl",
        "SAPGFlow/mu_ess",
        "grad_norm/flow_mean",
        "SAPG/block0_action_std",
    ):
        assert key in loss, key
    assert loss["SAPG/block0_action_std"] == pytest.approx(alg.flow_leader.std.mean().item())
    assert all(torch.isfinite(torch.tensor(v)) for v in loss.values())
    _collect(alg, seed=2)
    alg.update()


def test_awfm_loss_trains_flow_only_from_off_rows() -> None:
    """AWFM alone (lambda_off 0): positive finite fm loss; relabeled rows still train the flow leader."""
    alg = _alg(num_blocks=5, fm_coef=1.0, lambda_off=0.0)
    _collect(alg)
    off = torch.nonzero(alg._rows["off"]).view(-1)
    assert off.numel() > 0
    terms = _losses(alg, off)
    assert terms["fm_loss"].item() > 0.0 and torch.isfinite(terms["fm_w_mean"])
    alg.optimizer.zero_grad()
    terms["loss"].backward()
    assert _follower_head_grad(alg) == 0.0
    assert any(_nonzero(p) for p in alg.flow_leader.parameters())
    assert all(p.grad is None for p in alg.flow_old.parameters())
    loss = alg.update()
    assert torch.isfinite(torch.tensor(loss["SAPGFlow/fm_loss"])) and loss["SAPGFlow/fm_loss"] > 0.0
    assert torch.isfinite(torch.tensor(loss["SAPGFlow/fm_w_mean"]))


def test_awfm_off_by_default() -> None:
    alg = _alg(num_blocks=5)
    assert alg.fm_coef == 0.0
    _collect(alg)
    loss = alg.update()
    assert loss["SAPGFlow/fm_loss"] == 0.0 and loss["SAPGFlow/fm_w_mean"] == 0.0


def test_phi_spread_and_nearest_dist_stats() -> None:
    alg = _alg(num_blocks=5)
    _collect(alg)
    for key in ("SAPGFlow/phi_spread", "SAPGFlow/off_nearest_dist"):
        value = alg._flow_stats[key]
        assert torch.isfinite(torch.tensor(value)) and value >= 0.0, key


def test_checkpoint_roundtrip_restores_flow_leader() -> None:
    """(f) save -> load restores the flow leader (and the shared encoder via the actor)."""
    alg = _alg(num_blocks=2)
    _collect(alg)
    alg.update()
    saved = copy.deepcopy(alg.save())
    assert saved["sapg"]["flow_leader"] is True and saved["sapg"]["num_latents"] == K
    other = _alg(num_blocks=2)
    with torch.no_grad():
        for p in other.flow_leader.parameters():
            p.add_(1.0)
    other.load(saved, None, strict=True)
    for (name, a), b in zip(alg.flow_leader.state_dict().items(), other.flow_leader.state_dict().values()):
        assert torch.equal(a, b), name
    assert torch.equal(alg.actor.mlp[0].weight, other.actor.mlp[0].weight)
    assert other.flow_leader.encoder[1] is other.actor.mlp[0]
    obs = _obs()
    assert torch.allclose(alg.leader_policy(obs), other.leader_policy(obs))
    other.load(saved, {"actor": True}, strict=True)
    with pytest.raises(AssertionError):
        other.load({k: v for k, v in saved.items() if k != "flow_leader_state_dict"}, None, strict=True)


def test_requires_split_layout() -> None:
    actor, critic = _models()
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, _obs(), [NUM_ACTIONS])
    with pytest.raises(AssertionError):
        SAPGFlow(actor, critic, storage, flow_cfg=dict(FLOW_CFG), num_blocks=2, phi_dim=4, shared_layers=1)


def test_runner_learns_saves_and_loads(tmp_path) -> None:
    """``construct_algorithm`` path: the top-level ``flow_actor`` cfg reaches the algorithm; checkpoint round trip."""
    from rsl_rl.runners import OnPolicyRunner
    from tests.runners.test_on_policy_runner import DummyEnv

    cfg = {
        "num_steps_per_env": 8,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "SAPGFlow",
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "num_blocks": 2,
            "phi_dim": 0,
            "shared_layers": 1,
            "num_latents": 4,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [16, 16],
            "obs_normalization": True,
            "distribution_cfg": {"class_name": "GaussianDistribution"},
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [16, 16]},
        "flow_actor": {**FLOW_CFG, "stochastic": False},
    }
    for model in ("actor", "critic"):
        cfg[model]["class_name"] = "tests.test_sapg_flow:_OrthoMLPModel"
    runner = OnPolicyRunner(DummyEnv(), copy.deepcopy(cfg), log_dir=None, device="cpu")
    assert isinstance(runner.alg, SAPGFlow)
    runner.learn(num_learning_iterations=2)
    path = str(tmp_path / "model.pt")
    runner.save(path)
    other = OnPolicyRunner(DummyEnv(), copy.deepcopy(cfg), log_dir=None, device="cpu")
    other.load(path, load_cfg={"actor": True}, strict=True)
    for a, b in zip(runner.alg.flow_leader.parameters(), other.alg.flow_leader.parameters()):
        assert torch.equal(a, b)


RF_FLOW_CFG = {**FLOW_CFG, "integrator": "euler"}
GRAD_ENC_KEYS = (
    "follower",
    "leader_on",
    "leader_off",
    "leader_total",
    "leader_over_follower",
    "cos_leader_follower",
    "total",
)


def _lead_on(alg: SAPGFlow) -> torch.Tensor:
    rows = alg._rows
    return torch.nonzero((rows["block"] == 0) & ~rows["off"]).view(-1)


def test_encoder_grad_metric_reports_components() -> None:
    alg = _alg(num_blocks=5, fm_coef=1.0)
    _collect(alg)
    loss = alg.update()
    for key in GRAD_ENC_KEYS:
        assert f"grad_enc/{key}" in loss, key
        assert torch.isfinite(torch.tensor(loss[f"grad_enc/{key}"])), key
    assert loss["grad_enc/follower"] > 0.0
    assert loss["grad_enc/leader_on"] > 0.0
    assert loss["grad_enc/leader_off"] > 0.0
    assert -1.0 - 1e-6 <= loss["grad_enc/cos_leader_follower"] <= 1.0 + 1e-6


def test_encoder_grad_metric_off() -> None:
    alg = _alg(num_blocks=5, encoder_grad_minibatches=0)
    _collect(alg)
    loss = alg.update()
    assert not any(key.startswith("grad_enc/") for key in loss)


@pytest.mark.parametrize("leader_algo", ["policyflow", "fpo", "reinflow"])
def test_encoder_grad_total_is_sum_of_components(leader_algo: str) -> None:
    """One minibatch: ||g_total||^2 == ||g_F||^2 + ||g_L||^2 + 2 cos ||g_F|| ||g_L|| (critic shares nothing)."""
    flow_cfg = RF_FLOW_CFG if leader_algo == "reinflow" else FLOW_CFG
    alg = _alg(
        num_blocks=5,
        fm_coef=1.0,
        lambda_off=1.0 if leader_algo == "policyflow" else 0.0,
        leader_algo=leader_algo,
        flow_cfg=flow_cfg,
        num_learning_epochs=1,
        num_mini_batches=1,
        encoder_grad_minibatches=1,
    )
    _collect(alg)
    loss = alg.update()
    f, lead, cos = loss["grad_enc/follower"], loss["grad_enc/leader_total"], loss["grad_enc/cos_leader_follower"]
    expected = (f**2 + lead**2 + 2.0 * cos * f * lead) ** 0.5
    assert loss["grad_enc/total"] == pytest.approx(expected, rel=1e-4)
    assert loss["grad_enc/leader_off"] > 0.0 and loss["grad_enc/leader_on"] > 0.0


def test_policyflow_lambda_off_zero_skips_marginal_ratio() -> None:
    alg = _alg(num_blocks=5, fm_coef=1.0, lambda_off=0.0)
    _collect(alg)
    terms = _losses(alg, torch.arange(alg._rows["src"].shape[0]))
    assert terms["off_log_rho"].numel() == 0 and terms["off_mu"].numel() == 0
    off = terms["off"]
    assert bool((terms["ratio"][off] == 1.0).all()) and bool((terms["kl"][off] == 0.0).all())
    loss = alg.update()
    for key in ("SAPG/kl_relabeled", "SAPG/ratio_off_mean", "SAPGFlow/off_rho_mean", "SAPGFlow/off_clip_inside"):
        assert key not in loss, key
    assert loss["SAPGFlow/fm_loss"] > 0.0
    assert loss["grad_enc/leader_off"] > 0.0


def test_fpo_needs_lambda_off_zero() -> None:
    with pytest.raises(AssertionError):
        _alg(num_blocks=5, leader_algo="fpo", lambda_off=0.1)
    with pytest.raises(AssertionError):
        _alg(num_blocks=5, leader_algo="nope", lambda_off=0.0)


def test_fpo_leader_ratio_one_and_gradients() -> None:
    alg = _alg(num_blocks=5, leader_algo="fpo", lambda_off=0.0, fpo_samples=4, fpo_neg_adv_max_increase=0.5)
    n = alg.num_leader_envs
    _collect(alg)
    assert bool((alg.storage.actions_log_prob[:, :n] == 0.0).all())
    assert alg.fpo_old_loss.shape == (NUM_STEPS, n, 4) and bool((alg.fpo_old_loss > 0).all())
    lead = _lead_on(alg)
    assert lead.numel() > 0
    terms = _losses(alg, lead)
    assert (terms["ratio"] - 1.0).abs().max() < 1e-5
    assert terms["kl"].abs().max() < 1e-6
    assert terms["brownian"].item() == 0.0 and float(terms["entropy_raw"].abs().sum()) == 0.0
    alg.optimizer.zero_grad()
    terms["loss"].backward()
    assert _follower_head_grad(alg) == 0.0
    assert _nonzero(alg.actor.mlp[0].weight)
    assert any(_nonzero(p) for p in alg.flow_leader.parameters())
    assert alg.flow_leader.log_std.grad is None
    loss = alg.update()
    assert "SAPGFlow/fpo_cfm_loss" in loss and "SAPGFlow/fpo_neg_clamp_frac" in loss
    assert all(torch.isfinite(torch.tensor(v)) for v in loss.values())
    _collect(alg, seed=2)
    alg.update()


def test_fpo_act_is_noise_free_flow_sample() -> None:
    alg = _alg(num_blocks=2, leader_algo="fpo", lambda_off=0.0)
    n = alg.num_leader_envs
    obs = _obs()
    with torch.inference_mode():
        torch.manual_seed(7)
        actions = alg.act(obs).clone()
    flow = alg.flow_leader
    with torch.inference_mode():
        torch.manual_seed(7)
        SAPG.act(alg, obs)
        latent = flow.get_latent(obs[:n])
        expected = flow.integrate(latent, torch.randn(n, flow.num_actions))
    assert torch.allclose(actions[:n], expected, atol=1e-6)


def test_reinflow_requires_euler() -> None:
    with pytest.raises(AssertionError):
        _alg(num_blocks=5, leader_algo="reinflow", lambda_off=0.0)


def test_reinflow_ratio_one_update_and_head_trains() -> None:
    alg = _alg(num_blocks=5, leader_algo="reinflow", lambda_off=0.0, flow_cfg=RF_FLOW_CFG, fm_coef=0.5)
    n, steps = alg.num_leader_envs, RF_FLOW_CFG["sampling_steps"]
    head_params = list(alg.rf_noise_head.parameters())
    opt_ids = {id(p) for g in alg.optimizer.param_groups for p in g["params"]}
    assert all(id(p) in opt_ids for p in head_params)
    _collect(alg)
    assert alg.rf_chain.shape == (NUM_STEPS, n, steps + 1, NUM_ACTIONS)
    assert torch.equal(alg.storage.actions[:, :n], alg.rf_chain[:, :, -1])
    mid = 0.5 * (alg.reinflow_sigma_min + alg.reinflow_sigma_max)
    assert torch.allclose(alg.rf_old_sigma, torch.full_like(alg.rf_old_sigma, mid))
    # teacher-forced recompute reproduces the stored old log-likelihood
    obs = alg.storage.observations[:, :n].flatten(0, 1)
    _, _, logp = alg._rf_chain_stats(alg.flow_leader.get_latent(obs), alg.rf_chain.flatten(0, 1))
    assert torch.allclose(logp, alg.rf_old_logp.flatten(), atol=1e-4)
    terms = _losses(alg, _lead_on(alg))
    # _row_gaussian_kl's +1e-5 on the std ratio leaves ~1e-5 per (step, dim)
    assert (terms["ratio"] - 1.0).abs().max() < 1e-4 and terms["kl"].abs().max() < 2e-4
    assert bool((terms["entropy_raw"] != 0.0).all())
    alg.optimizer.zero_grad()
    terms["loss"].backward()
    assert _follower_head_grad(alg) == 0.0 and _nonzero(alg.actor.mlp[0].weight)
    before = [p.detach().clone() for p in head_params]
    loss = alg.update()
    assert alg.reinflow_sigma_min <= loss["SAPGFlow/reinflow_sigma_mean"] <= alg.reinflow_sigma_max
    assert all(torch.isfinite(torch.tensor(v)) for v in loss.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, head_params))


def test_reinflow_checkpoint_and_leader_policy() -> None:
    alg = _alg(num_blocks=2, leader_algo="reinflow", lambda_off=0.0, flow_cfg=RF_FLOW_CFG)
    _collect(alg)
    alg.update()
    saved = copy.deepcopy(alg.save())
    assert saved["sapg"]["leader_algo"] == "reinflow" and "reinflow_noise_head_state_dict" in saved
    other = _alg(num_blocks=2, leader_algo="reinflow", lambda_off=0.0, flow_cfg=RF_FLOW_CFG)
    with torch.no_grad():
        for p in other.rf_noise_head.parameters():
            p.add_(1.0)
    other.load(saved, None, strict=True)
    for a, b in zip(alg.rf_noise_head.parameters(), other.rf_noise_head.parameters()):
        assert torch.equal(a, b)
    with pytest.raises(AssertionError):
        other.load({k: v for k, v in saved.items() if k != "reinflow_noise_head_state_dict"}, None, strict=True)
    # leader_policy = the noise-free Euler x_K from x0 = 0
    obs = _obs()
    flow = other.flow_leader
    with torch.no_grad():
        latent = flow.get_latent(obs)
        x = torch.zeros(NUM_ENVS, NUM_ACTIONS)
        steps = flow.sampling_steps
        for i in range(steps):
            x = x + flow.velocity(latent, x, torch.full((NUM_ENVS,), i / steps)) / steps
    assert torch.allclose(other.leader_policy(obs), x, atol=1e-6)
    assert torch.allclose(alg.leader_policy(obs), other.leader_policy(obs))


class _OrthoMLPModel(MLPModel):
    """MLPModel with cleanRL orthogonal init (the per-block heads re-draw an orthogonal init)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for layer in self.mlp:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.orthogonal_(layer.weight, 1.0)
                torch.nn.init.zeros_(layer.bias)
