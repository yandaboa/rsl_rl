# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""SAPG (play2perfect port: whole-block relabeling, per-block std, joint loss / adv norm / epoch KL); CPU.

source /home/yandabao/miniforge3/etc/profile.d/conda.sh && conda activate patlab3
PYTHONPATH=. python -m pytest tests/test_sapg.py -q
"""

from __future__ import annotations

import copy
import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO, SAPG
from rsl_rl.models import MLPEncoderModel, MLPModel, block_context
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage
from tests.runners.test_on_policy_runner import DummyEnv

NUM_ENVS = 12
NUM_STEPS = 6
OBS_DIM = 8
PRIV_DIM = 5
NUM_ACTIONS = 3
OBS_GROUPS = {"actor": ["policy", "privileged"], "critic": ["policy", "privileged"]}
GAUSSIAN = {"class_name": "GaussianDistribution", "init_std": 0.8, "std_type": "log"}
GSDE = {"class_name": "GsdeDistribution", "init_std": 0.8, "std_type": "log"}
SAPG_KEYS = (
    "per_block_std",
    "off_policy_ratio",
    "sapg_entropy_scale",
    "leader_entropy_coef",
    "kl_rows",
    "kl_reference",
    "log_off_policy_grads_every",
    "relabel_chunk_size",
)


def _obs(num_envs: int = NUM_ENVS, generator: torch.Generator | None = None) -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(num_envs, OBS_DIM, generator=generator),
            "privileged": torch.randn(num_envs, PRIV_DIM, generator=generator),
        },
        batch_size=[num_envs],
    )


def _models(kind: str = "mlp", dist: dict = GAUSSIAN, seed: int = 0, norm: bool = True) -> tuple[MLPModel, MLPModel]:
    torch.manual_seed(seed)
    obs = _obs()
    if kind == "mlp":
        actor = MLPModel(
            obs,
            OBS_GROUPS,
            "actor",
            NUM_ACTIONS,
            hidden_dims=[32, 16],
            obs_normalization=norm,
            distribution_cfg=dict(dist),
        )
        critic = MLPModel(obs, OBS_GROUPS, "critic", 1, hidden_dims=[32, 16], obs_normalization=norm)
    else:
        enc = {"privileged": {"output_dim": 4, "hidden_dims": [16], "activation": "elu"}}
        kw = dict(hidden_dims=[32, 16], obs_normalization=norm, encoder_normalization=norm, head_layer_norm=True)
        actor = MLPEncoderModel(
            obs, OBS_GROUPS, "actor", NUM_ACTIONS, distribution_cfg=dict(dist), encoder_cfg=copy.deepcopy(enc), **kw
        )
        critic = MLPEncoderModel(obs, OBS_GROUPS, "critic", 1, encoder_cfg=copy.deepcopy(enc), **kw)
    return actor, critic


def _alg_kwargs(**overrides) -> dict:
    kwargs = dict(
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1e-3,
        schedule="adaptive",
        desired_kl=0.01,
        entropy_coef=0.006,
    )
    kwargs.update(overrides)
    return kwargs


def _storage(num_envs: int = NUM_ENVS) -> RolloutStorage:
    return RolloutStorage("rl", num_envs, NUM_STEPS, _obs(num_envs), [NUM_ACTIONS])


def _sapg(
    num_blocks: int = 3,
    phi_dim: int = 4,
    kind: str = "mlp",
    seed: int = 0,
    dist: dict = GAUSSIAN,
    norm: bool = True,
    **overrides,
) -> SAPG:
    actor, critic = _models(kind, dist, seed, norm)
    sapg_kw = {k: overrides.pop(k) for k in list(overrides) if k in SAPG_KEYS}
    return SAPG(
        actor, critic, _storage(), num_blocks=num_blocks, phi_dim=phi_dim, **sapg_kw, **_alg_kwargs(**overrides)
    )


def _collect(alg: PPO, seed: int = 1, inference: bool = True) -> None:
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed + 100)
    obs = _obs(generator=generator)
    ctx = torch.inference_mode() if inference else torch.no_grad()
    with ctx:
        for _ in range(NUM_STEPS):
            alg.act(obs)
            obs = _obs(generator=generator)
            rewards = torch.randn(NUM_ENVS, generator=generator)
            dones = (torch.rand(NUM_ENVS, generator=generator) < 0.25).float()
            time_outs = dones * (torch.rand(NUM_ENVS, generator=generator) < 0.5).float()
            alg.process_env_step(obs, rewards, dones, {"time_outs": time_outs})
        alg.compute_returns(obs)


def _randomize_blocks(alg: SAPG) -> None:
    """Distinct per-block phi and std so the blocks are genuinely different policies."""
    with torch.no_grad():
        for model in (alg.actor, alg.critic):
            if model.sapg_phi is not None:
                model.sapg_phi.normal_(0.0, 1.0)
                model.sapg_linear.phi_weight.normal_(0.0, 0.5)
        dist = alg.actor.distribution
        if hasattr(dist, "sapg_follower_std"):
            dist.sapg_follower_std.add_(torch.randn_like(dist.sapg_follower_std) * 0.3)


def _off_env_time(alg: SAPG) -> tuple[torch.Tensor, torch.Tensor]:
    src = alg._rows["src"][alg._rows["off"]]
    return src % NUM_ENVS, src // NUM_ENVS


# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dist", [GAUSSIAN, GSDE])
def test_single_block_reproduces_ppo(dist: dict) -> None:
    """M=1, no phi: PPO's returns, advantages, losses and parameters (fixed LR: SAPG adapts per epoch, PPO per mb)."""
    actor, critic = _models("mlp", dist)
    kw = _alg_kwargs(schedule="fixed")
    ppo = PPO(copy.deepcopy(actor), copy.deepcopy(critic), _storage(), **kw)
    sapg = SAPG(
        copy.deepcopy(actor),
        copy.deepcopy(critic),
        _storage(),
        num_blocks=1,
        phi_dim=0,
        leader_entropy_coef=0.006,
        **kw,
    )
    for alg in (ppo, sapg):
        _collect(alg)
    assert torch.equal(ppo.storage.returns, sapg.storage.returns)
    assert torch.equal(ppo.storage.advantages, sapg.storage.advantages)
    torch.manual_seed(7)
    ppo_losses = ppo.update()
    torch.manual_seed(7)
    sapg_losses = sapg.update()
    for key in ("value", "surrogate", "entropy"):
        assert ppo_losses[key] == pytest.approx(sapg_losses[key], rel=1e-6, abs=1e-7), key
    for (name, p), (_, q) in zip(
        list(ppo.actor.named_parameters()) + list(ppo.critic.named_parameters()),
        list(sapg.actor.named_parameters()) + list(sapg.critic.named_parameters()),
    ):
        assert torch.allclose(p, q, atol=1e-6), name


def test_block_layout_and_entropy_schedule() -> None:
    sapg = _sapg(num_blocks=3)
    assert sapg.env_block_ids.tolist() == [0] * 4 + [1] * 4 + [2] * 4
    with pytest.raises(AssertionError):
        _sapg(num_blocks=5)  # 12 envs
    six = _sapg(num_blocks=6, sapg_entropy_scale=0.002)
    # play2perfect linspace(0.5, 0, M) * scale, leader (theirs last, ours first) at leader_entropy_coef = 0.
    assert six.block_entropy_coefs == pytest.approx([0.0, 0.0002, 0.0004, 0.0006, 0.0008, 0.001])
    with pytest.raises(AssertionError):
        _sapg(num_blocks=3, eval_env_fraction=0.25)


@pytest.mark.parametrize("num_blocks", [3, 4])
def test_relabel_set_is_one_whole_follower_block(num_blocks: int) -> None:
    """off_policy_ratio 1: every timestep of every env of exactly one follower block, T x block rows."""
    seen = set()
    for seed in range(40):
        sapg = _sapg(num_blocks=num_blocks)
        torch.manual_seed(seed)
        _collect(sapg, seed=seed)
        env, t = _off_env_time(sapg)
        blocks = sapg.env_block_ids[env].unique()
        assert blocks.numel() == 1 and int(blocks) > 0
        block_envs = torch.nonzero(sapg.env_block_ids == blocks).view(-1)
        assert env.numel() == NUM_STEPS * block_envs.numel()
        pairs = set(zip(env.tolist(), t.tolist()))
        assert pairs == {(e, s) for e in block_envs.tolist() for s in range(NUM_STEPS)}
        assert bool((sapg._rows["block"][sapg._rows["off"]] == 0).all())
        seen.add(int(blocks))
    assert seen == set(range(1, num_blocks))  # every follower gets drawn


def test_off_policy_ratio_zero_and_two() -> None:
    sapg = _sapg(num_blocks=4, off_policy_ratio=0.0)
    _collect(sapg)
    assert not bool(sapg._rows["off"].any())
    sapg = _sapg(num_blocks=4, off_policy_ratio=2.0)
    _collect(sapg)
    env, _ = _off_env_time(sapg)
    assert sapg.env_block_ids[env].unique().numel() == 2
    assert env.numel() == 2 * NUM_STEPS * 3


def test_relabeled_rows_use_follower_behavior_and_leader_targets() -> None:
    """Behavior log-prob / distribution = the follower's stored ones; old value V_L(s); y = 1-step TD on V_L."""
    sapg = _sapg(num_blocks=3, relabel_chunk_size=5)
    _randomize_blocks(sapg)
    _collect(sapg)
    st, rows = sapg.storage, sapg._rows
    off = rows["off"]
    env, t = _off_env_time(sapg)
    src = rows["src"][off]
    assert torch.equal(st.actions_log_prob.flatten(0, 1)[src], st.actions_log_prob[t, env])
    assert torch.equal(st.distribution_params[1].flatten(0, 1)[src], st.distribution_params[1][t, env])

    envs = env.view(NUM_STEPS, -1)[0]
    with torch.no_grad(), block_context(None, sapg.actor, sapg.critic):
        v_lead = torch.stack([sapg.critic(st.observations[s][envs]) for s in range(NUM_STEPS)])
        # The leader evaluates the follower's actions with a different distribution than the one that sampled them.
        sapg.actor(st.observations[0][envs], stochastic_output=True)
        leader_logp = sapg.actor.get_output_log_prob(st.actions[0, envs])
    assert not torch.allclose(leader_logp, st.actions_log_prob[0, envs].view(-1), atol=1e-3)
    assert torch.allclose(rows["values"][off].view(NUM_STEPS, -1, 1), v_lead, atol=1e-6)
    r = st.rewards[:, envs] + sapg.gamma * sapg._time_outs[:, envs] * (v_lead - st.values[:, envs])
    y = rows["returns"][off].view(NUM_STEPS, -1, 1)
    not_done = 1.0 - st.dones[:, envs].float()
    assert torch.allclose(y[:-1], r[:-1] + not_done[:-1] * sapg.gamma * v_lead[1:], atol=1e-5)


def test_joint_advantage_normalization() -> None:
    """One normalization over on-policy rows of every block + relabeled rows."""
    raw = _sapg(num_blocks=3, normalize_advantage_per_mini_batch=True)
    _collect(raw)
    joint = _sapg(num_blocks=3)
    _collect(joint)
    assert torch.equal(raw._rows["src"], joint._rows["src"])
    a = raw._rows["advantages"]
    expected = (a - a.mean()) / (a.std() + 1e-8)
    assert torch.allclose(joint._rows["advantages"], expected, atol=1e-6)
    num_on = NUM_STEPS * NUM_ENVS
    on = a[:num_on]
    assert not torch.allclose(joint._rows["advantages"][:num_on], (on - on.mean()) / (on.std() + 1e-8), atol=1e-4)
    assert torch.equal(joint.storage.advantages.flatten(0, 1), joint._rows["advantages"][:num_on])


@pytest.mark.parametrize("dist", [GAUSSIAN, GSDE])
def test_per_block_std_gradients(dist: dict) -> None:
    """Block j's std gets gradient only from block j's rows; the leader's also from the relabeled rows."""
    sapg = _sapg(num_blocks=3, dist=dist)
    _randomize_blocks(sapg)
    _collect(sapg)
    dist_mod = sapg.actor.distribution
    rows = sapg._rows
    st = sapg.storage
    flat = {
        "observations": st.observations.flatten(0, 1),
        "actions": st.actions.flatten(0, 1),
        "actions_log_prob": st.actions_log_prob.flatten(0, 1),
    }
    old = tuple(p.flatten(0, 1)[rows["src"]] for p in st.distribution_params)

    def grads(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.nonzero(mask).view(-1)
        terms = sapg.minibatch_losses(idx, flat, tuple(p[idx] for p in old))
        sapg.optimizer.zero_grad()
        (terms["surrogate_rows"].sum() - terms["entropy_rows"].sum()).backward()
        return dist_mod.log_std_param.grad.clone(), dist_mod.sapg_follower_std.grad.clone()

    lead, fol = grads(rows["off"])
    assert lead.abs().sum() > 0 and torch.equal(fol, torch.zeros_like(fol))
    for j in (1, 2):
        lead, fol = grads((rows["block"] == j) & ~rows["off"])
        assert torch.equal(lead, torch.zeros_like(lead))
        assert fol[j - 1].abs().sum() > 0
        assert torch.equal(fol[2 - j], torch.zeros_like(fol[2 - j]))


def test_phi_is_randn_and_first_layer_is_one_fresh_linear() -> None:
    """play2perfect extra_params: phi ~ N(0, 1) per block (actor and critic tables differ)."""
    sapg = _sapg(num_blocks=6, phi_dim=32)
    for model in (sapg.actor, sapg.critic):
        phi = model.sapg_phi.detach()
        assert phi.shape == (6, 32)
        assert 0.7 < float(phi.std()) < 1.3
        linear = model.sapg_linear
        bound = 1.0 / (linear.in_features + linear.phi_dim) ** 0.5
        assert float(linear.weight.abs().max()) <= bound and float(linear.phi_weight.abs().max()) <= bound
        assert float(linear.phi_weight.abs().max()) > 0.5 * bound
    assert not torch.equal(sapg.actor.sapg_phi, sapg.critic.sapg_phi)


def test_rollout_samples_with_each_blocks_std() -> None:
    sapg = _sapg(num_blocks=3)
    with torch.no_grad():
        sapg.actor.distribution.log_std_param.fill_(-1.0)
        sapg.actor.distribution.sapg_follower_std[0].fill_(0.0)
        sapg.actor.distribution.sapg_follower_std[1].fill_(0.5)
    _collect(sapg)
    std = sapg.storage.distribution_params[1]
    for j, log_std in enumerate((-1.0, 0.0, 0.5)):
        rows = std[:, sapg.env_block_ids == j]
        assert torch.allclose(rows, torch.full_like(rows, float(torch.tensor(log_std).exp())))


def test_entropy_coef_per_row() -> None:
    sapg = _sapg(num_blocks=3, sapg_entropy_scale=0.01, leader_entropy_coef=0.0)
    _collect(sapg)
    rows = sapg._rows
    coef = rows["entropy_coef"]
    assert bool((coef[rows["off"]] == 0).all())
    for j, c in enumerate([0.0, 0.0025, 0.005]):
        assert torch.allclose(coef[(rows["block"] == j) & ~rows["off"]], torch.tensor(c))


@pytest.mark.parametrize("kl_rows", ["all", "leader"])
def test_learning_rate_adapts_once_per_epoch(kl_rows: str) -> None:
    sapg = _sapg(num_blocks=3, kl_rows=kl_rows, num_learning_epochs=3, num_mini_batches=2)
    _randomize_blocks(sapg)
    _collect(sapg)
    calls = []
    original = sapg._adapt_learning_rate

    def spy(kl):
        calls.append(float(kl))
        original(kl)

    sapg._adapt_learning_rate = spy  # type: ignore[method-assign]
    losses = sapg.update()
    assert len(calls) == 3
    key = "SAPG/kl_all" if kl_rows == "all" else "SAPG/kl_leader"
    assert sum(calls) / 3 == pytest.approx(losses[key], rel=1e-5)
    # Epoch 0 compares the relabeled rows with the FOLLOWER's behavior distribution: far larger than the leader's.
    assert losses["SAPG/kl_relabeled"] > losses["SAPG/kl_leader"]


def test_kl_reference_previous_epoch_overwrites_stored_distribution() -> None:
    """play2perfect update_mu_sigma: with a zero LR, epochs >= 1 measure KL = 0 (the stored params are current)."""
    sapg = _sapg(num_blocks=3, learning_rate=0.0, schedule="fixed", num_learning_epochs=2)
    _randomize_blocks(sapg)
    _collect(sapg)
    kls = []
    original = sapg.minibatch_losses

    def spy(idx, flat, old):
        terms = original(idx, flat, old)
        kls.append(terms["kl"].mean().item())
        return terms

    sapg.minibatch_losses = spy  # type: ignore[method-assign]
    sapg.update()
    assert kls[0] > 1e-4 and kls[1] > 1e-4  # epoch 0: relabeled rows vs the follower
    assert kls[2] == pytest.approx(0.0, abs=1e-6) and kls[3] == pytest.approx(0.0, abs=1e-6)


def test_checkpoint_roundtrip_and_leader_inference(tmp_path) -> None:
    sapg = _sapg(num_blocks=3, phi_dim=4, kind="encoder")
    _randomize_blocks(sapg)
    _collect(sapg)
    sapg.update()
    saved = copy.deepcopy(sapg.save())
    other = _sapg(num_blocks=3, phi_dim=4, kind="encoder", seed=99)
    other.load(saved, None, strict=True)
    for a, b in ((sapg.actor, other.actor), (sapg.critic, other.critic)):
        assert torch.equal(a.sapg_phi, b.sapg_phi)
        assert torch.equal(a.sapg_linear.phi_weight, b.sapg_linear.phi_weight)
    assert torch.equal(sapg.actor.distribution.sapg_follower_std, other.actor.distribution.sapg_follower_std)
    assert torch.equal(sapg.actor.distribution.log_std_param, other.actor.distribution.log_std_param)

    obs = _obs(generator=torch.Generator().manual_seed(3))
    policy = other.get_policy()
    other.eval_mode()
    with torch.no_grad():
        leader = policy(obs)
        policy(obs, stochastic_output=True)
        leader_std = policy.output_std
        with block_context(torch.zeros(NUM_ENVS, dtype=torch.long), policy):
            assert torch.equal(policy(obs), leader)
            policy(obs, stochastic_output=True)
            assert torch.allclose(policy.output_std, leader_std)
        with block_context(torch.ones(NUM_ENVS, dtype=torch.long), policy):
            assert not torch.allclose(policy(obs), leader)
        jit = torch.jit.script(policy.as_jit())
        passthrough = torch.cat([obs[g] for g in policy.obs_groups], dim=-1)
        encoded = [obs[g].flatten(1) for g in policy.obs_groups_encoded]
        assert torch.allclose(jit(passthrough, encoded), leader, atol=1e-6)

    with pytest.raises(AssertionError):
        _sapg(num_blocks=2, phi_dim=4, kind="encoder").load(saved, None, strict=True)
    # No warm start from a plain PPO checkpoint.
    actor, critic = _models("encoder")
    plain = PPO(actor, critic, _storage(), **_alg_kwargs()).save()
    with pytest.raises(AssertionError):
        _sapg(num_blocks=3, phi_dim=4, kind="encoder").load(plain, None, strict=False)


def test_episode_metrics_per_block() -> None:
    sapg = _sapg(num_blocks=3, phi_dim=4)
    sapg.episode_success_fn = lambda: torch.arange(NUM_ENVS) % 2 == 0
    rewards = torch.arange(NUM_ENVS, dtype=torch.float32)
    sapg._track_episodes(rewards, torch.zeros(NUM_ENVS))
    sapg._track_episodes(rewards, torch.ones(NUM_ENVS))
    metrics = sapg.episode_metrics()
    assert metrics["SAPG/leader_return"] == pytest.approx(2 * (0 + 1 + 2 + 3) / 4)
    assert metrics["SAPG/block2_return"] == pytest.approx(2 * (8 + 9 + 10 + 11) / 4)
    assert metrics["SAPG/leader_success"] == pytest.approx(0.5)
    assert metrics["SAPG/leader_episode_length"] == pytest.approx(2.0)
    assert sapg.episode_metrics() == {}


def test_update_reports_metrics_and_grad_diagnostic() -> None:
    sapg = _sapg(num_blocks=3, log_off_policy_grads_every=2)
    _collect(sapg)
    losses = sapg.update()
    for key in (
        "SAPG/leader_return",
        "SAPG/kl_all",
        "SAPG/kl_leader",
        "SAPG/kl_relabeled",
        "SAPG/block2_action_std",
        "SAPG/relabel_rows",
        "SAPG/grad_cos_off_on",
        "SAPG/grad_cos_off_leader_on",
        "SAPG/grad_norm_off_over_on",
    ):
        assert key in losses, key
    assert losses["SAPG/relabel_rows"] == NUM_STEPS * 4
    _collect(sapg)
    assert "SAPG/grad_cos_off_on" not in sapg.update()  # every 2nd iteration


@pytest.mark.parametrize("dist", ["GaussianDistribution", "GsdeDistribution"])
def test_runner_learns_saves_and_loads(tmp_path, dist: str) -> None:
    cfg = {
        "num_steps_per_env": 8,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "SAPG",
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "num_blocks": 2,
            "phi_dim": 4,
        },
        "actor": {"class_name": "MLPModel", "hidden_dims": [16, 16], "distribution_cfg": {"class_name": dist}},
        "critic": {"class_name": "MLPModel", "hidden_dims": [16]},
    }
    runner = OnPolicyRunner(DummyEnv(), copy.deepcopy(cfg), log_dir=None, device="cpu")
    assert isinstance(runner.alg, SAPG)
    runner.learn(num_learning_iterations=2)
    assert runner.alg.episode_metrics() == {}  # drained into the loss dict by update()
    path = str(tmp_path / "model.pt")
    runner.save(path)
    other = OnPolicyRunner(DummyEnv(), copy.deepcopy(cfg), log_dir=None, device="cpu")
    other.load(path)
    assert torch.equal(runner.alg.actor.sapg_phi, other.alg.actor.sapg_phi)
    assert torch.equal(runner.alg.actor.distribution.sapg_follower_std, other.alg.actor.distribution.sapg_follower_std)
    runner.export_policy_to_jit(str(tmp_path), "policy.pt")
