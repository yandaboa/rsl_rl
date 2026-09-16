# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The trailing eval env pool of the base :class:`PPO` (``eval_env_fraction`` / ``eval_deterministic``)."""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tests.conftest import make_obs

NUM_ENVS = 8
NUM_STEPS = 6
OBS_DIM = 8
NUM_ACTIONS = 4
EVAL_FRACTION = 0.25  # -> 2 eval envs (6, 7), 6 training envs

GAUSSIAN = {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"}
TANH = {"class_name": "TanhGaussianDistribution", "init_std": 0.5, "std_type": "log", "std_range": (1e-3, 10.0)}


def _make_actor(obs: TensorDict, distribution_cfg: dict) -> MLPModel:
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    return MLPModel(
        obs,
        obs_groups,
        "actor",
        NUM_ACTIONS,
        hidden_dims=[32, 32],
        activation="elu",
        distribution_cfg=dict(distribution_cfg),
    )


def _unit_params() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(NUM_STEPS, NUM_ENVS, NUM_ACTIONS), torch.ones(NUM_STEPS, NUM_ENVS, NUM_ACTIONS)


def _make_critic(obs: TensorDict) -> MLPModel:
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    return MLPModel(obs, obs_groups, "critic", 1, hidden_dims=[32, 32], activation="elu")


def _build_ppo(eval_env_fraction: float, distribution_cfg: dict = GAUSSIAN, seed: int = 0, **overrides) -> PPO:
    torch.manual_seed(seed)
    obs = make_obs(NUM_ENVS, OBS_DIM)
    actor, critic = _make_actor(obs, distribution_cfg), _make_critic(obs)
    num_eval_envs = PPO._num_eval_envs(eval_env_fraction, NUM_ENVS)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS], num_eval_envs=num_eval_envs)
    defaults = dict(
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1e-3,
        schedule="fixed",
        desired_kl=None,
        eval_env_fraction=eval_env_fraction,
    )
    defaults.update(overrides)
    return PPO(actor, critic, storage, **defaults)


def _collect(ppo: PPO, seed: int = 1) -> None:
    """Fill the storage with a rollout whose rows are all distinct, then compute returns."""
    generator = torch.Generator().manual_seed(seed)
    obs = make_obs(NUM_ENVS, OBS_DIM)
    with torch.no_grad():
        for step in range(NUM_STEPS):
            ppo.act(obs)
            obs = TensorDict({"policy": torch.randn(NUM_ENVS, OBS_DIM, generator=generator)}, batch_size=[NUM_ENVS])
            rewards = torch.randn(NUM_ENVS, generator=generator)
            dones = (torch.rand(NUM_ENVS, generator=generator) < 0.2).float()
            ppo.process_env_step(obs, rewards, dones, {})
        ppo.compute_returns(obs)


# --------------------------------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------------------------------


def test_storage_generator_never_yields_an_eval_column() -> None:
    """Every training row is handed out once per epoch; no eval row ever; ``batch_size == num_train * T``."""
    num_eval = 2
    obs = make_obs(NUM_ENVS, OBS_DIM)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS], num_eval_envs=num_eval)
    assert storage.num_train_envs == NUM_ENVS - num_eval
    # Tag every row with its env id in the first action dim (eval rows are negative).
    env_tag = torch.arange(NUM_ENVS, dtype=torch.float32)
    env_tag[-num_eval:] = -1.0
    storage.actions[..., 0] = env_tag.view(1, -1)
    storage.distribution_params = _unit_params()
    storage.step = NUM_STEPS

    num_epochs, num_mini_batches = 3, 2
    seen = torch.zeros(NUM_ENVS)
    rows = 0
    for batch in storage.mini_batch_generator(num_mini_batches, num_epochs):
        tags = batch.actions[:, 0]
        assert (tags >= 0).all(), f"an eval row reached the generator: {tags}"
        rows += tags.numel()
        for tag in tags.long():
            seen[tag] += 1
    assert rows == num_epochs * storage.num_train_envs * NUM_STEPS, "batch_size is not num_train_envs * T"
    assert seen[-num_eval:].sum() == 0
    assert torch.equal(seen[:-num_eval], torch.full((NUM_ENVS - num_eval,), float(num_epochs * NUM_STEPS)))


def test_recurrent_generator_never_yields_an_eval_column() -> None:
    num_eval = 2
    obs = make_obs(NUM_ENVS, OBS_DIM)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS], num_eval_envs=num_eval)
    env_tag = torch.arange(NUM_ENVS, dtype=torch.float32)
    env_tag[-num_eval:] = -1.0
    storage.actions[..., 0] = env_tag.view(1, -1)
    storage.distribution_params = _unit_params()
    storage.dones[2, :, 0] = 1
    storage.step = NUM_STEPS
    columns = 0
    for batch in storage.recurrent_mini_batch_generator(2, 1):
        assert (batch.actions[..., 0] >= 0).all(), "an eval column reached the recurrent generator"
        assert batch.observations.shape[1] == batch.masks.shape[1]
        columns += batch.actions.shape[1]
    assert columns == storage.num_train_envs


def test_storage_fraction_zero_is_the_stock_generator() -> None:
    """``num_eval_envs=0`` yields the same batches (same RNG stream) as before the pool existed."""
    obs = make_obs(NUM_ENVS, OBS_DIM)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    assert storage.num_eval_envs == 0 and storage.num_train_envs == NUM_ENVS
    storage.actions.normal_()
    storage.distribution_params = _unit_params()
    storage.step = NUM_STEPS
    torch.manual_seed(0)
    batches = [b.actions.clone() for b in storage.mini_batch_generator(2, 1)]
    torch.manual_seed(0)
    indices = torch.randperm(NUM_ENVS * NUM_STEPS)
    flat = storage.actions.flatten(0, 1)
    half = NUM_ENVS * NUM_STEPS // 2
    assert torch.equal(batches[0], flat[indices[:half]]) and torch.equal(batches[1], flat[indices[half:]])


# --------------------------------------------------------------------------------------------------
# Algorithm
# --------------------------------------------------------------------------------------------------


def test_pool_members() -> None:
    ppo = _build_ppo(EVAL_FRACTION)
    assert ppo.num_eval_envs == 2 and ppo.eval_env_ids.tolist() == [6, 7]
    assert ppo.eval_env_mask.tolist() == [False] * 6 + [True] * 2
    assert ppo._eval_mask is ppo.eval_env_mask
    zero = _build_ppo(0.0)
    assert zero.num_eval_envs == 0 and zero.eval_env_ids.numel() == 0 and zero.eval_env_mask is None


def test_storage_and_fraction_must_agree() -> None:
    obs = make_obs(NUM_ENVS, OBS_DIM)
    actor, critic = _make_actor(obs, GAUSSIAN), _make_critic(obs)
    storage = RolloutStorage("rl", NUM_ENVS, NUM_STEPS, obs, [NUM_ACTIONS])
    with pytest.raises(AssertionError):
        PPO(actor, critic, storage, eval_env_fraction=EVAL_FRACTION)
    with pytest.raises(AssertionError):
        PPO._num_eval_envs(1.0, NUM_ENVS)


@pytest.mark.parametrize("distribution_cfg", [GAUSSIAN, TANH], ids=["gaussian", "tanh"])
def test_act_swaps_pool_rows_to_the_deterministic_output(distribution_cfg: dict) -> None:
    ppo = _build_ppo(EVAL_FRACTION, distribution_cfg)
    obs = make_obs(NUM_ENVS, OBS_DIM)
    with torch.no_grad():
        actions = ppo.act(obs)
        expected = ppo.actor(obs)  # deterministic forward: ``distribution.deterministic_output``
        assert torch.equal(actions[6:], expected[6:]), "the pool did not execute the deterministic output"
        assert (actions[:6] - expected[:6]).abs().sum() > 0.0, "the training rows stopped sampling"
        # The stored log-prob is the log-prob of the action actually executed.
        ppo.actor(obs, stochastic_output=True)
        assert torch.allclose(ppo.transition.actions_log_prob, ppo.actor.get_output_log_prob(actions))


def test_act_leaves_train_rows_untouched() -> None:
    """With and without a pool, the training rows come off the same RNG stream."""
    with_pool = _build_ppo(EVAL_FRACTION)
    without = _build_ppo(0.0)
    obs = make_obs(NUM_ENVS, OBS_DIM)
    with torch.no_grad():
        torch.manual_seed(5)
        pooled = with_pool.act(obs)
        torch.manual_seed(5)
        stock = without.act(obs)
    assert torch.equal(pooled[:6], stock[:6])
    assert not torch.equal(pooled[6:], stock[6:])


def test_eval_deterministic_false_keeps_the_sample() -> None:
    sampling = _build_ppo(EVAL_FRACTION, eval_deterministic=False)
    stock = _build_ppo(0.0)
    obs = make_obs(NUM_ENVS, OBS_DIM)
    with torch.no_grad():
        torch.manual_seed(5)
        pooled = sampling.act(obs)
        torch.manual_seed(5)
        expected = stock.act(obs)
    assert torch.equal(pooled, expected), "eval_deterministic=False must not touch any action"
    # ... but the pool is still excluded from the update.
    _collect(sampling)
    for batch in sampling.storage.mini_batch_generator(2, 1):
        assert batch.observations.shape[0] == 6 * NUM_STEPS // 2


def test_normalization_ignores_the_pool() -> None:
    ppo = _build_ppo(EVAL_FRACTION)
    _collect(ppo)
    st = ppo.storage
    # Undo the normalisation: the raw advantages are returns - values.
    raw = st.returns - st.values
    train = raw[:, :6]
    expected = (raw - train.mean()) / (train.std() + 1e-8)
    assert torch.allclose(st.advantages, expected, atol=1e-6)
    assert abs(st.advantages[:, :6].mean().item()) < 1e-5 and abs(st.advantages[:, :6].std().item() - 1.0) < 1e-3
    # Make the pool wildly different: the training statistics must not move.
    st.returns[:, 6:] += 1e3
    ppo_stats = (st.returns - st.values)[:, :6]
    assert torch.allclose(ppo_stats, train)


def test_fraction_zero_is_the_old_behaviour() -> None:
    """``eval_env_fraction=0`` reproduces the stock rollout, normalisation and update bit-for-bit."""

    def run(**kwargs) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        ppo = _build_ppo(**kwargs) if kwargs else _build_ppo(0.0)
        torch.manual_seed(11)
        _collect(ppo)
        stored = {
            "actions": ppo.storage.actions.clone(),
            "advantages": ppo.storage.advantages.clone(),
            "log_prob": ppo.storage.actions_log_prob.clone(),
        }
        torch.manual_seed(12)
        ppo.update()
        params = torch.cat([p.detach().flatten() for p in ppo.actor.parameters()])
        return stored, params

    explicit, explicit_params = run(eval_env_fraction=0.0, eval_deterministic=True)
    stock, stock_params = run(eval_env_fraction=0.0, eval_deterministic=False)
    for key, value in explicit.items():
        assert torch.equal(value, stock[key]), f"{key} differs with eval_env_fraction=0"
    assert torch.equal(explicit_params, stock_params)


def test_update_runs_with_a_pool() -> None:
    ppo = _build_ppo(EVAL_FRACTION)
    _collect(ppo)
    before = torch.cat([p.detach().flatten().clone() for p in ppo.actor.parameters()])
    losses = ppo.update()
    after = torch.cat([p.detach().flatten() for p in ppo.actor.parameters()])
    assert set(losses) >= {"value", "surrogate", "entropy"} and not torch.equal(before, after)
