# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for trial-aware rollout storage, GAE and timeout bootstrapping.

Self-contained: pure torch, no Isaac Sim / SimulationApp needed. Run with::

    source /home/yandabao/miniforge3/etc/profile.d/conda.sh && conda activate patlab
    PYTHONPATH=/home/yandabao/rsl_rl-wt/meta-memory:$PYTHONPATH python tests/test_trial_aware_storage.py

A *trial* is K consecutive episodes sharing a hidden latent; memory persists across it. PPO must treat
episode boundaries inside a trial as non-terminal and only cut the bootstrap/trace at trial boundaries.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

# Configuration under test: K = 3 episodes of 80 steps => a 240-step trial
T_EPISODE = 80
K = 3
T_TRIAL = T_EPISODE * K
GAMMA = 0.999
LAM = 0.99
DEVICE = "cpu"


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _stock_compute_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    lam: float,
    normalize_advantage: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verbatim copy of upstream ``RolloutStorage.compute_returns`` (gated on episode dones).

    Kept identical op-for-op so the comparison below can be bit-for-bit.
    """
    num_transitions_per_env = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    advantage = 0
    for step in reversed(range(num_transitions_per_env)):
        next_values = last_values if step == num_transitions_per_env - 1 else values[step + 1]
        next_is_not_terminal = 1.0 - dones[step].float()
        delta = rewards[step] + next_is_not_terminal * gamma * next_values - values[step]
        advantage = delta + next_is_not_terminal * gamma * lam * advantage
        returns[step] = advantage + values[step]
    advantages = returns - values
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages


def _fill_storage(
    storage: RolloutStorage,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    trial_dones: torch.Tensor | None,
    obs: TensorDict,
    actions: torch.Tensor,
) -> None:
    """Push T transitions through the public ``add_transitions`` API."""
    storage.clear()
    transition = RolloutStorage.Transition()
    for step in range(rewards.shape[0]):
        transition.observations = obs
        transition.actions = actions
        transition.rewards = rewards[step].squeeze(-1)
        transition.dones = dones[step].squeeze(-1)
        # None here is the "caller does not know about trials" path
        transition.trial_dones = None if trial_dones is None else trial_dones[step].squeeze(-1)
        transition.values = values[step]
        transition.actions_log_prob = torch.zeros_like(rewards[step])
        transition.action_mean = actions
        transition.action_sigma = actions
        storage.add_transitions(transition)
        transition.clear()


def _make_storage(num_envs: int, obs: TensorDict, action_dim: int) -> RolloutStorage:
    return RolloutStorage("rl", num_envs, T_TRIAL, obs, [action_dim], device=DEVICE)


class _StubPolicy(nn.Module):
    """Minimal ActorCritic stand-in: no Isaac, no real network, fixed value output."""

    is_recurrent = False

    def __init__(self, num_envs: int, action_dim: int, value: float) -> None:
        super().__init__()
        self.dummy = nn.Linear(1, 1)
        self.num_envs = num_envs
        self.action_dim = action_dim
        self._value = value
        self.num_normalization_updates = 0
        self.normalization_batch_sizes: list[int] = []

    def act(self, obs: TensorDict) -> torch.Tensor:
        return torch.zeros(self.num_envs, self.action_dim, device=DEVICE)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        return torch.full((self.num_envs, 1), self._value, device=DEVICE)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=DEVICE)

    @property
    def action_mean(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, self.action_dim, device=DEVICE)

    @property
    def action_std(self) -> torch.Tensor:
        return torch.ones(self.num_envs, self.action_dim, device=DEVICE)

    def update_normalization(self, obs: TensorDict) -> None:
        self.num_normalization_updates += 1
        self.normalization_batch_sizes.append(obs.shape[0])

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass


# --------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------


def test_trace_survives_episode_boundaries_inside_a_trial() -> None:
    """A reward at step 100 must reach the advantage at step 79 iff step 79 is not a trial end."""
    num_envs, action_dim = 4, 3
    obs = TensorDict({"policy": torch.zeros(num_envs, 5)}, batch_size=[num_envs], device=DEVICE)
    actions = torch.zeros(num_envs, action_dim, device=DEVICE)

    torch.manual_seed(0)
    # Non-trivial values / rewards so that the check is not an artifact of a zeroed buffer
    values = torch.randn(T_TRIAL, num_envs, 1, device=DEVICE)
    base_rewards = torch.randn(T_TRIAL, num_envs, 1, device=DEVICE)
    last_values = torch.randn(num_envs, 1, device=DEVICE)

    # Episode dones at the last step of each 80-step episode: 79, 159, 239
    dones = torch.zeros(T_TRIAL, num_envs, 1, device=DEVICE)
    for k in range(1, K + 1):
        dones[k * T_EPISODE - 1] = 1.0
    # Trial done only at the very last step
    trial_dones = torch.zeros(T_TRIAL, num_envs, 1, device=DEVICE)
    trial_dones[T_TRIAL - 1] = 1.0

    probe_step, reward_step = T_EPISODE - 1, 100  # 79 -> 100, i.e. across the 80/160 episode boundary
    perturbation = 1.0

    def advantage_at_probe(trial_done_buffer: torch.Tensor, with_reward: bool) -> torch.Tensor:
        rewards = base_rewards.clone()
        if with_reward:
            rewards[reward_step] += perturbation
        storage = _make_storage(num_envs, obs, action_dim)
        _fill_storage(storage, rewards, values, dones, trial_done_buffer, obs, actions)
        storage.compute_returns(last_values, GAMMA, LAM, normalize_advantage=False)
        return storage.advantages[probe_step].clone()

    # Stock behavior: trial boundary == episode boundary -> the trace is cut at 79
    stock_sensitivity = advantage_at_probe(dones, True) - advantage_at_probe(dones, False)
    # Trial-aware: the trace runs through the episode boundary
    trial_sensitivity = advantage_at_probe(trial_dones, True) - advantage_at_probe(trial_dones, False)

    expected = perturbation * (GAMMA * LAM) ** (reward_step - probe_step)
    assert torch.allclose(stock_sensitivity, torch.zeros_like(stock_sensitivity), atol=1e-7), (
        f"episode-gated GAE must cut the trace at step {probe_step}, got {stock_sensitivity}"
    )
    assert torch.allclose(trial_sensitivity, torch.full_like(trial_sensitivity, expected), atol=1e-5), (
        f"trial-gated GAE must propagate {expected:.6f} to step {probe_step}, got {trial_sensitivity}"
    )
    assert trial_sensitivity.min().item() > 0.5, "sensitivity should be clearly non-zero, not numerical noise"

    # Same story for the returns themselves, and the cut must still happen at the trial end
    storage = _make_storage(num_envs, obs, action_dim)
    _fill_storage(storage, base_rewards, values, dones, trial_dones, obs, actions)
    storage.compute_returns(last_values, GAMMA, LAM, normalize_advantage=False)
    # At the trial end the advantage collapses to r - V(s) (no bootstrap, no trace)
    assert torch.allclose(
        storage.advantages[T_TRIAL - 1], base_rewards[T_TRIAL - 1] - values[T_TRIAL - 1], atol=1e-6
    ), "the trial end must cut both the bootstrap and the trace"

    summary = (
        f"[ok] trace survives episode boundaries: stock={stock_sensitivity.mean():.3e} "
        f"trial={trial_sensitivity.mean():.6f} (expected {expected:.6f})"
    )
    print(summary)


def test_default_trial_dones_reproduce_stock_returns_bitwise() -> None:
    """Never populating ``trial_dones`` must reproduce upstream ``compute_returns`` bit-for-bit."""
    num_envs, action_dim = 8, 3
    obs = TensorDict({"policy": torch.zeros(num_envs, 5)}, batch_size=[num_envs], device=DEVICE)
    actions = torch.zeros(num_envs, action_dim, device=DEVICE)

    torch.manual_seed(1)
    values = torch.randn(T_TRIAL, num_envs, 1, device=DEVICE)
    rewards = torch.randn(T_TRIAL, num_envs, 1, device=DEVICE)
    last_values = torch.randn(num_envs, 1, device=DEVICE)
    # Random, ragged episode boundaries (not aligned across envs)
    dones = (torch.rand(T_TRIAL, num_envs, 1, device=DEVICE) < 0.02).float()

    for normalize in (False, True):
        storage = _make_storage(num_envs, obs, action_dim)
        _fill_storage(storage, rewards, values, dones, None, obs, actions)
        assert torch.equal(storage.trial_dones, storage.dones), "trial_dones must default to dones"
        storage.compute_returns(last_values, GAMMA, LAM, normalize_advantage=normalize)

        ref_returns, ref_advantages = _stock_compute_returns(
            rewards, values, dones, last_values, GAMMA, LAM, normalize
        )
        assert torch.equal(storage.returns, ref_returns), f"returns differ from stock (normalize={normalize})"
        assert torch.equal(storage.advantages, ref_advantages), f"advantages differ from stock (normalize={normalize})"

    print("[ok] default trial_dones reproduce stock returns/advantages bit-for-bit")


def _run_ppo_steps(
    ppo: PPO,
    policy: _StubPolicy,
    num_envs: int,
    obs: TensorDict,
    raw_reward: float,
    steps: list[tuple[bool, bool]],
) -> torch.Tensor:
    """Run ``len(steps)`` collection steps; ``steps[i] = (time_out, trial_done)``. Returns stored rewards."""
    for time_out, trial_done in steps:
        ppo.act(obs)
        rewards = torch.full((num_envs,), raw_reward, device=DEVICE)
        dones = torch.ones(num_envs, dtype=torch.bool, device=DEVICE)  # every episode ends by timeout
        extras = {
            "time_outs": torch.full((num_envs,), time_out, dtype=torch.bool, device=DEVICE),
            "trial_done": torch.full((num_envs,), trial_done, dtype=torch.bool, device=DEVICE),
        }
        ppo.process_env_step(obs, rewards, dones, extras)
    return ppo.storage.rewards[: len(steps)].clone()


def test_timeout_bootstrap_only_at_trial_end() -> None:
    """time_outs at 80/160/240 but trial_done only at 240 -> the value is added exactly once."""
    num_envs, action_dim, value = 4, 3, 2.5
    raw_reward = 0.25
    obs = TensorDict({"policy": torch.zeros(num_envs, 5, device=DEVICE)}, batch_size=[num_envs], device=DEVICE)

    policy = _StubPolicy(num_envs, action_dim, value)
    ppo = PPO(policy, gamma=GAMMA, lam=LAM, device=DEVICE)
    ppo.init_storage("rl", num_envs, T_TRIAL, obs, [action_dim])

    # The three episode ends of one trial; only the last one closes the trial
    stored = _run_ppo_steps(ppo, policy, num_envs, obs, raw_reward, [(True, False), (True, False), (True, True)])

    bootstrap = GAMMA * value
    assert torch.allclose(stored[0], torch.full_like(stored[0], raw_reward)), "in-trial timeout must not bootstrap"
    assert torch.allclose(stored[1], torch.full_like(stored[1], raw_reward)), "in-trial timeout must not bootstrap"
    assert torch.allclose(stored[2], torch.full_like(stored[2], raw_reward + bootstrap)), (
        "the trial-ending timeout must bootstrap exactly once"
    )
    # Total reward added over the trial == exactly one bootstrap
    total_added = (stored.sum() - 3 * num_envs * raw_reward) / num_envs
    assert abs(total_added.item() - bootstrap) < 1e-5, f"expected one bootstrap ({bootstrap}), got {total_added}"

    # And the trial boundary made it into storage
    assert ppo.storage.trial_dones[0].sum() == 0 and ppo.storage.trial_dones[2].sum() == num_envs

    # Without "trial_done" in extras, the stock behavior (bootstrap on every timeout) is preserved,
    # because IsaacLab time_outs imply dones and trial_dones then defaults to dones.
    policy2 = _StubPolicy(num_envs, action_dim, value)
    ppo2 = PPO(policy2, gamma=GAMMA, lam=LAM, device=DEVICE)
    ppo2.init_storage("rl", num_envs, T_TRIAL, obs, [action_dim])
    for _ in range(3):
        ppo2.act(obs)
        ppo2.process_env_step(
            obs,
            torch.full((num_envs,), raw_reward, device=DEVICE),
            torch.ones(num_envs, dtype=torch.bool, device=DEVICE),
            {"time_outs": torch.ones(num_envs, dtype=torch.bool, device=DEVICE)},
        )
    legacy = ppo2.storage.rewards[:3]
    assert torch.allclose(legacy, torch.full_like(legacy, raw_reward + bootstrap)), (
        "without trial_done in extras, every timeout must bootstrap as before"
    )

    print("[ok] timeout bootstrap fires only at trial ends (and legacy path is unchanged)")


def test_deferred_obs_normalization() -> None:
    """``defer_obs_normalization`` moves the normalizer update out of the collection loop."""
    num_envs, action_dim, num_steps = 4, 3, 5
    obs = TensorDict({"policy": torch.zeros(num_envs, 5, device=DEVICE)}, batch_size=[num_envs], device=DEVICE)
    steps = [(False, False)] * num_steps

    # Default: one update per collection step (today's behavior)
    eager_policy = _StubPolicy(num_envs, action_dim, 0.0)
    eager = PPO(eager_policy, gamma=GAMMA, lam=LAM, device=DEVICE)
    eager.init_storage("rl", num_envs, T_TRIAL, obs, [action_dim])
    _run_ppo_steps(eager, eager_policy, num_envs, obs, 0.0, steps)
    assert eager_policy.num_normalization_updates == num_steps, "default must update the normalizer every step"
    assert eager_policy.normalization_batch_sizes == [num_envs] * num_steps

    # Deferred: nothing during collection, one commit afterwards over the whole rollout
    lazy_policy = _StubPolicy(num_envs, action_dim, 0.0)
    lazy = PPO(lazy_policy, gamma=GAMMA, lam=LAM, device=DEVICE, defer_obs_normalization=True)
    lazy.init_storage("rl", num_envs, T_TRIAL, obs, [action_dim])
    _run_ppo_steps(lazy, lazy_policy, num_envs, obs, 0.0, steps)
    assert lazy_policy.num_normalization_updates == 0, "deferred mode must not update during collection"

    # Committing over the [T, N, ...] rollout flattens the time dimension into the batch
    lazy.commit_obs_normalization(lazy.storage.observations[:num_steps])
    assert lazy_policy.num_normalization_updates == 1
    assert lazy_policy.normalization_batch_sizes == [num_steps * num_envs], (
        f"expected a flattened batch of {num_steps * num_envs}, got {lazy_policy.normalization_batch_sizes}"
    )

    print("[ok] deferred obs normalization: 0 updates during collection, 1 flattened commit after")


if __name__ == "__main__":
    test_trace_survives_episode_boundaries_inside_a_trial()
    test_default_trial_dones_reproduce_stock_returns_bitwise()
    test_timeout_bootstrap_only_at_trial_end()
    test_deferred_obs_normalization()
    print("all trial-aware storage tests passed")
