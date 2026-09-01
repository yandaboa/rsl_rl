# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gradient accumulation in :class:`EpisodeContextPPO` (``grad_accumulation_steps``).

Self-contained: pure torch, no Isaac Sim. Run with::

    source /home/yandabao/miniforge3/etc/profile.d/conda.sh && conda activate patlab
    python -m pytest tests/test_epctx_grad_accumulation.py -q

The four statements: ``G = 1`` is the frozen ``ppo.py`` update itself, ``G > 1`` lands on the same parameters
as ``G = 1`` (equal micro-batches, so their means average to the minibatch mean), the optimizer still steps
``num_mini_batches`` times per epoch whatever ``G`` is, and the eval pool is unaffected.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import EpisodeContextPPO
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules import ActorCriticEpisodeContext

DEVICE = "cpu"
OBS_DIM = 5
CRITIC_OBS_DIM = 7
ACTION_DIM = 4
NUM_ENVS = 8
T_EPISODE = 6
STEPS_PER_ENV = 8
GAMMA = 0.99
LAM = 0.95


def _sample_obs(num_envs: int, generator: torch.Generator | None = None) -> TensorDict:
    if generator is None:
        policy_obs = torch.zeros(num_envs, OBS_DIM, device=DEVICE)
        critic_obs = torch.zeros(num_envs, CRITIC_OBS_DIM, device=DEVICE)
    else:
        policy_obs = torch.randn(num_envs, OBS_DIM, generator=generator, device=DEVICE)
        critic_obs = torch.randn(num_envs, CRITIC_OBS_DIM, generator=generator, device=DEVICE)
    return TensorDict({"policy": policy_obs, "critic": critic_obs}, batch_size=[num_envs], device=DEVICE)


def _make_policy(seed: int = 0, num_envs: int = NUM_ENVS, memory_tokens: int = 0) -> ActorCriticEpisodeContext:
    torch.manual_seed(seed)
    policy = ActorCriticEpisodeContext(
        obs=_sample_obs(num_envs),
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=ACTION_DIM,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        context_length=T_EPISODE,
        d_model=32,
        num_layers=2,
        num_heads=4,
        max_episode_length=T_EPISODE,
        ff_mult=2,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        init_noise_std=0.5,
        memory_tokens=memory_tokens,
        episodes_per_trial=2,
    )
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if name in ("std", "log_std"):
                continue
            parameter.add_(0.1 * torch.randn_like(parameter))
    return policy


def _make_ppo(policy: ActorCriticEpisodeContext, num_envs: int = NUM_ENVS, **kwargs) -> EpisodeContextPPO:
    defaults = dict(
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1e-3,
        schedule="fixed",
        desired_kl=None,
        gamma=GAMMA,
        lam=LAM,
        device=DEVICE,
    )
    defaults.update(kwargs)
    ppo = EpisodeContextPPO(policy, **defaults)
    ppo.init_storage("rl", num_envs, STEPS_PER_ENV, _sample_obs(num_envs), [ACTION_DIM])
    return ppo


def _episode_schedule(num_steps: int, num_envs: int = NUM_ENVS) -> torch.Tensor:
    """Desynchronized episode boundaries: env e ends its first episode e steps early."""
    dones = torch.zeros(num_steps, num_envs, dtype=torch.bool, device=DEVICE)
    for env in range(num_envs):
        step = T_EPISODE - 1 - (env % T_EPISODE)
        while step < num_steps:
            dones[step, env] = True
            step += T_EPISODE
    return dones


def _collect(ppo: EpisodeContextPPO, dones: torch.Tensor, generator: torch.Generator, num_envs: int = NUM_ENVS):
    obs = _sample_obs(num_envs, generator)
    with torch.no_grad():
        for step in range(STEPS_PER_ENV):
            ppo.act(obs)
            next_obs = _sample_obs(num_envs, generator)
            rewards = torch.randn(num_envs, generator=generator, device=DEVICE)
            ppo.process_env_step(next_obs, rewards, dones[step], {})
            obs = next_obs
        ppo.compute_returns(obs)


def _run(
    num_envs: int = NUM_ENVS, seed: int = 5, memory_tokens: int = 0, **kwargs
) -> tuple[EpisodeContextPPO, list[torch.Tensor]]:
    """One rollout + one update from a fixed initialization, with a fixed data stream."""
    policy = _make_policy(seed=seed, num_envs=num_envs, memory_tokens=memory_tokens)
    ppo = _make_ppo(policy, num_envs=num_envs, **kwargs)
    generator = torch.Generator(device=DEVICE).manual_seed(17)
    torch.manual_seed(123)  # the acting path samples off the global RNG
    _collect(ppo, _episode_schedule(STEPS_PER_ENV, num_envs), generator, num_envs)
    torch.manual_seed(321)
    ppo.update()
    return ppo, [parameter.detach().clone() for parameter in policy.parameters()]


def _gradients(grad_accumulation_steps: int, memory_tokens: int = 0, **kwargs) -> list[list[torch.Tensor]]:
    """The gradients each optimizer step is taken with, over one rollout and one update."""
    policy = _make_policy(seed=5, memory_tokens=memory_tokens)
    ppo = _make_ppo(
        policy,
        num_mini_batches=2,
        num_learning_epochs=1,
        learning_rate=0.0,  # the parameters stay put, so every minibatch is compared at the same point
        grad_accumulation_steps=grad_accumulation_steps,
        **kwargs,
    )
    generator = torch.Generator(device=DEVICE).manual_seed(17)
    torch.manual_seed(123)
    _collect(ppo, _episode_schedule(STEPS_PER_ENV), generator)

    captured: list[list[torch.Tensor]] = []
    original_step = ppo.optimizer.step

    def capturing_step(*args, **kwargs):
        captured.append([p.grad.detach().clone() for p in policy.parameters() if p.grad is not None])
        return original_step(*args, **kwargs)

    ppo.optimizer.step = capturing_step  # type: ignore[method-assign]
    torch.manual_seed(321)
    ppo.update()
    return captured


# --------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------


def test_default_is_the_frozen_ppo_update() -> None:
    """``grad_accumulation_steps=1`` runs ``PPO.update`` itself and lands on the same parameters."""
    _, baseline = _run()
    _, explicit = _run(grad_accumulation_steps=1)
    for before, after in zip(baseline, explicit):
        assert torch.equal(before, after), "grad_accumulation_steps=1 moved to different parameters"

    seen: list[int] = []
    original = PPO.update

    def spy(self):
        seen.append(self.grad_accumulation_steps)
        return original(self)

    PPO.update = spy  # type: ignore[method-assign]
    try:
        _run(grad_accumulation_steps=1)
        assert seen == [1], f"G=1 did not delegate to the frozen PPO.update (saw {seen})"
        seen.clear()
        _run(grad_accumulation_steps=2)
        assert seen == [], "G=2 fell through to the frozen PPO.update"
    finally:
        PPO.update = original  # type: ignore[method-assign]
    print("[ok] G=1 delegates to ppo.py's update; G>1 takes the accumulating path")


def test_accumulation_matches_the_single_pass_update() -> None:
    """Two micro-batches per minibatch reach the same parameters as one full minibatch."""
    _, single = _run(num_mini_batches=2, grad_accumulation_steps=1)
    _, accumulated = _run(num_mini_batches=2, grad_accumulation_steps=2)
    moved = max((a - b).abs().max().item() for a, b in zip(single, accumulated))
    scale = max(parameter.abs().max().item() for parameter in single)
    assert moved < 1e-5, f"G=2 diverged from G=1 by {moved:.3e} (parameter scale {scale:.3e})"
    # Sanity: the update did move the parameters at all, so the comparison is not vacuous.
    _, before_update = _run(num_mini_batches=2, learning_rate=0.0)
    assert max((a - b).abs().max().item() for a, b in zip(single, before_update)) > 1e-6
    # The load-bearing quantity is the gradient each step is taken with (Adam hides its scale).
    one_gradients, two_gradients = _gradients(1), _gradients(2)
    assert len(one_gradients) == len(two_gradients) == 2
    gradient_difference = max(
        (a - b).abs().max().item() for one, two in zip(one_gradients, two_gradients) for a, b in zip(one, two)
    )
    gradient_scale = max(a.abs().max().item() for one in one_gradients for a in one)
    assert gradient_difference / gradient_scale < 1e-5, f"gradients differ by {gradient_difference:.3e}"
    print(f"[ok] G=2 matches G=1 to {moved:.3e} (gradients to {gradient_difference / gradient_scale:.2e})")


@pytest.mark.parametrize("grad_accumulation_steps", [1, 2, 4])
def test_one_optimizer_step_per_mini_batch(grad_accumulation_steps: int) -> None:
    """The optimizer steps ``num_mini_batches`` times per epoch, whatever ``G`` is."""
    num_mini_batches = 2
    num_learning_epochs = 3
    num_envs = 16  # G=4 needs 2 environments per micro-batch
    policy = _make_policy(seed=8, num_envs=num_envs)
    ppo = _make_ppo(
        policy,
        num_envs=num_envs,
        num_mini_batches=num_mini_batches,
        num_learning_epochs=num_learning_epochs,
        grad_accumulation_steps=grad_accumulation_steps,
    )
    generator = torch.Generator(device=DEVICE).manual_seed(19)
    _collect(ppo, _episode_schedule(STEPS_PER_ENV, num_envs), generator, num_envs)

    steps = 0
    micro_batches = 0
    original_step = ppo.optimizer.step
    original_generator = ppo.storage.recurrent_mini_batch_generator

    def counting_step(*args, **kwargs):
        nonlocal steps
        steps += 1
        return original_step(*args, **kwargs)

    def counting_generator(*args, **kwargs):
        nonlocal micro_batches
        for batch in original_generator(*args, **kwargs):
            micro_batches += 1
            yield batch

    ppo.optimizer.step = counting_step  # type: ignore[method-assign]
    ppo.storage.recurrent_mini_batch_generator = counting_generator  # type: ignore[method-assign]
    ppo.update()

    assert steps == num_learning_epochs * num_mini_batches, f"{steps} optimizer steps at G={grad_accumulation_steps}"
    expected_micro = num_learning_epochs * num_mini_batches * grad_accumulation_steps
    assert micro_batches == expected_micro, f"{micro_batches} micro-batches, expected {expected_micro}"
    print(f"[ok] G={grad_accumulation_steps}: {steps} optimizer steps, {micro_batches} micro-batches")


def test_accumulation_with_an_eval_pool() -> None:
    """With an eval pool the split covers the TRAINING environments only, and G=2 still matches G=1."""
    num_envs = 12  # 4 eval, 8 training -> 2 minibatches x 2 micro-batches of 2 envs
    kwargs = dict(num_envs=num_envs, eval_env_fraction=1.0 / 3.0, num_mini_batches=2)
    single_ppo, single = _run(**kwargs, grad_accumulation_steps=1)
    accumulated_ppo, accumulated = _run(**kwargs, grad_accumulation_steps=2)
    assert single_ppo.storage.num_train_envs == 8 and accumulated_ppo.eval_env_ids.numel() == 4
    moved = max((a - b).abs().max().item() for a, b in zip(single, accumulated))
    assert moved < 1e-5, f"G=2 diverged from G=1 by {moved:.3e} with an eval pool"

    # The micro-batches never reach into the eval block.
    policy = _make_policy(seed=5, num_envs=num_envs)
    ppo = _make_ppo(policy, num_envs=num_envs, eval_env_fraction=1.0 / 3.0, grad_accumulation_steps=2)
    generator = torch.Generator(device=DEVICE).manual_seed(17)
    _collect(ppo, _episode_schedule(STEPS_PER_ENV, num_envs), generator, num_envs)
    seen = torch.zeros(num_envs, device=DEVICE)
    stored_actions = ppo.storage.actions.clone()
    for batch in ppo.storage.recurrent_mini_batch_generator(2, 1, 2):
        actions = batch[1]
        assert actions.shape[1] == 2, f"micro-batch of {actions.shape[1]} environments, expected 2"
        for column in range(actions.shape[1]):
            matches = (stored_actions == actions[:, column : column + 1]).all(dim=-1).all(dim=0)
            found = matches.nonzero(as_tuple=False)
            assert found.numel() == 1, "could not uniquely locate a micro-batch column in the storage"
            seen[int(found.item())] += 1.0
    assert torch.equal(seen[:8], torch.ones(8, device=DEVICE)), f"training coverage is not exactly once: {seen}"
    assert seen[8:].sum().item() == 0.0, f"eval env rows reached a micro-batch: {seen}"
    print("[ok] the eval pool is excluded and every training env appears in exactly one micro-batch")


def test_adaptive_kl_is_averaged_over_the_micro_batches() -> None:
    """The adaptive schedule sees the MINIBATCH's KL: equal micro-batches average to it exactly."""
    kwargs = dict(num_mini_batches=2, schedule="adaptive", desired_kl=0.01)
    single_ppo, single = _run(**kwargs, grad_accumulation_steps=1)
    accumulated_ppo, accumulated = _run(**kwargs, grad_accumulation_steps=2)
    assert accumulated_ppo.learning_rate == pytest.approx(single_ppo.learning_rate, rel=1e-6)
    moved = max((a - b).abs().max().item() for a, b in zip(single, accumulated))
    assert moved < 1e-5, f"G=2 diverged from G=1 by {moved:.3e} under the adaptive schedule"
    print(f"[ok] adaptive schedule: same learning rate ({single_ppo.learning_rate:.3e}), same parameters")


def test_advantages_are_normalized_over_the_whole_mini_batch() -> None:
    """``normalize_advantage_per_mini_batch`` uses the minibatch's statistics, not the micro-batch's."""
    kwargs = dict(num_mini_batches=2, normalize_advantage_per_mini_batch=True)
    _, single = _run(**kwargs, grad_accumulation_steps=1)
    _, accumulated = _run(**kwargs, grad_accumulation_steps=2)
    moved = max((a - b).abs().max().item() for a, b in zip(single, accumulated))
    assert moved < 1e-5, f"G=2 diverged from G=1 by {moved:.3e} with per-minibatch advantage normalization"
    print(f"[ok] per-minibatch advantage normalization is unchanged by accumulation ({moved:.3e})")


def test_accumulation_with_a_memory_policy() -> None:
    """A memory policy re-slices its per-segment sources per micro-batch; the gradient is unchanged.

    Compared on the GRADIENT, not on the parameters: Adam normalizes per parameter, so fp noise on a
    near-zero gradient entry moves it by up to the learning rate and says nothing about correctness.
    """
    single = _gradients(memory_tokens=4, grad_accumulation_steps=1)
    accumulated = _gradients(memory_tokens=4, grad_accumulation_steps=2)
    difference = max(
        (a - b).abs().max().item() for one, two in zip(single, accumulated) for a, b in zip(one, two)
    )
    scale = max(a.abs().max().item() for one in single for a in one)
    assert difference / scale < 1e-5, f"G=2 gradients differ from G=1 by {difference:.3e} (scale {scale:.3e})"
    print(f"[ok] memory policy: the accumulated gradient matches to {difference / scale:.2e} relative")


def test_uneven_split_is_refused() -> None:
    """An environment count that does not split evenly into ``num_mini_batches x G`` is an error, not a silent
    reweighting of the gradient."""
    policy = _make_policy(seed=3)
    ppo = _make_ppo(policy, num_mini_batches=3, grad_accumulation_steps=2)
    generator = torch.Generator(device=DEVICE).manual_seed(23)
    _collect(ppo, _episode_schedule(STEPS_PER_ENV), generator)
    with pytest.raises(AssertionError, match="do not split evenly"):
        ppo.update()
    print("[ok] an uneven micro-batch split is refused")


if __name__ == "__main__":
    test_default_is_the_frozen_ppo_update()
    test_accumulation_matches_the_single_pass_update()
    for steps in (1, 2, 4):
        test_one_optimizer_step_per_mini_batch(steps)
    test_accumulation_with_an_eval_pool()
    test_adaptive_kl_is_averaged_over_the_micro_batches()
    test_advantages_are_normalized_over_the_whole_mini_batch()
    test_accumulation_with_a_memory_policy()
    test_uneven_split_is_refused()
    print("all gradient-accumulation tests passed")
