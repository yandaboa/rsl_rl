# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The deterministic eval pool of :class:`EpisodeContextPPO` (``eval_env_fraction``).

Self-contained: pure torch, no Isaac Sim. Run with::

    source /home/yandabao/miniforge3/etc/profile.d/conda.sh && conda activate patlab
    python -m pytest tests/test_epctx_eval_pool.py -q

The four statements: the eval envs execute the distribution MEAN, the update never sees their rows, they still
act every step (so their KV ring stays in sync and their stored rows re-infer exactly), and ``0.0`` is bit-for-bit
the old behavior.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import EpisodeContextPPO
from rsl_rl.modules import ActorCriticEpisodeContext
from rsl_rl.modules.actor_critic_episode_context import EpisodeContextPrefix

DEVICE = "cpu"
OBS_DIM = 5
CRITIC_OBS_DIM = 7
ACTION_DIM = 4
NUM_ENVS = 8
EVAL_FRACTION = 0.25  # -> 2 eval envs, 6 training envs
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


def _make_policy(seed: int = 0, noise_std_type: str = "scalar") -> ActorCriticEpisodeContext:
    torch.manual_seed(seed)
    policy = ActorCriticEpisodeContext(
        obs=_sample_obs(NUM_ENVS),
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
        noise_std_type=noise_std_type,
        init_noise_std=0.5,
    )
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if name in ("std", "log_std"):
                continue
            parameter.add_(0.1 * torch.randn_like(parameter))
    return policy


def _make_ppo(policy: ActorCriticEpisodeContext, **kwargs) -> EpisodeContextPPO:
    defaults = dict(
        num_learning_epochs=1,
        num_mini_batches=2,
        learning_rate=0.0,
        schedule="fixed",
        desired_kl=None,
        gamma=GAMMA,
        lam=LAM,
        device=DEVICE,
    )
    defaults.update(kwargs)
    ppo = EpisodeContextPPO(policy, **defaults)
    ppo.init_storage("rl", NUM_ENVS, STEPS_PER_ENV, _sample_obs(NUM_ENVS), [ACTION_DIM])
    return ppo


def _episode_schedule(num_steps: int) -> torch.Tensor:
    """Desynchronized episode boundaries: env e ends its first episode e steps early."""
    dones = torch.zeros(num_steps, NUM_ENVS, dtype=torch.bool, device=DEVICE)
    for env in range(NUM_ENVS):
        step = T_EPISODE - 1 - (env % T_EPISODE)
        while step < num_steps:
            dones[step, env] = True
            step += T_EPISODE
    return dones


def _collect(ppo: EpisodeContextPPO, dones: torch.Tensor, offset: int, generator: torch.Generator) -> None:
    obs = _sample_obs(NUM_ENVS, generator)
    with torch.no_grad():
        for step in range(STEPS_PER_ENV):
            ppo.act(obs)
            next_obs = _sample_obs(NUM_ENVS, generator)
            rewards = torch.randn(NUM_ENVS, generator=generator, device=DEVICE)
            ppo.process_env_step(next_obs, rewards, dones[offset + step], {})
            obs = next_obs
        ppo.compute_returns(obs)


# --------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------


def test_eval_envs_act_on_the_mean() -> None:
    """Eval envs execute exactly ``action_mean``; training envs sample (so they differ from it)."""
    dones = _episode_schedule(STEPS_PER_ENV)
    policy = _make_policy(noise_std_type="gsde")
    ppo = _make_ppo(policy, eval_env_fraction=EVAL_FRACTION)
    assert ppo.eval_env_ids.tolist() == [6, 7]

    generator = torch.Generator(device=DEVICE).manual_seed(3)
    obs = _sample_obs(NUM_ENVS, generator)
    train_deviation = 0.0
    with torch.no_grad():
        for step in range(STEPS_PER_ENV):
            actions = ppo.act(obs)
            mean = policy.action_mean
            assert torch.equal(actions[6:], mean[6:]), f"eval envs did not act on the mean at step {step}"
            train_deviation += (actions[:6] - mean[:6]).abs().sum().item()
            next_obs = _sample_obs(NUM_ENVS, generator)
            ppo.process_env_step(next_obs, torch.randn(NUM_ENVS, generator=generator), dones[step], {})
            obs = next_obs
    assert train_deviation > 0.0, "the training envs acted on the mean too; nothing is exploring"
    assert torch.equal(ppo.storage.actions[:, 6:], ppo.storage.mu[:, 6:]), "stored eval rows are not the mean"
    # The stored log-prob of an eval row is the log-prob of the action it actually executed.
    assert torch.isfinite(ppo.storage.actions_log_prob[:, 6:]).all()
    print("[ok] the eval pool executes the distribution mean, the training pool samples")


def test_update_never_sees_eval_rows() -> None:
    """Every training env is handed out exactly once per epoch; no eval env ever is."""
    dones = _episode_schedule(STEPS_PER_ENV)
    policy = _make_policy(seed=1)
    ppo = _make_ppo(policy, eval_env_fraction=EVAL_FRACTION, num_mini_batches=2, num_learning_epochs=3)
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    _collect(ppo, dones, 0, generator)

    stored_actions = ppo.storage.actions.clone()
    seen = torch.zeros(NUM_ENVS, device=DEVICE)
    original = ppo.storage.recurrent_mini_batch_generator

    def spy(num_mini_batches: int, num_epochs: int = 8):
        for batch in original(num_mini_batches, num_epochs):
            actions = batch[1]
            for column in range(actions.shape[1]):
                matches = (stored_actions == actions[:, column : column + 1]).all(dim=-1).all(dim=0)
                found = matches.nonzero(as_tuple=False)
                assert found.numel() == 1, "could not uniquely locate a minibatch column in the storage"
                seen[int(found.item())] += 1.0
            yield batch

    ppo.storage.recurrent_mini_batch_generator = spy  # type: ignore[method-assign]
    ppo.update()
    assert seen[6:].sum().item() == 0.0, f"eval env rows reached the update: {seen}"
    assert torch.equal(seen[:6], torch.full((6,), 3.0, device=DEVICE)), f"training coverage is not 3 epochs: {seen}"
    print("[ok] the update covers every training env once per epoch and no eval env ever")


def test_eval_envs_keep_the_kv_ring_in_sync() -> None:
    """The eval envs act every step, so their stored rows re-infer EXACTLY from the ring buffer."""
    dones = _episode_schedule(3 * STEPS_PER_ENV)
    policy = _make_policy(seed=4)
    ppo = _make_ppo(policy, eval_env_fraction=EVAL_FRACTION)
    generator = torch.Generator(device=DEVICE).manual_seed(11)

    for rollout in range(3):
        _collect(ppo, dones, rollout * STEPS_PER_ENV, generator)
        prefix_obs, prefix_positions, window_positions = ppo.storage.context_slice()
        prefix = EpisodeContextPrefix(
            obs=prefix_obs, positions=prefix_positions, window_positions=window_positions
        )
        with torch.no_grad():
            policy.act(ppo.storage.observations, hidden_state=prefix)
            log_prob = policy.get_actions_log_prob(ppo.storage.actions)
        # Both pools: the eval rows are the interesting ones (they are never re-inferred by the update, so
        # nothing else would catch a desynchronized ring).
        mean_error = (policy.action_mean - ppo.storage.mu).abs().max().item()
        log_prob_error = (log_prob - ppo.storage.actions_log_prob.squeeze(-1)).abs().max().item()
        assert mean_error < 1e-5, f"re-inferred means drifted by {mean_error:.3e} (rollout {rollout})"
        assert log_prob_error < 1e-5, f"re-inferred log-probs drifted by {log_prob_error:.3e} (rollout {rollout})"
        # The eval envs advanced the same episode-step schedule as everybody else.
        assert torch.equal(ppo.storage.episode_step[6:], ppo.storage.episode_step[:2])
        ppo.update()
    print("[ok] the eval pool advances the KV ring in lockstep; its rows re-infer exactly")


def test_zero_fraction_is_the_old_behavior() -> None:
    """``eval_env_fraction=0`` must be bit-for-bit identical to not passing it at all."""
    dones = _episode_schedule(2 * STEPS_PER_ENV)

    def run(**kwargs) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
        policy = _make_policy(seed=6)
        ppo = _make_ppo(policy, learning_rate=1e-3, num_learning_epochs=2, **kwargs)
        generator = torch.Generator(device=DEVICE).manual_seed(21)
        torch.manual_seed(99)  # the acting path samples off the global RNG
        for rollout in range(2):
            _collect(ppo, dones, rollout * STEPS_PER_ENV, generator)
            snapshot = {
                "actions": ppo.storage.actions.clone(),
                "log_prob": ppo.storage.actions_log_prob.clone(),
                "advantages": ppo.storage.advantages.clone(),
                "returns": ppo.storage.returns.clone(),
            }
            ppo.update()
        return snapshot, [parameter.detach().clone() for parameter in policy.parameters()]

    baseline, baseline_parameters = run()
    zeroed, zeroed_parameters = run(eval_env_fraction=0.0)
    for key, value in baseline.items():
        assert torch.equal(value, zeroed[key]), f"{key} differs with eval_env_fraction=0"
    for before, after in zip(baseline_parameters, zeroed_parameters):
        assert torch.equal(before, after), "the update moved to different parameters with eval_env_fraction=0"
    print("[ok] eval_env_fraction=0 reproduces the stock rollout and the stock update exactly")


if __name__ == "__main__":
    test_eval_envs_act_on_the_mean()
    test_update_never_sees_eval_rows()
    test_eval_envs_keep_the_kv_ring_in_sync()
    test_zero_fraction_is_the_old_behavior()
    print("all eval-pool tests passed")
