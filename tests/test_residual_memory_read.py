# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the gated residual memory read (``memory_interface="residual_read"``) and the lag-aware KL.

Self-contained: pure torch, no Isaac Sim / SimulationApp needed. Run with::

    source /home/yandabao/miniforge3/etc/profile.d/conda.sh && conda activate patlab
    PYTHONPATH=/home/yandabao/rsl_rl-wt/worktree-local-exploration:$PYTHONPATH \
        python tests/test_residual_memory_read.py

The load-bearing statements:

* ``test_identity_at_init`` -- with both gates at zero the trunk is a memory-FREE causal transformer, bit for
  bit, whatever ``Z`` contains. This is what lets a BC'd trunk survive the first PPO update,
* ``test_gate_open_*`` -- once a gate is off zero the memory does reach the output and the writer does receive
  a gradient, so the identity above is a property of the initialization and not a dead code path,
* ``test_step_matches_sequence_*`` -- the acting path (cached read K/V) and the batched training path agree,
  including across a memory write,
* ``test_lag_aware_kl_*`` -- the adaptive LR and the early stop are driven by fresh rows only.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCriticTrialMemory

DEVICE = "cpu"
OBS_DIM = 5
ACTION_DIM = 4
NUM_ENVS = 3
T_EPISODE = 6
NUM_MEMORY = 3
D_MODEL = 32


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _make_policy(
    memory_interface: str = "residual_read",
    critic_trunk: str = "shared",
    attention_window: int | None = None,
    memory_read_layer: int = -1,
    perturb: bool = True,
    num_envs: int = NUM_ENVS,
    seed: int = 0,
) -> ActorCriticTrialMemory:
    """A small float64 policy. Double precision makes "bit-for-bit" and "tight tolerance" mean something."""
    torch.manual_seed(seed)
    sample_obs = TensorDict({"policy": torch.zeros(num_envs, OBS_DIM)}, batch_size=[num_envs], device=DEVICE)
    policy = ActorCriticTrialMemory(
        obs=sample_obs,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=ACTION_DIM,
        d_model=D_MODEL,
        num_layers=2,
        num_heads=4,
        num_memory_tokens=NUM_MEMORY,
        attention_window=attention_window,
        max_episode_length=T_EPISODE,
        ff_mult=2,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        init_noise_std=0.5,
        critic_trunk=critic_trunk,
        memory_interface=memory_interface,
        memory_read_layer=memory_read_layer,
    )
    if perturb:
        # Move the network off its initialization (gates included) so the tests are not measuring an identity.
        with torch.no_grad():
            for name, parameter in policy.named_parameters():
                if name in ("std", "log_std"):
                    continue
                parameter.add_(0.1 * torch.randn_like(parameter))
    return policy.double().eval()


def _make_episode(num_steps: int, num_envs: int = NUM_ENVS, seed: int = 1) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64, device=DEVICE)  # noqa: E731
    return {
        "obs": randn(num_steps, num_envs, OBS_DIM),
        "prev_actions": randn(num_steps, num_envs, ACTION_DIM),
        "prev_rewards": randn(num_steps, num_envs, 1),
        "prev_dones": (randn(num_steps, num_envs, 1) > 1.5).double(),
    }


def _roll_batched(
    policy: ActorCriticTrialMemory, episode: dict[str, torch.Tensor], memory: torch.Tensor, critic: bool = False
) -> torch.Tensor:
    forward = policy.forward_sequence_critic if critic else policy.forward_sequence
    return forward(
        episode["obs"], episode["prev_actions"], episode["prev_rewards"], episode["prev_dones"], memory=memory
    )


def _roll_incrementally(
    policy: ActorCriticTrialMemory, episode: dict[str, torch.Tensor], memory: torch.Tensor | None = None
) -> torch.Tensor:
    """Step the acting path one token at a time, optionally starting from an explicit ``Z``."""
    num_steps, num_envs = episode["obs"].shape[0], episode["obs"].shape[1]
    policy.initialize_state(num_envs, DEVICE, dtype=torch.float64)
    if memory is not None:
        policy._memory = memory.clone()
        policy._memory_kv = None
        policy._read_kv = None
    hidden = []
    for step in range(num_steps):
        hidden.append(
            policy.forward_step(
                episode["obs"][step],
                prev_actions=episode["prev_actions"][step],
                prev_rewards=episode["prev_rewards"][step],
                prev_dones=episode["prev_dones"][step],
                commit=True,
            )
        )
    return torch.stack(hidden)


def _random_memory(policy: ActorCriticTrialMemory, num_envs: int = NUM_ENVS, seed: int = 99) -> torch.Tensor:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    return torch.randn(
        num_envs, policy.num_memory_tokens, policy.d_model, generator=generator, dtype=torch.float64, device=DEVICE
    )


# --------------------------------------------------------------------------------------------------
# A1 -- the residual read layer
# --------------------------------------------------------------------------------------------------


def test_defaults_and_placement() -> None:
    """The default interface is the residual read, placed at ``num_layers // 2``; the legacy one is opt-in."""
    residual = _make_policy(perturb=False)
    assert residual.memory_interface == "residual_read" and residual.residual_memory_read
    assert residual.memory_read_layer == residual.num_layers // 2
    assert float(residual.memory_read.gate_attn) == 0.0 and float(residual.memory_read.gate_ff) == 0.0
    assert residual.memory_gate_values() == {"memory_gate_attn": 0.0, "memory_gate_ff": 0.0}

    explicit = _make_policy(memory_read_layer=0, perturb=False)
    assert explicit.memory_read_layer == 0

    legacy = _make_policy(memory_interface="kv_prepend", perturb=False)
    assert not legacy.residual_memory_read and not hasattr(legacy, "memory_read")
    assert legacy.memory_gate_values() == {}

    separate = _make_policy(critic_trunk="separate", perturb=False)
    gates = separate.memory_gate_values()
    assert set(gates) == {"memory_gate_attn", "memory_gate_ff", "critic_memory_gate_attn", "critic_memory_gate_ff"}
    assert separate.critic_memory_read is not separate.memory_read

    try:
        _make_policy(memory_interface="nonsense", perturb=False)
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown memory_interface must raise")
    print("[ok] residual_read is the default, placed after block num_layers // 2; kv_prepend still available")


def test_identity_at_init() -> None:
    """With the gates at zero the output must not depend on ``Z`` at all -- on either path, bit for bit."""
    episode = _make_episode(T_EPISODE + 1, seed=21)

    for critic_trunk in ("shared", "separate"):
        policy = _make_policy(critic_trunk=critic_trunk, perturb=False)
        z_init = policy.initial_memory(NUM_ENVS).double().detach()
        random_memory = _random_memory(policy) * 10.0

        with torch.no_grad():
            reference = _roll_batched(policy, episode, z_init)
            perturbed = _roll_batched(policy, episode, random_memory)
        assert torch.equal(reference, perturbed), f"forward_sequence depends on Z at init ({critic_trunk})"

        with torch.no_grad():
            reference_step = _roll_incrementally(policy, episode, memory=z_init)
            perturbed_step = _roll_incrementally(policy, episode, memory=random_memory)
        assert torch.equal(reference_step, perturbed_step), f"forward_step depends on Z at init ({critic_trunk})"
        # ... and the acting path is the memory-free trunk the batched path computes.
        assert (reference_step - reference).abs().max().item() < 1e-12

        if critic_trunk == "separate":
            with torch.no_grad():
                reference_value = _roll_batched(policy, episode, z_init, critic=True)
                perturbed_value = _roll_batched(policy, episode, random_memory, critic=True)
            assert torch.equal(reference_value, perturbed_value), "the critic trunk depends on Z at init"

    # Non-vacuous: the trunk itself is not a constant map, it just ignores the memory.
    varied = {key: value.clone() for key, value in episode.items()}
    varied["obs"] += 1.0
    with torch.no_grad():
        moved = (_roll_batched(policy, varied, z_init) - reference).abs().max().item()
    assert moved > 1e-6, f"the trunk output barely moved with the input ({moved:.3e}); the test would be vacuous"
    print("[ok] exact identity at init: h is bit-identical for Z_init and for a random Z (step and sequence)")


def test_gate_open_lets_the_memory_through() -> None:
    """Opening ``gate_attn`` must make the output depend on ``Z``, on both paths."""
    policy = _make_policy(perturb=False)
    episode = _make_episode(T_EPISODE + 1, seed=22)
    z_init = policy.initial_memory(NUM_ENVS).double().detach()
    other = _random_memory(policy, seed=123)

    with torch.no_grad():
        policy.memory_read.gate_attn.data.fill_(1.0)
        moved = (_roll_batched(policy, episode, z_init) - _roll_batched(policy, episode, other)).abs().max().item()
        moved_step = (
            (_roll_incrementally(policy, episode, memory=z_init) - _roll_incrementally(policy, episode, memory=other))
            .abs()
            .max()
            .item()
        )
    assert moved > 1e-6, f"an open gate still ignores the memory ({moved:.3e})"
    assert moved_step > 1e-6, f"an open gate still ignores the memory on the acting path ({moved_step:.3e})"
    print(f"[ok] gate open: |dh| = {moved:.3e} (sequence), {moved_step:.3e} (step)")


def test_gate_gates_the_gradient_to_the_writer() -> None:
    """A pair-style backward reaches the writer only once a gate is open; at init it is exactly zero."""
    episode = _make_episode(T_EPISODE + 1, seed=23)

    def pair_backward(policy: ActorCriticTrialMemory) -> dict[str, float]:
        policy.zero_grad(set_to_none=True)
        memory = policy.initial_memory(NUM_ENVS).double()
        source_hidden = _roll_batched(policy, episode, memory)
        written, _ = policy.write_memory(memory, source_hidden.transpose(0, 1))
        target_hidden = _roll_batched(policy, episode, written)
        target_hidden.square().mean().backward()
        grad = lambda parameter: 0.0 if parameter.grad is None else float(parameter.grad.abs().sum())  # noqa: E731
        return {
            "writer": grad(policy.writer.attn.q_proj.weight) + grad(policy.writer.ff[0].weight),
            "gate_attn": grad(policy.memory_read.gate_attn),
            "gate_ff": grad(policy.memory_read.gate_ff),
            "trunk": grad(policy.blocks[0].attn.q_proj.weight),
        }

    closed = pair_backward(_make_policy(perturb=False))
    assert closed["writer"] == 0.0, f"the writer is trained through a closed gate: {closed['writer']}"
    assert closed["gate_attn"] > 0.0, "the gate itself must receive a gradient, otherwise it can never open"
    assert closed["trunk"] > 0.0, "the trunk got no gradient; the backward is broken, not gated"

    policy = _make_policy(perturb=False)
    with torch.no_grad():
        policy.memory_read.gate_attn.data.fill_(1.0)
        policy.memory_read.gate_ff.data.fill_(0.5)
    opened = pair_backward(policy)
    assert opened["writer"] > 0.0, "an open gate must let a gradient reach the writer"
    assert opened["gate_ff"] > 0.0
    print(
        f"[ok] gated gradients: writer |g| = {closed['writer']:.1e} (closed) -> {opened['writer']:.3e} (open),"
        f" gate |g| = {closed['gate_attn']:.3e} at init"
    )


def test_step_matches_sequence_in_residual_mode() -> None:
    """The KV-cached acting path must reproduce the batched forward, full and windowed attention."""
    episode = _make_episode(T_EPISODE + 1, seed=24)
    for window, label in ((None, "full (W >= T)"), (3, "windowed (W = 3)")):
        policy = _make_policy(attention_window=window, seed=3)
        with torch.no_grad():
            policy.memory_read.gate_attn.data.fill_(0.8)  # a closed gate would make the test vacuous
            policy.memory_read.gate_ff.data.fill_(0.4)
            memory = _random_memory(policy, seed=31)
            incremental = _roll_incrementally(policy, episode, memory=memory)
            batched = _roll_batched(policy, episode, memory)
        error = (incremental - batched).abs().max().item()
        assert incremental.shape == batched.shape == (T_EPISODE + 1, NUM_ENVS, policy.d_model)
        assert error < 1e-10, f"incremental vs batched mismatch ({label}): {error:.3e}"
        print(f"[ok] residual read: incremental == batched, {label}: max |dh| = {error:.3e}")


def test_cached_read_kv_follows_the_memory() -> None:
    """The cached read K/V is invalidated by every write, so an episode after a write uses the new ``Z``."""
    policy = _make_policy(seed=4)
    with torch.no_grad():
        policy.memory_read.gate_attn.data.fill_(0.8)
    first = _make_episode(T_EPISODE + 1, seed=25)
    second = _make_episode(T_EPISODE, seed=26)

    with torch.no_grad():
        _roll_incrementally(policy, first)
        assert policy._read_kv is not None, "the read K/V was never cached"
        policy.update_memory()
        assert policy._read_kv is None, "update_memory() left a stale read K/V cache"
        written = policy.memory.clone()

        policy.reset_episode()
        acting = []
        for step in range(second["obs"].shape[0]):
            acting.append(
                policy.forward_step(
                    second["obs"][step],
                    prev_actions=second["prev_actions"][step],
                    prev_rewards=second["prev_rewards"][step],
                    prev_dones=second["prev_dones"][step],
                )
            )
        acting = torch.stack(acting)
        batched = _roll_batched(policy, second, written)

        # ... and a trial reset drops the cache again, returning the acting path to Z_init.
        policy.reset_trial()
        assert policy._read_kv is None, "reset_trial() left a stale read K/V cache"

    error = (acting - batched).abs().max().item()
    assert error < 1e-10, f"the episode after a write did not use the written Z: {error:.3e}"
    assert not torch.allclose(written, policy.initial_memory(NUM_ENVS).double()), "the writer did not move Z"
    print(f"[ok] read K/V cache follows the writer: post-write step == batched, max |dh| = {error:.3e}")


def test_legacy_kv_prepend_is_unchanged() -> None:
    """The legacy interface still prepends ``Z`` per layer, and its two paths still agree."""
    episode = _make_episode(T_EPISODE + 1, seed=27)
    policy = _make_policy(memory_interface="kv_prepend", seed=5)
    memory = _random_memory(policy, seed=41)

    with torch.no_grad():
        incremental = _roll_incrementally(policy, episode, memory=memory)
        batched = _roll_batched(policy, episode, memory)
        other = _roll_batched(policy, episode, _random_memory(policy, seed=42))
    error = (incremental - batched).abs().max().item()
    assert error < 1e-10, f"legacy incremental vs batched mismatch: {error:.3e}"
    assert (batched - other).abs().max().item() > 1e-6, "legacy mode stopped depending on the memory"
    # The per-layer memory cache is the legacy path's, and only its.
    assert policy._memory_kv is not None and len(policy._memory_kv) == policy.num_layers
    assert policy._read_kv is None
    print(f"[ok] legacy kv_prepend preserved: incremental == batched ({error:.3e}), Z still matters")


# --------------------------------------------------------------------------------------------------
# A2 / A3 -- lag-aware KL and the instrumentation pack
# --------------------------------------------------------------------------------------------------

K = 3
T_TRIAL = K * T_EPISODE


def _make_rl_policy(num_envs: int, seed: int = 0) -> ActorCriticTrialMemory:
    torch.manual_seed(seed)
    sample_obs = TensorDict({"policy": torch.zeros(num_envs, OBS_DIM)}, batch_size=[num_envs], device=DEVICE)
    policy = ActorCriticTrialMemory(
        obs=sample_obs,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=ACTION_DIM,
        d_model=D_MODEL,
        num_layers=2,
        num_heads=4,
        num_memory_tokens=NUM_MEMORY,
        max_episode_length=T_EPISODE,
        ff_mult=2,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        init_noise_std=0.5,
    )
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if name in ("std", "log_std"):
                continue
            parameter.add_(0.1 * torch.randn_like(parameter))
    return policy


def _make_rl_ppo(policy: ActorCriticTrialMemory, num_envs: int, **kwargs) -> PPO:
    defaults = dict(
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=0.0,
        schedule="fixed",
        desired_kl=None,
        gamma=0.999,
        lam=0.99,
        device=DEVICE,
        defer_obs_normalization=True,
    )
    defaults.update(kwargs)
    ppo = PPO(policy, **defaults)
    sample_obs = TensorDict({"policy": torch.zeros(num_envs, OBS_DIM)}, batch_size=[num_envs], device=DEVICE)
    ppo.init_storage("rl", num_envs, T_TRIAL, sample_obs, [ACTION_DIM])
    return ppo


def _collect(ppo: PPO, num_envs: int, seed: int = 0) -> None:
    """One aligned trial of ``K`` episodes per environment."""
    dones = torch.zeros(T_TRIAL, num_envs, dtype=torch.bool, device=DEVICE)
    trial_dones = torch.zeros_like(dones)
    for episode in range(1, K + 1):
        dones[episode * T_EPISODE - 1] = True
    trial_dones[T_TRIAL - 1] = True

    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator, device=DEVICE)  # noqa: E731
    obs = TensorDict({"policy": randn(num_envs, OBS_DIM)}, batch_size=[num_envs], device=DEVICE)
    with torch.no_grad():
        for step in range(T_TRIAL):
            ppo.act(obs)
            next_obs = TensorDict({"policy": randn(num_envs, OBS_DIM)}, batch_size=[num_envs], device=DEVICE)
            extras = {"time_outs": dones[step], "trial_done": trial_dones[step]}
            ppo.process_env_step(next_obs, randn(num_envs), dones[step], extras)
            obs = next_obs
        ppo.compute_returns(obs)


def _stale_rollout(num_envs: int, stale_envs: list[int], lag: int, seed: int, **ppo_kwargs) -> tuple[PPO, dict]:
    """Collect one rollout, backdate ``stale_envs`` by ``lag`` updates and desynchronize their stored ``mu``.

    The backdated environments are exactly the rows whose PPO ratio is *allowed* to have drifted: their data was
    collected under an older policy. Moving their stored behavior mean is how a real stale row looks to the KL.
    """
    policy = _make_rl_policy(num_envs, seed=seed)
    ppo = _make_rl_ppo(policy, num_envs, **ppo_kwargs)
    _collect(ppo, num_envs, seed=seed + 1)
    storage = ppo.storage
    for env in stale_envs:
        storage.policy_versions[:, env] = storage.policy_version - lag
        storage.mu[:, env] += 2.0  # the stale policy's mean, far from the current one
    return ppo, {"num_envs": num_envs, "stale_envs": stale_envs}


# A trial that does NOT fit one collection window: 3 episodes of 6 steps collected 4 steps at a time, i.e.
# the production regime (num_steps_per_env=32 against a 240-step trial) in miniature.
WINDOW = 4
CARRY = 20
LONG_EP = 6
LONG_TRIAL = 3 * LONG_EP  # 18 steps == 4.5 windows
NUM_WINDOWS = 5


def _collect_windowed(ppo: PPO, num_envs: int, num_windows: int, seed: int = 0) -> None:
    """Collect ``num_windows`` rollouts of ``WINDOW`` steps, advancing the policy version between them.

    The trial spans several windows, so it is carried across every rollout boundary and only becomes
    trainable in the last one -- exactly the situation in which the *trial's* lag and a *row's* lag differ.
    Between windows the storage is rolled and the version bumped by hand, which is what ``update()`` does at
    its end; ``update()`` itself cannot run there because no trial has completed yet.
    """
    total = num_windows * WINDOW
    dones = torch.zeros(total, num_envs, dtype=torch.bool, device=DEVICE)
    trial_dones = torch.zeros_like(dones)
    for episode in range(1, 4):
        dones[episode * LONG_EP - 1] = True
    trial_dones[LONG_TRIAL - 1] = True

    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator, device=DEVICE)  # noqa: E731
    obs = TensorDict({"policy": randn(num_envs, OBS_DIM)}, batch_size=[num_envs], device=DEVICE)
    with torch.no_grad():
        for window in range(num_windows):
            for local in range(WINDOW):
                step = window * WINDOW + local
                ppo.act(obs)
                next_obs = TensorDict({"policy": randn(num_envs, OBS_DIM)}, batch_size=[num_envs], device=DEVICE)
                extras = {"time_outs": dones[step], "trial_done": trial_dones[step]}
                ppo.process_env_step(next_obs, randn(num_envs), dones[step], extras)
                obs = next_obs
            ppo.compute_returns(obs)
            if window < num_windows - 1:
                ppo.storage.clear()
                ppo.storage.policy_version += 1


def test_row_lags_ramp_across_collection_windows() -> None:
    """A trial spanning 5 windows must carry a per-row lag ramp, not one trial-wide lag.

    This is the regime the lag-aware KL exists for: the pair's ``lags`` entry is 4 for every row (the trial
    STARTED four updates ago), so a trial-level ``lag <= 1`` mask would be empty and the KL control would
    silently fall back to "all rows" on every update.
    """
    num_envs = 2
    policy = _make_rl_policy(num_envs, seed=17)
    ppo = PPO(
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1e-4,
        schedule="adaptive",
        desired_kl=0.01,
        kl_early_stop_factor=2.0,
        gamma=0.999,
        lam=0.99,
        device=DEVICE,
        defer_obs_normalization=True,
        trial_carry_steps=CARRY,
        max_policy_lag=9,
    )
    sample_obs = TensorDict({"policy": torch.zeros(num_envs, OBS_DIM)}, batch_size=[num_envs], device=DEVICE)
    ppo.init_storage("rl", num_envs, WINDOW, sample_obs, [ACTION_DIM])
    _collect_windowed(ppo, num_envs, NUM_WINDOWS, seed=19)

    storage = ppo.storage
    index = storage.build_trial_pairs()
    assert index["num_pairs"] == 3 * num_envs, f"expected one closed trial per env, got {index}"

    batch = next(storage.trial_pair_mini_batch_generator(policy, num_mini_batches=1, num_epochs=1))
    loss_mask = batch["target"]["loss_mask"]
    row_lags = batch["target"]["row_lags"][loss_mask]
    # Window w was collected at version w and the pool is consumed at version 4, so its rows are at lag 4 - w.
    # Windows 0..3 contribute 4 trial rows each, window 4 contributes the trial's last 2 steps.
    counts = torch.bincount(row_lags, minlength=5).tolist()
    assert counts == [2 * num_envs, 4 * num_envs, 4 * num_envs, 4 * num_envs, 4 * num_envs], (
        f"row lags did not ramp across the carry boundaries: {counts}"
    )
    # ... and the trial-level lag is a single stale number for every one of those rows.
    assert bool((batch["lags"] == 4).all()), f"trial lags: {batch['lags'].tolist()}"

    fresh_fraction = float((row_lags <= 1).to(torch.float32).mean())
    assert 0.2 < fresh_fraction < 0.5, f"fresh rows are {fresh_fraction:.2%}; neither ~0 nor ~all was expected"

    # Only the rows older than one update are desynchronized, so a correct kl_fresh sees none of them.
    stale = storage.policy_versions.squeeze(-1) <= storage.policy_version - 2
    storage.mu[stale] += 2.0
    loss_dict = ppo.update()

    assert loss_dict["kl_all"] > 0.5, f"the stale rows did not move kl_all ({loss_dict['kl_all']}); test is vacuous"
    assert loss_dict["kl_fresh"] < 1e-3, f"kl_fresh picked up rows collected more than one update ago: {loss_dict}"
    assert loss_dict["kl_early_stopped"] == 0.0
    assert ppo.learning_rate > 1e-4, "the adaptive LR reacted to kl_all instead of kl_fresh"
    assert loss_dict["ratio_max_lag_0_1"] > 0.0 and loss_dict["ratio_max_lag_2_4"] > 0.0
    assert loss_dict["ratio_max_lag_5plus"] == 0.0
    # The drift metric picks its rows the same way, so it must not go missing in this regime.
    assert "drift/kl_vs_init" in loss_dict, "the drift diagnostic found no fresh row and disappeared"
    assert loss_dict["drift/mean_abs_dmu"] == 0.0
    print(
        f"[ok] per-row lags across {NUM_WINDOWS} windows: bincount {counts}, trial lag 4 for all,"
        f" {fresh_fraction:.1%} of rows fresh, kl_fresh {loss_dict['kl_fresh']:.2e} vs kl_all"
        f" {loss_dict['kl_all']:.2f}"
    )


def test_row_lags_are_zero_within_a_single_window() -> None:
    """The old situation must be unchanged: a trial that closes inside its own rollout is fully on-policy."""
    num_envs = 4
    policy = _make_rl_policy(num_envs, seed=21)
    ppo = _make_rl_ppo(policy, num_envs)
    _collect(ppo, num_envs, seed=22)
    batch = next(ppo.storage.trial_pair_mini_batch_generator(policy, num_mini_batches=1, num_epochs=1))
    row_lags = batch["target"]["row_lags"][batch["target"]["loss_mask"]]
    assert row_lags.numel() == num_envs * T_TRIAL and bool((row_lags == 0).all()), "a fresh rollout is not lag 0"
    assert bool((batch["lags"] == 0).all())
    print(f"[ok] single-window trial: all {row_lags.numel()} rows at lag 0")


def test_lag_aware_kl_ignores_stale_rows() -> None:
    """``kl_fresh`` must aggregate lag <= 1 rows only, and it -- not ``kl_all`` -- must drive LR and early stop."""
    num_envs = 4
    ppo, _ = _stale_rollout(
        num_envs,
        stale_envs=[0, 1],
        lag=3,
        seed=7,
        max_policy_lag=9,
        num_learning_epochs=2,
        num_mini_batches=1,
        learning_rate=1e-4,
        schedule="adaptive",
        desired_kl=0.01,
        kl_early_stop_factor=2.0,
    )
    loss_dict = ppo.update()

    assert loss_dict["kl_all"] > 0.5, f"the stale rows did not move kl_all ({loss_dict['kl_all']}); test is vacuous"
    assert loss_dict["kl_fresh"] < 1e-3, f"kl_fresh picked up the stale rows: {loss_dict['kl_fresh']}"
    assert loss_dict["kl_early_stopped"] == 0.0, "the early stop fired on stale data"
    assert loss_dict["kl_minibatches_run"] == 2.0, "the update did not run both epochs"
    assert ppo.learning_rate > 1e-4, "the adaptive LR reacted to kl_all instead of kl_fresh"
    # Lag buckets: rows at lag 3 exist, rows at lag >= 5 do not.
    assert loss_dict["ratio_max_lag_0_1"] > 0.0 and loss_dict["ratio_max_lag_2_4"] > 0.0
    assert loss_dict["ratio_max_lag_5plus"] == 0.0

    # The control: when the FRESH rows are the divergent ones, the very same machinery does react.
    ppo_fresh, _ = _stale_rollout(
        num_envs,
        stale_envs=[0, 1, 2, 3],
        lag=0,  # everything is on-policy, so every perturbed row counts as fresh
        seed=7,
        max_policy_lag=9,
        num_learning_epochs=2,
        num_mini_batches=1,
        learning_rate=1e-4,
        schedule="adaptive",
        desired_kl=0.01,
        kl_early_stop_factor=2.0,
    )
    control = ppo_fresh.update()
    assert control["kl_fresh"] > 0.5, "the control did not diverge; it cannot separate kl_fresh from kl_all"
    assert control["kl_early_stopped"] == 1.0, "the early stop must fire on fresh divergence"
    assert ppo_fresh.learning_rate < 1e-4, "the adaptive LR did not back off on fresh divergence"
    print(
        f"[ok] lag-aware KL: kl_fresh {loss_dict['kl_fresh']:.2e} vs kl_all {loss_dict['kl_all']:.2f} (no early"
        f" stop, lr up); fresh divergence -> kl_fresh {control['kl_fresh']:.2f}, early stop, lr down"
    )


def test_instrumentation_pack_is_logged() -> None:
    """Every key of the instrumentation pack is present, finite and reacts to a real update."""
    num_envs = 4
    policy = _make_rl_policy(num_envs, seed=9)
    ppo = _make_rl_ppo(policy, num_envs, learning_rate=1e-3, num_learning_epochs=2, num_mini_batches=2)
    _collect(ppo, num_envs, seed=10)
    loss_dict = ppo.update()

    expected = {
        "kl_fresh",
        "kl_all",
        "update_norm/embed",
        "update_norm/blocks",
        "update_norm/action_head",
        "update_norm/sigma",
        "update_norm/memory_read",
        "update_norm/writer",
        "update_norm/critic",
        "drift/kl_vs_init",
        "drift/mean_abs_dmu",
        "drift/sigma_ratio",
        "adv/return_corr",
        "adv/frac_pos_in_top_quartile_episodes",
        "adv/frac_pos_in_bottom_quartile_episodes",
        "memory_gate_attn",
        "memory_gate_ff",
    }
    missing = expected - set(loss_dict)
    assert not missing, f"missing loss_dict keys: {sorted(missing)}"
    for key, value in loss_dict.items():
        assert torch.isfinite(torch.tensor(value)), f"{key} = {value} is not finite"

    # The reference is the policy as it was at the first update, and the drift is measured before that update's
    # first optimizer step -- so it is exactly zero there, and grows afterwards. (``drift/kl_vs_init`` has a
    # floor of ``num_actions * 1e-5`` from the epsilon in the KL formula this repo uses everywhere.)
    assert loss_dict["drift/mean_abs_dmu"] == 0.0, f"the drift reference is not the acting policy: {loss_dict}"
    assert loss_dict["drift/kl_vs_init"] < 1e-4
    assert abs(loss_dict["drift/sigma_ratio"] - 1.0) < 1e-6
    assert loss_dict["update_norm/blocks"] > 0.0, "a real update moved no trunk parameter"
    assert loss_dict["update_norm/memory_read"] > 0.0, "the read layer (gates included) never moved"
    assert -1.0 <= loss_dict["adv/return_corr"] <= 1.0
    for key in ("adv/frac_pos_in_top_quartile_episodes", "adv/frac_pos_in_bottom_quartile_episodes"):
        assert 0.0 <= loss_dict[key] <= 1.0

    _collect(ppo, num_envs, seed=11)
    second = ppo.update()
    assert second["drift/mean_abs_dmu"] > 0.0, "the drift never grows; the reference is being re-captured"

    # ... and the flags actually switch the diagnostics off.
    quiet_policy = _make_rl_policy(num_envs, seed=9)
    quiet = _make_rl_ppo(
        quiet_policy, num_envs, learning_rate=1e-3, log_policy_drift=False, log_advantage_audit=False
    )
    _collect(quiet, num_envs, seed=10)
    quiet_dict = quiet.update()
    assert not any(key.startswith(("drift/", "adv/")) for key in quiet_dict), "the diagnostics flags are ignored"
    assert quiet.reference_policy is None
    print(
        f"[ok] instrumentation: {len(expected)} keys logged, |dmu| vs init 0 ->"
        f" {second['drift/mean_abs_dmu']:.2e}, update_norm/blocks = {loss_dict['update_norm/blocks']:.2e}"
    )


def test_reference_policy_copy_is_frozen_and_light() -> None:
    """The drift reference is a detached eval-mode copy that does not carry the acting-time buffers."""
    num_envs = 4
    policy = _make_rl_policy(num_envs, seed=12)
    ppo = _make_rl_ppo(policy, num_envs)
    _collect(ppo, num_envs, seed=13)  # allocates the KV cache and a live memory
    assert policy._key_cache is not None and policy._memory is not None

    ppo.capture_reference_policy()
    reference = ppo.reference_policy
    assert reference is not policy
    assert reference._key_cache is None and reference._memory is None, "the reference copied the acting buffers"
    assert not reference.training
    assert all(not parameter.requires_grad for parameter in reference.parameters())
    for (name, before), (_, after) in zip(policy.named_parameters(), reference.named_parameters()):
        assert torch.equal(before.detach(), after), f"{name} was not copied faithfully"
    # The original is untouched by the swap-out the copy performs.
    assert policy._key_cache is not None and policy._memory is not None
    print("[ok] reference policy: frozen, eval mode, no acting-time buffers, parameters identical")


if __name__ == "__main__":
    test_defaults_and_placement()
    test_identity_at_init()
    test_gate_open_lets_the_memory_through()
    test_gate_gates_the_gradient_to_the_writer()
    test_step_matches_sequence_in_residual_mode()
    test_cached_read_kv_follows_the_memory()
    test_legacy_kv_prepend_is_unchanged()
    test_row_lags_ramp_across_collection_windows()
    test_row_lags_are_zero_within_a_single_window()
    test_lag_aware_kl_ignores_stale_rows()
    test_instrumentation_pack_is_logged()
    test_reference_policy_copy_is_frozen_and_light()
    print("all residual memory read tests passed")
