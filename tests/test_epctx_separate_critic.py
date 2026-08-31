# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ``ActorCriticEpisodeContext(critic_design="separate_trunk")``.

Self-contained: pure torch, no Isaac Sim / SimulationApp needed. Run with::

    source /home/yandabao/miniforge3/etc/profile.d/conda.sh && conda activate patlab
    python tests/test_epctx_separate_critic.py

The separate design gives the value function its own transformer over the SAME (non-privileged) observation,
with every module named ``critic_*`` so a critic-only warm-up trains all of it and no actor parameter is
reachable from the value loss. The load-bearing statements:

* ``test_only_critic_prefixed_parameters_are_new`` -- the warm-up freeze filter (``name.startswith("critic")``)
  partitions the model exactly,
* ``test_acting_values_match_batched_reinference`` -- the critic's KV-cache mirror reproduces a from-scratch
  batched pass across staggered episode (and trial) boundaries, with and without a memory,
* ``test_value_loss_does_not_touch_the_actor`` -- the entire point of the second trunk,
* ``test_ppo_update_runs_end_to_end`` / ``test_critic_only_warmup_step`` -- through the real PPO update.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import EpisodeContextPPO
from rsl_rl.modules import ActorCriticEpisodeContext

DEVICE = "cpu"
OBS_DIM = 5
CRITIC_OBS_DIM = 7
ACTION_DIM = 4
NUM_ENVS = 4
T_EPISODE = 12
STEPS_PER_ENV = 8
GAMMA = 0.99
LAM = 0.95
# Per-environment episode lengths (cycled): different per env AND from one episode to the next.
EPISODE_LENGTHS = (
    (6, 9, 7, 11, 8, 5),
    (9, 7, 11, 6, 12, 4),
    (7, 11, 6, 9, 5, 12),
    (11, 6, 9, 7, 10, 3),
)


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _sample_obs(num_envs: int = NUM_ENVS, generator: torch.Generator | None = None, dtype=torch.float32) -> TensorDict:
    if generator is None:
        policy_obs = torch.zeros(num_envs, OBS_DIM, device=DEVICE, dtype=dtype)
        critic_obs = torch.zeros(num_envs, CRITIC_OBS_DIM, device=DEVICE, dtype=dtype)
    else:
        policy_obs = torch.randn(num_envs, OBS_DIM, generator=generator, device=DEVICE, dtype=dtype)
        critic_obs = torch.randn(num_envs, CRITIC_OBS_DIM, generator=generator, device=DEVICE, dtype=dtype)
    return TensorDict({"policy": policy_obs, "critic": critic_obs}, batch_size=[num_envs], device=DEVICE)


def _make_policy(
    critic_design: str = "separate_trunk",
    seed: int = 0,
    context_length: int = T_EPISODE,
    memory_tokens: int = 0,
    episodes_per_trial: int = 2,
    perturb: bool = True,
) -> ActorCriticEpisodeContext:
    torch.manual_seed(seed)
    policy = ActorCriticEpisodeContext(
        obs=_sample_obs(),
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=ACTION_DIM,
        context_length=context_length,
        d_model=32,
        num_layers=2,
        num_heads=4,
        max_episode_length=T_EPISODE,
        ff_mult=2,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        init_noise_std=0.5,
        critic_design=critic_design,
        memory_tokens=memory_tokens,
        episodes_per_trial=episodes_per_trial,
    )
    if perturb:
        # Off the initialization: a near-identity trunk (and a zero-delta writer) makes every comparison vacuous.
        with torch.no_grad():
            for name, parameter in policy.named_parameters():
                if name in ("std", "log_std"):
                    continue
                parameter.add_(0.1 * torch.randn_like(parameter))
    return policy.eval()


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
    ppo.init_storage("rl", NUM_ENVS, STEPS_PER_ENV, _sample_obs(), [ACTION_DIM])
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


def _staggered_schedule(num_steps: int, episodes_per_trial: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Dones / trial dones ``[S, N]`` from :data:`EPISODE_LENGTHS`, a trial closing every K-th done."""
    dones = torch.zeros(num_steps, NUM_ENVS, dtype=torch.bool, device=DEVICE)
    trial_dones = torch.zeros(num_steps, NUM_ENVS, dtype=torch.bool, device=DEVICE)
    for env, lengths in enumerate(EPISODE_LENGTHS):
        step, closed, i = -1, 0, 0
        while True:
            step += lengths[i % len(lengths)]
            i += 1
            if step >= num_steps:
                break
            dones[step, env] = True
            closed += 1
            trial_dones[step, env] = closed % episodes_per_trial == 0
    return dones, trial_dones


def _episodes(dones: torch.Tensor, trial_dones: torch.Tensor, env: int) -> list[tuple[int, int, bool]]:
    """``(start, last step, trial_end)`` of every episode of ``env``; a trailing partial episode is included."""
    num_steps = dones.shape[0]
    episodes, start = [], 0
    for step in range(num_steps):
        if bool(dones[step, env]) or step == num_steps - 1:
            episodes.append((start, step, bool(trial_dones[step, env])))
            start = step + 1
    return episodes


def _collect(ppo: EpisodeContextPPO, dones: torch.Tensor, offset: int, generator: torch.Generator) -> None:
    """One rollout through ``PPO.act`` / ``PPO.process_env_step``, ending with ``compute_returns``."""
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
# (a) parameter naming
# --------------------------------------------------------------------------------------------------


def test_only_critic_prefixed_parameters_are_new() -> None:
    """Everything the separate design adds must be inside the warm-up filter, and nothing else may move."""
    for memory_tokens in (0, 2):
        separate = _make_policy(memory_tokens=memory_tokens, perturb=False)
        shared = _make_policy(critic_design="shared_trunk", memory_tokens=memory_tokens, perturb=False)
        separate_names = {name for name, _ in separate.named_parameters()}
        shared_names = {name for name, _ in shared.named_parameters()}

        trainable = {name for name in separate_names if name.startswith("critic")}
        frozen = separate_names - trainable
        expected_frozen = {name for name in shared_names if not name.startswith("critic")}
        assert frozen == expected_frozen, f"parameters outside the warm-up filter changed: {frozen ^ expected_frozen}"

        prefixes = ["critic.", "critic_token_embed.", "critic_start_embed", "critic_blocks.", "critic_final_norm."]
        if memory_tokens > 0:
            prefixes.append("critic_memory_pos_embed")
        assert all(name.startswith(tuple(prefixes)) for name in trainable), f"unexpected critic tensor: {trainable}"
        for prefix in prefixes:
            assert any(name.startswith(prefix) for name in trainable), f"no parameter under {prefix}"
        # Buffers too (the observation normalizer is shared, so there must be no critic_* buffer at all).
        assert not any(name.startswith("critic_") for name, _ in separate.named_buffers())
        # ... and the shared design is untouched by the flag.
        assert not any(name.startswith("critic_") for name in shared_names), "shared_trunk grew a critic_* pathway"
        num_new = len(separate_names) - len(shared_names)
        assert num_new > 0
        print(
            f"[ok] M={memory_tokens}: {len(trainable)} critic tensors trainable ({num_new} new),"
            f" {len(frozen)} frozen"
        )


def test_value_from_hidden_refuses_the_actor_readout() -> None:
    """``value_from_hidden`` must raise rather than silently return a value off the ACTOR trunk."""
    policy = _make_policy()
    try:
        policy.value_from_hidden(torch.zeros(2, policy.d_model))
    except RuntimeError:
        pass
    else:
        raise AssertionError("value_from_hidden() returned an actor-trunk value under separate_trunk")
    # ... while the shared design keeps working exactly as before.
    shared = _make_policy(critic_design="shared_trunk")
    assert shared.value_from_hidden(torch.zeros(2, 3, shared.d_model)).shape == (2, 3, 1)
    print("[ok] value_from_hidden() blocked under separate_trunk, unchanged under shared_trunk")


# --------------------------------------------------------------------------------------------------
# (c) the acting-time critic cache reproduces a batched re-inference
# --------------------------------------------------------------------------------------------------


def _reference_values(
    policy: ActorCriticEpisodeContext, obs: torch.Tensor, dones: torch.Tensor, trial_dones: torch.Tensor
) -> torch.Tensor:
    """Values ``[S, N, 1]`` from the NO-CACHE path: one ``forward_sequence(critic=True)`` per episode.

    With a memory the ``Z`` chain is rebuilt by hand off the ACTOR pass (``z_init`` at a trial start, else
    ``G(H_prev)``) -- exactly the ``Z`` the acting path fed the critic mirror.
    """
    num_steps, num_envs = dones.shape
    num_memory, span = policy.num_memory_tokens, policy.hidden_history_span
    values = torch.zeros(num_steps, num_envs, 1, dtype=obs.dtype, device=DEVICE)
    with torch.no_grad():
        z_init = policy.z_init.detach().unsqueeze(0) if num_memory > 0 else None
        for env in range(num_envs):
            memory = z_init
            for start, end, trial_end in _episodes(dones, trial_dones, env):
                frames = obs[start : end + 1, env : env + 1]
                critic_hidden = policy.forward_sequence(frames, memory=memory, critic=True)  # [S_e, 1, d]
                values[start : end + 1, env] = policy.critic(critic_hidden.squeeze(1))
                if num_memory == 0:
                    continue
                hidden = policy.forward_sequence(frames, memory=memory)
                length = end + 1 - start
                episode_hidden = torch.zeros(1, span, policy.d_model, dtype=obs.dtype, device=DEVICE)
                valid = torch.zeros(1, span, dtype=torch.bool, device=DEVICE)
                episode_hidden[0, :num_memory] = policy.memory_readout(memory)[0]
                episode_hidden[0, num_memory : num_memory + length] = hidden.squeeze(1)
                valid[0, : num_memory + length] = True
                memory = z_init if trial_end else policy.write_memory(episode_hidden, mask=valid)[0]
    return values


def _roll_values(
    policy: ActorCriticEpisodeContext, obs: torch.Tensor, dones: torch.Tensor, trial_dones: torch.Tensor
) -> torch.Tensor:
    """Drive the acting path as PPO does (``act`` -> ``evaluate`` -> ``reset``); return the values ``[S, N, 1]``."""
    num_steps, num_envs = dones.shape
    policy.initialize_state(num_envs, DEVICE, dtype=obs.dtype)
    values = []
    with torch.no_grad():
        for step in range(num_steps):
            frame = TensorDict(
                {"policy": obs[step], "critic": torch.zeros(num_envs, CRITIC_OBS_DIM, dtype=obs.dtype)},
                batch_size=[num_envs],
                device=DEVICE,
            )
            policy.act(frame)
            values.append(policy.evaluate(frame))
            policy.reset(dones[step], trial_dones=trial_dones[step])
    return torch.stack(values)


def test_acting_values_match_batched_reinference() -> None:
    """The critic KV mirror must reproduce a from-scratch batched pass across staggered episode boundaries."""
    num_steps = 40
    for label, memory_tokens, episodes_per_trial, context_length in (
        ("no memory, L = T", 0, 1, T_EPISODE),
        ("no memory, sliding L = 5", 0, 1, 5),
        ("M = 2, K = 3, L = T", 2, 3, T_EPISODE),
    ):
        policy = _make_policy(
            seed=3, context_length=context_length, memory_tokens=memory_tokens, episodes_per_trial=episodes_per_trial
        ).to(torch.float64)
        dones, trial_dones = _staggered_schedule(num_steps, episodes_per_trial)
        generator = torch.Generator(device=DEVICE).manual_seed(11)
        obs = torch.randn(num_steps, NUM_ENVS, OBS_DIM, generator=generator, device=DEVICE, dtype=torch.float64)

        acted = _roll_values(policy, obs, dones, trial_dones)
        reference = _reference_values(policy, obs, dones, trial_dones)
        error = (acted - reference).abs().max().item()
        assert reference.abs().max().item() > 1e-3, "the reference values are ~0; the comparison would be vacuous"
        assert error < 1e-5, f"[{label}] acting critic values drift from the batched pass by {error:.3e}"
        # The critic must not be a constant function of the context, or the test would pass trivially.
        assert acted.std().item() > 1e-3, "the acting values barely vary; the comparison is vacuous"
        # Negative control: the same value head on the ACTOR trunk must disagree, otherwise ``critic=True``
        # would be a no-op and everything above would be comparing the actor pathway with itself.
        start, end, _ = _episodes(dones, trial_dones, 0)[0]
        with torch.no_grad():
            actor_side = policy.critic(policy.forward_sequence(obs[start : end + 1, 0:1]).squeeze(1))
        gap = (actor_side - reference[start : end + 1, 0]).abs().max().item()
        assert gap > 1e-3, f"[{label}] the critic pathway reproduces the actor trunk's value ({gap:.3e})"
        print(
            f"[ok] critic KV mirror [{label}]: max |dV| = {error:.3e} on values up to"
            f" {reference.abs().max().item():.3f} (actor-pathway control differs by {gap:.3e})"
        )


def test_bootstrap_value_does_not_commit_to_the_critic_cache() -> None:
    """``compute_returns`` peeks the terminal frame; neither cache may move (the next rollout re-reads it)."""
    dones = _episode_schedule(STEPS_PER_ENV)
    policy = _make_policy(seed=6)
    ppo = _make_ppo(policy)
    generator = torch.Generator(device=DEVICE).manual_seed(21)

    obs = _sample_obs(NUM_ENVS, generator)
    with torch.no_grad():
        for step in range(STEPS_PER_ENV):
            ppo.act(obs)
            next_obs = _sample_obs(NUM_ENVS, generator)
            ppo.process_env_step(next_obs, torch.randn(NUM_ENVS, generator=generator, device=DEVICE), dones[step], {})
            obs = next_obs
        positions_before = policy.positions.clone()
        actor_before = [cache.clone() for cache in policy._key_cache]
        critic_before = [cache.clone() for cache in policy._critic_key_cache]
        ppo.compute_returns(obs)

    assert torch.equal(policy.positions, positions_before), "the bootstrap advanced the episode step"
    for before, after in zip(actor_before, policy._key_cache):
        assert torch.equal(before, after), "the bootstrap frame was committed to the ACTOR KV cache"
    for before, after in zip(critic_before, policy._critic_key_cache):
        assert torch.equal(before, after), "the bootstrap frame was committed to the CRITIC KV cache"
    assert ppo.storage.returns.abs().sum().item() > 0.0
    print("[ok] the bootstrap peeks the last frame in both pathways without committing it")


# --------------------------------------------------------------------------------------------------
# (b) gradient separation
# --------------------------------------------------------------------------------------------------


def test_value_loss_does_not_touch_the_actor() -> None:
    """A value loss must leave every actor-side parameter (trunk, heads, writer, z_init, noise) untouched."""
    for memory_tokens in (0, 2):
        dones = _episode_schedule(2 * STEPS_PER_ENV)
        policy = _make_policy(seed=7, memory_tokens=memory_tokens)
        ppo = _make_ppo(policy)
        generator = torch.Generator(device=DEVICE).manual_seed(31)
        _collect(ppo, dones, 0, generator)
        ppo.update()  # so the second rollout's prefix reaches into the previous one
        _collect(ppo, dones, STEPS_PER_ENV, generator)

        batch = next(ppo.storage.recurrent_mini_batch_generator(num_mini_batches=1, num_epochs=1))
        obs_batch, prefix = batch[0], batch[8][1]
        assert prefix is not None, "the generator must hand the prefix to the critic slot as well"

        policy.zero_grad(set_to_none=True)
        values = policy.critic(policy.forward_window(obs_batch, prefix=prefix, critic=True))
        values.pow(2).mean().backward()
        touched = [
            name
            for name, parameter in policy.named_parameters()
            if not name.startswith("critic") and parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        ]
        assert not touched, f"the value loss reached actor-side parameters: {touched}"
        trained = [
            name
            for name, parameter in policy.named_parameters()
            if name.startswith("critic") and parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        ]
        assert any(name.startswith("critic_blocks") for name in trained), "the critic trunk received no gradient"
        assert any(name.startswith("critic_token_embed") for name in trained), "the critic embedding got no gradient"
        assert any(name.startswith("critic.") for name in trained), "the value head received no gradient"
        if memory_tokens > 0:
            assert any(name.startswith("critic_memory_pos_embed") for name in trained), "the memory rows are inert"

        # ... and conversely: the actor's own loss still trains the actor and never the critic pathway.
        policy.zero_grad(set_to_none=True)
        policy.forward_window(obs_batch, prefix=prefix).square().sum().backward()
        assert float(policy.blocks[0].attn.q_proj.weight.grad.abs().sum()) > 0.0, "the actor trunk got no gradient"
        if memory_tokens > 0:
            assert float(policy.z_init.grad.abs().sum()) > 0.0, "the actor loss lost z_init"
        assert all(
            parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0
            for name, parameter in policy.named_parameters()
            if name.startswith("critic")
        ), "the actor loss reached the critic pathway"
        print(f"[ok] M={memory_tokens}: {len(trained)} critic tensors trained, 0 actor tensors touched")


# --------------------------------------------------------------------------------------------------
# (d) loading a BC checkpoint that has no critic pathway
# --------------------------------------------------------------------------------------------------


def test_bc_state_dict_without_critic_keys_loads() -> None:
    """A BC init (actor only) must load with ``strict=False``, leaving exactly the ``critic*`` keys missing."""
    donor = _make_policy(seed=1, memory_tokens=2)
    bc_state = {key: value for key, value in donor.state_dict().items() if not key.startswith("critic")}
    assert bc_state, "the donor state dict is empty; the test would be vacuous"

    policy = _make_policy(seed=2, memory_tokens=2)
    fresh_critic = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
        if name.startswith("critic")
    }
    report = nn.Module.load_state_dict(policy, bc_state, strict=False)
    assert not report.unexpected_keys, f"unexpected keys: {report.unexpected_keys}"
    assert report.missing_keys, "nothing was missing; the donor still carried a critic pathway"
    assert all(key.startswith("critic") for key in report.missing_keys), (
        f"a non-critic key is missing from a BC checkpoint: "
        f"{[key for key in report.missing_keys if not key.startswith('critic')]}"
    )
    # The actor side really loaded, and the critic pathway is left at its fresh initialization.
    for name, parameter in policy.named_parameters():
        if name.startswith("critic"):
            assert torch.equal(parameter, fresh_critic[name]), f"{name} was overwritten by the BC load"
        else:
            assert torch.equal(parameter, donor.state_dict()[name]), f"{name} did not load"

    # The module's own (noise-checking, RoPE-stripping) entry point must accept it too.
    policy = _make_policy(seed=2, memory_tokens=2)
    assert policy.load_state_dict(bc_state, strict=False) is True
    print(f"[ok] BC load: {len(report.missing_keys)} missing keys, all critic*; the actor side is bit-exact")


# --------------------------------------------------------------------------------------------------
# (e) the real PPO update
# --------------------------------------------------------------------------------------------------


def test_ppo_update_runs_end_to_end() -> None:
    """A full separate-critic update must run, keep the epoch-0 ratio AND value canaries, and stay finite."""
    dones = _episode_schedule(2 * STEPS_PER_ENV)
    policy = _make_policy(seed=13)
    ppo = _make_ppo(policy)
    generator = torch.Generator(device=DEVICE).manual_seed(17)

    for rollout in range(2):
        _collect(ppo, dones, rollout * STEPS_PER_ENV, generator)
        stored_values = ppo.storage.values.clone()

        recorded: list[torch.Tensor] = []
        stock_evaluate = policy.evaluate

        def recording_evaluate(*args, _stock=stock_evaluate, _sink=recorded, **kwargs) -> torch.Tensor:
            value = _stock(*args, **kwargs)
            _sink.append(value.detach().clone())
            return value

        policy.evaluate = recording_evaluate
        try:
            loss_dict = ppo.update()
        finally:
            del policy.evaluate

        assert len(recorded) == 2, f"expected one evaluate() per minibatch, got {len(recorded)}"
        re_inferred = torch.cat(recorded, dim=1)
        assert re_inferred.shape == stored_values.shape
        value_error = (re_inferred - stored_values).abs().max().item()
        assert stored_values.abs().max().item() > 1e-4, "the stored values are ~0; the canary would be vacuous"
        assert value_error < 1e-5, f"epoch-0 values drifted by {value_error:.3e} (rollout {rollout})"
        assert abs(loss_dict["ratio_mean"] - 1.0) < 1e-5, f"ratio_mean = {loss_dict['ratio_mean']}"
        for key, value in loss_dict.items():
            assert value == value and abs(value) < float("inf"), f"{key} is not finite: {value}"
        assert loss_dict["value_function"] > 0.0
        ppo.commit_obs_normalization(ppo.storage.collected_observations)
        print(f"[ok] end-to-end update, rollout {rollout}: max |dV| = {value_error:.3e}, ratio 1")


def test_critic_only_warmup_step() -> None:
    """The warm-up freeze (train ``critic*`` only) must move the whole critic pathway and no actor parameter."""
    dones = _episode_schedule(STEPS_PER_ENV)
    policy = _make_policy(seed=8)
    ppo = _make_ppo(policy, learning_rate=1e-2, value_loss_coef=1.0, entropy_coef=0.0)
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(name.startswith("critic"))
    generator = torch.Generator(device=DEVICE).manual_seed(2)
    _collect(ppo, dones, 0, generator)
    before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}

    ppo.update()

    moved = {
        name: (parameter - before[name]).abs().max().item()
        for name, parameter in policy.named_parameters()
    }
    frozen_moved = {name: delta for name, delta in moved.items() if not name.startswith("critic") and delta > 0.0}
    assert not frozen_moved, f"the critic-only step moved actor parameters: {frozen_moved}"
    for prefix in ("critic.", "critic_token_embed.", "critic_blocks.", "critic_final_norm.", "critic_start_embed"):
        assert any(name.startswith(prefix) and delta > 0.0 for name, delta in moved.items()), (
            f"the critic-only step left {prefix} untrained"
        )
    print(f"[ok] critic-only warm-up step: max |d critic| = {max(moved.values()):.3e}, 0 actor parameters moved")


if __name__ == "__main__":
    test_only_critic_prefixed_parameters_are_new()
    test_value_from_hidden_refuses_the_actor_readout()
    test_acting_values_match_batched_reinference()
    test_bootstrap_value_does_not_commit_to_the_critic_cache()
    test_value_loss_does_not_touch_the_actor()
    test_bc_state_dict_without_critic_keys_loads()
    test_ppo_update_runs_end_to_end()
    test_critic_only_warmup_step()
    print("all separate-critic episode-context tests passed")
