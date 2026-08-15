# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ``ActorCriticTrialMemory(critic_trunk="separate")`` and the batched value sweep that feeds it.

Self-contained: pure torch, no Isaac Sim / SimulationApp needed. Run with::

    source /home/yandabao/miniforge3/etc/profile.d/conda.sh && conda activate patlab
    PYTHONPATH=/home/yandabao/rsl_rl-wt/worktree-local-exploration:$PYTHONPATH \
        python tests/test_separate_critic_trunk.py

With a separate critic trunk the value function has no incremental path, so PPO stores placeholder zeros
during collection and fills ``storage.values`` afterwards with one batched ``forward_sequence_critic`` sweep.
The load-bearing tests here are:

* ``test_value_sweep_matches_per_step_reference`` -- the sweep (segmentation, memory checkpoints, scatter) must
  agree with a naive per-step critic forward rebuilt by hand from the raw buffers, including the GAE bootstrap,
* ``test_timeout_bootstrap_reapplication_is_bit_exact`` -- re-applying the bootstrap after the sweep must
  reproduce the collection-time formula exactly,
* ``test_value_loss_does_not_touch_the_actor`` -- the whole point of the second trunk.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCriticTrialMemory

DEVICE = "cpu"
OBS_DIM = 5
ACTION_DIM = 3
T_EPISODE = 4
NUM_STEPS = 14  # 3 full episodes (a closed trial) + a 2-step open trial, so the rollout ends mid-trial
NUM_ENVS = 3
NUM_MEMORY = 2
GAMMA = 0.99
LAM = 0.95


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _make_policy(critic_trunk: str = "separate", seed: int = 0) -> ActorCriticTrialMemory:
    torch.manual_seed(seed)
    sample_obs = TensorDict({"policy": torch.zeros(NUM_ENVS, OBS_DIM)}, batch_size=[NUM_ENVS], device=DEVICE)
    policy = ActorCriticTrialMemory(
        obs=sample_obs,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=ACTION_DIM,
        actor_obs_normalization=False,
        d_model=16,
        num_layers=2,
        num_heads=2,
        num_memory_tokens=NUM_MEMORY,
        max_episode_length=T_EPISODE + 2,
        ff_mult=2,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        init_noise_std=0.5,
        critic_trunk=critic_trunk,
    )
    # Move everything off its initialization so the tests are not measuring a near-identity network.
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if name in ("std", "log_std"):
                continue
            parameter.add_(0.1 * torch.randn_like(parameter))
    return policy


def _make_ppo(policy: ActorCriticTrialMemory, **kwargs) -> PPO:
    defaults = dict(
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=0.0,
        schedule="fixed",
        desired_kl=None,
        gamma=GAMMA,
        lam=LAM,
        device=DEVICE,
        defer_obs_normalization=True,
    )
    defaults.update(kwargs)
    ppo = PPO(policy, **defaults)
    sample_obs = TensorDict({"policy": torch.zeros(NUM_ENVS, OBS_DIM)}, batch_size=[NUM_ENVS], device=DEVICE)
    ppo.init_storage("rl", NUM_ENVS, NUM_STEPS, sample_obs, [ACTION_DIM])
    return ppo


def _schedule() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three 4-step episodes forming one trial, then a 2-step tail that exercises all bootstrap branches.

    * env 0: the tail episode is still running at the rollout end  -> "prefix + pending token" branch,
    * env 1: the tail episode ends at the last row but the trial does not -> writer-advanced memory branch,
    * env 2: the tail ends the episode AND the trial                -> Z_init branch.
    """
    dones = torch.zeros(NUM_STEPS, NUM_ENVS, dtype=torch.bool, device=DEVICE)
    trial_dones = torch.zeros_like(dones)
    for episode in range(1, 4):
        dones[episode * T_EPISODE - 1] = True
    trial_dones[3 * T_EPISODE - 1] = True
    dones[NUM_STEPS - 1, 1] = True
    dones[NUM_STEPS - 1, 2] = True
    trial_dones[NUM_STEPS - 1, 2] = True
    time_outs = dones.clone()
    return dones, trial_dones, time_outs


def _collect(ppo: PPO, seed: int = 0) -> TensorDict:
    """Run a rollout through ``PPO.act`` / ``process_env_step``; returns the final (pending) observation."""
    dones, trial_dones, time_outs = _schedule()
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator, device=DEVICE)  # noqa: E731
    obs = TensorDict({"policy": randn(NUM_ENVS, OBS_DIM)}, batch_size=[NUM_ENVS], device=DEVICE)
    with torch.no_grad():
        for step in range(NUM_STEPS):
            ppo.act(obs)
            next_obs = TensorDict({"policy": randn(NUM_ENVS, OBS_DIM)}, batch_size=[NUM_ENVS], device=DEVICE)
            extras = {"time_outs": time_outs[step], "trial_done": trial_dones[step]}
            ppo.process_env_step(next_obs, randn(NUM_ENVS), dones[step], extras)
            obs = next_obs
    return obs


def _episode_spans(storage) -> list[list[tuple[int, int]]]:
    """``[env][episode] -> (first_row, length)`` over the live window, read straight off the stored dones.

    The live window starts at ``carry_steps - carry_len[env]``, so this covers a trial carried in from the
    previous rollout as well as the rollout just collected.
    """
    carry, end = storage.carry_steps, storage.carry_steps + storage.step
    dones = storage.dones.reshape(storage.num_transitions_per_env, NUM_ENVS).bool()
    trial_dones = storage.trial_dones.reshape(storage.num_transitions_per_env, NUM_ENVS).bool()
    spans = []
    for env in range(NUM_ENVS):
        window = carry - int(storage.carry_len[env].item())
        env_spans, start = [], window
        for row in range(window, end):
            if bool(dones[row, env]) or bool(trial_dones[row, env]):
                env_spans.append((start, row - start + 1))
                start = row + 1
        if start < end:
            env_spans.append((start, end - start))
        spans.append(env_spans)
    return spans


def _tokens(storage, start: int, length: int, env: int) -> tuple[torch.Tensor, ...]:
    """Rebuild an episode's token stream by hand: ``x_j = (o_j, a_{j-1}, r_{j-1}, d_{j-1})``, zeros at j = 0."""
    obs = storage.observations["policy"][start : start + length, env].clone()
    prev_actions = torch.zeros(length, ACTION_DIM, device=DEVICE)
    prev_rewards = torch.zeros(length, 1, device=DEVICE)
    prev_dones = torch.zeros(length, 1, device=DEVICE)
    if length > 1:
        prev_actions[1:] = storage.actions[start : start + length - 1, env]
        prev_rewards[1:] = storage.raw_rewards[start : start + length - 1, env]
        prev_dones[1:] = storage.dones[start : start + length - 1, env].float()
    return obs, prev_actions, prev_rewards, prev_dones


def _terminal_row(storage, start: int, length: int, env: int) -> tuple[torch.Tensor, ...]:
    """The terminal token ``(0, a_T, r_T, d_T)`` appended to a closed episode for the writer."""
    last = start + length - 1
    return (
        torch.zeros(1, OBS_DIM, device=DEVICE),
        storage.actions[last, env].view(1, ACTION_DIM),
        storage.raw_rewards[last, env].view(1, 1),
        storage.dones[last, env].float().view(1, 1),
    )


def _reference_memories(policy, storage, spans: list[tuple[int, int]], env: int) -> list[torch.Tensor]:
    """``Zbar`` of every episode of one environment, by walking the writer over the trial by hand."""
    memories, memory = [], policy.z_init.detach().unsqueeze(0).clone()
    trial_dones = storage.trial_dones.reshape(storage.num_transitions_per_env, NUM_ENVS).bool()
    for start, length in spans:
        memories.append(memory)
        pieces = _tokens(storage, start, length, env)
        terminal = _terminal_row(storage, start, length, env)
        full = [torch.cat([piece, extra]).unsqueeze(1) for piece, extra in zip(pieces, terminal)]
        with torch.no_grad():
            hidden = policy.forward_sequence(full[0], full[1], full[2], full[3], memory=memory)
            written, _ = policy.write_memory(memory, hidden.transpose(0, 1))
        memory = policy.z_init.detach().unsqueeze(0).clone() if bool(
            trial_dones[start + length - 1, env]
        ) else written
    return memories


def _reference_value(policy, tokens: tuple[torch.Tensor, ...], memory: torch.Tensor, upto: int) -> torch.Tensor:
    """Value of step ``upto``, from a forward over the prefix ``[0, upto]`` only (causality reference)."""
    prefix = [piece[: upto + 1].unsqueeze(1) for piece in tokens]
    with torch.no_grad():
        values = policy.forward_sequence_critic(prefix[0], prefix[1], prefix[2], prefix[3], memory=memory)
    return values[upto, 0]


# --------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------


def test_shared_mode_is_unchanged() -> None:
    """A ``"shared"`` policy must expose exactly the parameters (and the APIs) it did before this flag."""
    shared = _make_policy(critic_trunk="shared")
    names = [name for name, _ in shared.named_parameters()]
    assert not any(name.startswith("critic_") for name in names), "shared mode grew a critic_* pathway"
    hidden = torch.randn(2, 3, shared.d_model)
    assert shared.value_from_hidden(hidden).shape == (2, 3, 1)
    try:
        shared.forward_sequence_critic(
            torch.zeros(1, 1, OBS_DIM), torch.zeros(1, 1, ACTION_DIM), torch.zeros(1, 1, 1),
            torch.zeros(1, 1, 1), memory=shared.initial_memory(1)
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("forward_sequence_critic must refuse to run without a separate critic trunk")
    print(f"[ok] shared mode untouched: {len(names)} parameters, no critic_* pathway")


def test_warmup_freeze_filter_covers_the_critic_pathway() -> None:
    """``train.py`` freezes every parameter whose name does not start with "critic"; check the partition."""
    separate = _make_policy(critic_trunk="separate")
    shared = _make_policy(critic_trunk="shared")
    separate_names = {name for name, _ in separate.named_parameters()}
    shared_names = {name for name, _ in shared.named_parameters()}

    trainable = {name for name in separate_names if name.startswith("critic")}
    frozen = separate_names - trainable
    # Nothing new escapes the filter: the frozen set is exactly the shared policy's non-critic parameters.
    expected_frozen = {name for name in shared_names if not name.startswith("critic")}
    assert frozen == expected_frozen, f"parameters outside the warm-up filter changed: {frozen ^ expected_frozen}"
    # ... and every module of the new pathway is inside it.
    prefixes = (
        "critic.",
        "critic_token_embed.",
        "critic_pos_embed",
        "critic_start_embed",
        "critic_memory_pos_embed",
        "critic_blocks.",
        "critic_final_norm.",
    )
    if separate.residual_memory_read:
        # The residual memory interface adds the critic's own gated read layer (gates included).
        prefixes = prefixes + ("critic_memory_read.",)
    assert all(name.startswith(prefixes) for name in trainable), f"unexpected critic parameter: {trainable}"
    for prefix in prefixes:
        assert any(name.startswith(prefix) for name in trainable), f"no parameter under {prefix}"
    num_new = len(separate_names) - len(shared_names)
    assert num_new > 0
    print(f"[ok] warm-up filter: {len(trainable)} critic tensors trainable ({num_new} new), {len(frozen)} frozen")


def test_separate_mode_refuses_the_acting_time_value_api() -> None:
    """``evaluate`` / ``value_from_hidden`` must raise rather than silently return an actor-trunk value."""
    policy = _make_policy(critic_trunk="separate")
    obs = TensorDict({"policy": torch.zeros(NUM_ENVS, OBS_DIM)}, batch_size=[NUM_ENVS], device=DEVICE)
    policy.act(obs)
    for call, label in ((lambda: policy.evaluate(obs), "evaluate"),
                        (lambda: policy.value_from_hidden(torch.zeros(2, policy.d_model)), "value_from_hidden")):
        try:
            call()
        except RuntimeError:
            continue
        raise AssertionError(f"{label}() returned an actor-trunk value in separate mode")
    # evaluate_sequence is rerouted rather than blocked (its signature is forward_sequence_critic's).
    values = policy.evaluate_sequence(
        torch.zeros(2, 1, OBS_DIM), torch.zeros(2, 1, ACTION_DIM), torch.zeros(2, 1, 1), torch.zeros(2, 1, 1),
        memory=policy.initial_memory(1)
    )
    assert values.shape == (2, 1, 1)
    print("[ok] separate mode blocks evaluate()/value_from_hidden() and reroutes evaluate_sequence()")


def test_value_sweep_matches_per_step_reference() -> None:
    """The batched sweep must reproduce a naive per-step critic forward, on every live row and the bootstrap."""
    policy = _make_policy(seed=1)
    ppo = _make_ppo(policy)
    last_obs = _collect(ppo, seed=3)
    storage = ppo.storage
    carry = storage.carry_steps

    # Collection stored placeholders only; the values must be exactly zero before the sweep.
    assert float(storage.values.abs().max()) == 0.0, "the collection path evaluated the critic per step"

    ppo.compute_returns(last_obs)

    spans = _episode_spans(storage)
    max_error, num_checked = 0.0, 0
    for env in range(NUM_ENVS):
        memories = _reference_memories(policy, storage, spans[env], env)
        for (start, length), memory in zip(spans[env], memories):
            tokens = _tokens(storage, start, length, env)
            for step in range(length):
                reference = _reference_value(policy, tokens, memory, step)
                actual = storage.values[start + step, env]
                max_error = max(max_error, float((reference - actual).abs().max()))
                num_checked += 1
    assert num_checked == NUM_ENVS * NUM_STEPS, f"only {num_checked} of {NUM_ENVS * NUM_STEPS} rows swept"
    assert max_error < 1e-5, f"the swept values drift from the per-step reference by {max_error:.3e}"
    assert float(storage.values[carry:].abs().sum()) > 0.0, "the sweep wrote only zeros; the test is vacuous"
    # Rows in front of the window are never read; make sure the sweep did not write there either.
    assert float(storage.values[:carry].abs().max()) == 0.0

    # -- the GAE bootstrap, all three branches --
    bootstrap = storage.compute_critic_values(policy, chunk_size=None, last_obs=last_obs)
    obs = last_obs["policy"]
    for env in range(NUM_ENVS):
        start, length = spans[env][-1]
        memory = _reference_memories(policy, storage, spans[env], env)[-1]
        if not bool(storage.dones[start + length - 1, env]):
            # Still running: the pending token continues the episode.
            tokens = list(_tokens(storage, start, length, env))
            tokens[0] = torch.cat([tokens[0], obs[env].view(1, OBS_DIM)])
            tokens[1] = torch.cat([tokens[1], storage.actions[start + length - 1, env].view(1, ACTION_DIM)])
            tokens[2] = torch.cat([tokens[2], storage.raw_rewards[start + length - 1, env].view(1, 1)])
            tokens[3] = torch.cat([tokens[3], storage.dones[start + length - 1, env].float().view(1, 1)])
            reference = _reference_value(policy, tuple(tokens), memory, length)
        else:
            # The episode ended: the pending token is the first token of the next one, under the memory the
            # acting path would have moved on to.
            if bool(storage.trial_dones[start + length - 1, env]):
                next_memory = policy.z_init.detach().unsqueeze(0)
            else:
                pieces = _tokens(storage, start, length, env)
                terminal = _terminal_row(storage, start, length, env)
                full = [torch.cat([piece, extra]).unsqueeze(1) for piece, extra in zip(pieces, terminal)]
                with torch.no_grad():
                    hidden = policy.forward_sequence(full[0], full[1], full[2], full[3], memory=memory)
                    next_memory, _ = policy.write_memory(memory, hidden.transpose(0, 1))
            zeros = lambda width: torch.zeros(1, 1, width, device=DEVICE)  # noqa: E731
            with torch.no_grad():
                reference = policy.forward_sequence_critic(
                    obs[env].view(1, 1, OBS_DIM), zeros(ACTION_DIM), zeros(1), zeros(1), memory=next_memory
                )[0, 0]
        error = float((reference - bootstrap[env]).abs().max())
        assert error < 1e-5, f"env {env}: bootstrap value off by {error:.3e}"
    print(
        f"[ok] value sweep: {num_checked} rows match the per-step reference (max {max_error:.2e});"
        " bootstrap exact on all three boundary branches"
    )


def test_carried_rows_are_swept_too() -> None:
    """The rows of a trial carried across the rollout boundary must get values from the NEW sweep.

    GAE runs over the whole buffer and the advantage normalization reduces over every live row, so leaving the
    carried (still-open) trial at the collection-time placeholder zeros would both mis-scale every trained
    advantage and hand the trial garbage returns once it completes.
    """
    policy = _make_policy(seed=19)
    ppo = _make_ppo(policy)
    ppo.compute_returns(_collect(ppo, seed=23))
    ppo.update()  # rolls each environment's open trial into the carry region
    assert int(ppo.storage.carry_len.max().item()) > 0, "nothing was carried; the test would be vacuous"

    last_obs = _collect(ppo, seed=29)
    storage = ppo.storage
    carried = storage.carry_len.clone()
    # The carried rows still hold the values of the previous sweep; the new one must rewrite all of them.
    ppo.compute_returns(last_obs)

    spans = _episode_spans(storage)
    max_error, num_checked, carried_checked = 0.0, 0, 0
    for env in range(NUM_ENVS):
        window = storage.carry_steps - int(carried[env].item())
        memories = _reference_memories(policy, storage, spans[env], env)
        for (start, length), memory in zip(spans[env], memories):
            tokens = _tokens(storage, start, length, env)
            for step in range(length):
                reference = _reference_value(policy, tokens, memory, step)
                max_error = max(max_error, float((reference - storage.values[start + step, env]).abs().max()))
                num_checked += 1
                carried_checked += int(start + step < storage.carry_steps)
    assert num_checked == NUM_ENVS * NUM_STEPS + int(carried.sum().item())
    assert carried_checked == int(carried.sum().item()) and carried_checked > 0
    assert max_error < 1e-5, f"values drift by {max_error:.3e} on the second rollout"
    print(
        f"[ok] second rollout: {num_checked} rows swept ({carried_checked} of them carried in),"
        f" max drift {max_error:.2e}"
    )


def test_timeout_bootstrap_reapplication_is_bit_exact() -> None:
    """Re-applying the bootstrap after the sweep must equal the collection-time formula, bitwise."""
    policy = _make_policy(seed=5)
    ppo = _make_ppo(policy)
    storage = ppo.storage
    generator = torch.Generator(device=DEVICE).manual_seed(9)
    shape = (storage.num_transitions_per_env, NUM_ENVS, 1)
    storage.raw_rewards.copy_(torch.randn(*shape, generator=generator))
    storage.values.copy_(10.0 * torch.randn(*shape, generator=generator))
    storage.trial_dones.copy_((torch.rand(*shape, generator=generator) < 0.3).byte())
    ppo._time_out_flags.copy_((torch.rand(*shape, generator=generator) < 0.5).float())
    storage.rewards.zero_()

    # The reference: the exact expression PPO.process_env_step runs at collection time, row by row.
    expected = torch.zeros_like(storage.rewards)
    for row in range(storage.num_transitions_per_env):
        rewards = storage.raw_rewards[row].view(-1).clone()
        time_outs = ppo._time_out_flags[row].view(-1, 1)
        trial_ends = storage.trial_dones[row].view(-1, 1).float()
        rewards += ppo.gamma * torch.squeeze(storage.values[row] * time_outs * trial_ends, 1)
        expected[row] = rewards.view(-1, 1)

    ppo._reapply_timeout_bootstrap()
    assert torch.equal(storage.rewards, expected), (
        f"max deviation {float((storage.rewards - expected).abs().max()):.3e} (must be exactly zero)"
    )
    bootstrapped = int(((ppo._time_out_flags > 0) & (storage.trial_dones > 0)).sum().item())
    assert bootstrapped > 0, "no row was bootstrapped; the test is vacuous"
    # Idempotent: a second application must not add the bootstrap twice.
    ppo._reapply_timeout_bootstrap()
    assert torch.equal(storage.rewards, expected), "re-applying the bootstrap is not idempotent"
    print(f"[ok] timeout bootstrap: {bootstrapped} rows re-bootstrapped bit-exactly, and idempotently")


def test_value_loss_does_not_touch_the_actor() -> None:
    """A value loss must leave every actor-side parameter (trunk, writer, z_init, heads, noise) untouched."""
    policy = _make_policy(seed=7)
    ppo = _make_ppo(policy)
    last_obs = _collect(ppo, seed=11)
    ppo.compute_returns(last_obs)

    batch = next(ppo.storage.trial_pair_mini_batch_generator(policy, num_mini_batches=1, num_epochs=1))
    target = batch["target"]
    hidden, memory = ppo.trial_pair_forward(batch)
    values = policy.forward_sequence_critic(
        target["obs"], target["prev_actions"], target["prev_rewards"], target["prev_dones"],
        memory=memory, mask=target["mask"],
    )
    policy.zero_grad(set_to_none=True)
    values[target["loss_mask"]].pow(2).mean().backward()

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
    assert any(name == "critic.0.weight" for name in trained), "the value head received no gradient"

    # ... and conversely the policy loss still reaches the actor trunk and the writer (nothing was rerouted).
    policy.zero_grad(set_to_none=True)
    hidden, _ = ppo.trial_pair_forward(batch)
    hidden.square().sum().backward()
    assert float(policy.writer.attn.q_proj.weight.grad.abs().sum()) > 0.0, "the actor path lost the writer"
    assert float(policy.blocks[0].attn.q_proj.weight.grad.abs().sum()) > 0.0, "the actor trunk got no gradient"
    assert all(
        parameter.grad is None or float(parameter.grad.abs().sum()) == 0.0
        for name, parameter in policy.named_parameters()
        if name.startswith("critic")
    ), "the actor loss reached the critic pathway"
    print(f"[ok] gradient separation: {len(trained)} critic tensors trained, 0 actor tensors touched")


def test_ppo_update_runs_end_to_end() -> None:
    """A full separate-critic update must run, stay finite and keep the epoch-0 reconstruction canary at 1."""
    policy = _make_policy(seed=13)
    ppo = _make_ppo(policy, num_mini_batches=2)
    last_obs = _collect(ppo, seed=17)
    ppo.compute_returns(last_obs)
    swept_values = ppo.storage.values[ppo.storage.carry_steps :].clone()
    carried = int(ppo.storage.carry_len.max().item()) if ppo.storage.carry_steps else 0
    loss_dict = ppo.update()

    assert abs(loss_dict["ratio_mean"] - 1.0) < 1e-4, f"ratio_mean = {loss_dict['ratio_mean']}"
    assert loss_dict["ratio_max_dev_lag0_first_mb"] < 1e-3, (
        f"the reconstruction canary broke: {loss_dict['ratio_max_dev_lag0_first_mb']}"
    )
    for key in ("value_function", "surrogate", "entropy"):
        assert torch.isfinite(torch.tensor(loss_dict[key])), f"{key} is not finite: {loss_dict[key]}"
    assert loss_dict["value_function"] > 0.0
    # update() only rolls the open trial into the carry region; it never recomputes a value.
    if carried:
        storage = ppo.storage
        kept = storage.values[storage.carry_steps - carried : storage.carry_steps]
        assert torch.equal(kept, swept_values[NUM_STEPS - carried :]), "the carried values were not preserved"
    print(
        f"[ok] end-to-end update: value_loss {loss_dict['value_function']:.3f},"
        f" canary {loss_dict['ratio_max_dev_lag0_first_mb']:.2e}, {int(loss_dict['pool_pairs'])} pairs"
    )


if __name__ == "__main__":
    test_shared_mode_is_unchanged()
    test_warmup_freeze_filter_covers_the_critic_pathway()
    test_separate_mode_refuses_the_acting_time_value_api()
    test_value_sweep_matches_per_step_reference()
    test_carried_rows_are_swept_too()
    test_timeout_bootstrap_reapplication_is_bit_exact()
    test_value_loss_does_not_touch_the_actor()
    test_ppo_update_runs_end_to_end()
    print("all separate-critic-trunk tests passed")
