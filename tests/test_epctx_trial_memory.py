# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OPTIONAL cross-episode memory of the episode-context policy (``memory_tokens > 0``).

Self-contained: pure torch, no Isaac Sim / SimulationApp needed. Run with::

    source /home/yandabao/miniforge3/etc/profile.d/conda.sh && conda activate patlab
    python -m pytest tests/test_epctx_trial_memory.py -q

The memory is ``M`` rows PREPENDED to the trunk's own sequence (no read layer, no gate), written at every
episode end by a writer that attends the memory rows over the whole finished pass. The load-bearing tests are

* ``test_memory_off_builds_nothing``: ``memory_tokens=0`` (the default) adds no parameter and no behavior --
  the whole feature has to be invisible until it is asked for.
* ``test_step_matches_window_with_memory``: the acting path (memory prefill + per-step KV cache) and the batched
  path (memory rows really in the sequence) agree to floating-point noise. That is what makes the PPO
  reconstruction canary possible at all.
* ``test_grad_flow_through_the_writer``: a loss on the rows of the *second* segment of a window trains the
  writer, ``memory_pos_embed``, the trunk THROUGH the memory rows and ``z_init`` (the writer's anchor), while
  the cached ``H`` of the source episode stays out of the graph.
* ``test_written_memory_is_z_init_at_init``: the write is ``Z_next = z_init + delta`` with ``delta`` exactly
  zero at initialization, so a fresh policy's episode 2 is bit-for-bit its episode 1.

There is deliberately NO identity-at-init test for the TRUNK: prepending rows to the sequence perturbs a loaded
BC trunk from the first step by construction (the user's decision). ``z_init``/``memory_pos_embed`` keep the
small ``0.02`` init so the perturbation is small; how small is a closed-loop question, not a unit test. What is
exact at init is the write, above.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

import pytest

from rsl_rl.modules import ActorCriticEpisodeContext

DEVICE = "cpu"
OBS_DIM = 5
ACTION_DIM = 4
NUM_ENVS = 3
T_EPISODE = 6  # small T so the tests stay fast; the shapes are the same at T = 80
MEMORY_TOKENS = 8

# Every state_dict key the memory adds. Nothing else may appear when the feature is switched on, and none of it
# may exist when it is off.
MEMORY_KEY_PREFIXES = ("z_init", "memory_pos_embed", "writer.")


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _make_policy(memory_tokens: int = 0, seed: int = 0, perturb: bool = True) -> ActorCriticEpisodeContext:
    """A small float64 policy. Double precision makes "tight tolerance" mean something."""
    torch.manual_seed(seed)
    sample_obs = TensorDict({"policy": torch.zeros(NUM_ENVS, OBS_DIM)}, batch_size=[NUM_ENVS], device=DEVICE)
    policy = ActorCriticEpisodeContext(
        obs=sample_obs,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=ACTION_DIM,
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
    if perturb:
        # Move the LayerNorms/heads off their initialization so the tests are not measuring a near-identity
        # network (and so that a memory row genuinely changes an output).
        with torch.no_grad():
            for name, parameter in policy.named_parameters():
                if name in ("std", "log_std"):
                    continue
                parameter.add_(0.1 * torch.randn_like(parameter))
    return policy.double().eval()


def _memory_twin(base: ActorCriticEpisodeContext, seed: int = 1) -> ActorCriticEpisodeContext:
    """A ``memory_tokens=8`` policy carrying ``base``'s trunk/actor/critic weights and fresh memory modules."""
    twin = _make_policy(memory_tokens=MEMORY_TOKENS, seed=seed)
    twin.load_state_dict(base.state_dict(), strict=False)
    return twin.double().eval()


def _episode_dones(num_steps: int) -> torch.Tensor:
    """A done on the last step of every ``T_EPISODE`` block, staggered so the environments desynchronize."""
    dones = torch.zeros(num_steps, NUM_ENVS, dtype=torch.bool, device=DEVICE)
    for env in range(NUM_ENVS):
        step = T_EPISODE - 1 - env
        while step < num_steps:
            dones[step, env] = True
            step += T_EPISODE
    return dones


def _roll(
    policy: ActorCriticEpisodeContext,
    obs: torch.Tensor,
    dones: torch.Tensor,
    trial_dones: torch.Tensor | None = None,
) -> torch.Tensor:
    """Step the acting path one frame at a time and collect ``h`` ``[S, N, d]``."""
    policy.initialize_state(obs.shape[1], DEVICE, dtype=obs.dtype)
    hidden = []
    for step in range(obs.shape[0]):
        hidden.append(policy.forward_step(obs[step], commit=True))
        policy.reset(dones[step], None if trial_dones is None else trial_dones[step])
    return torch.stack(hidden)


def _set_memory(policy: ActorCriticEpisodeContext, memory: torch.Tensor) -> None:
    """Install ``Z`` ``[N, M, d]`` on the acting path and prefill it (K/V + rows 0..M-1 of ``H``)."""
    with torch.no_grad():
        policy._memory = memory.clone()
        policy._hidden_history.zero_()
        policy._history_valid.zero_()
        policy._prefill_memory(torch.arange(memory.shape[0], device=memory.device))


# --------------------------------------------------------------------------------------------------
# A. Default off
# --------------------------------------------------------------------------------------------------


def test_memory_off_builds_nothing() -> None:
    """``memory_tokens=0`` must add no parameter, no buffer, no runtime state and no behavior."""
    policy = _make_policy(memory_tokens=0)
    keys = list(policy.state_dict().keys())
    offending = [key for key in keys if key.startswith(MEMORY_KEY_PREFIXES)]
    assert not offending, f"a memory-free policy grew parameters: {offending}"
    for attribute in ("z_init", "memory_pos_embed", "writer"):
        assert not hasattr(policy, attribute), f"a memory-free policy built {attribute}"
    assert policy.num_memory_tokens == 0
    assert policy.hidden_history_span == T_EPISODE

    policy.initialize_state(NUM_ENVS, DEVICE, dtype=torch.float64)
    assert policy.memory is None and policy.episode_index_in_trial is None
    assert policy._memory_key_cache is None and policy._memory_value_cache is None
    with pytest.raises(RuntimeError):
        policy.get_episode_hidden()
    with pytest.raises(RuntimeError):
        policy.initial_memory(NUM_ENVS)

    # The new arguments are rejected rather than silently ignored.
    generator = torch.Generator(device=DEVICE).manual_seed(3)
    obs = torch.randn(T_EPISODE, NUM_ENVS, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)
    with pytest.raises(AssertionError):
        policy.forward_window(obs.transpose(0, 1), memory=torch.zeros(NUM_ENVS, 1, 4, policy.d_model))
    with pytest.raises(AssertionError):
        policy.forward_sequence(obs, memory=torch.zeros(NUM_ENVS, 4, policy.d_model))

    # ... and ``trial_dones`` is inert: a memory-free policy has nothing that survives an episode anyway.
    dones = _episode_dones(2 * T_EPISODE)
    obs = torch.randn(2 * T_EPISODE, NUM_ENVS, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)
    with torch.no_grad():
        plain = _roll(policy, obs, dones)
        with_trials = _roll(policy, obs, dones, trial_dones=dones)
    assert torch.equal(plain, with_trials), "trial_dones changed a memory-free policy"
    print("[ok] memory_tokens=0: no parameters, no state, no behavior change")


def test_memory_only_adds_memory_parameters() -> None:
    """Switching the memory on keeps every memory-free parameter and adds only the memory's own."""
    base = _make_policy(memory_tokens=0, seed=4)
    memory_policy = _memory_twin(base, seed=5)
    shared = set(base.state_dict()) & set(memory_policy.state_dict())
    added = set(memory_policy.state_dict()) - set(base.state_dict())
    assert shared == set(base.state_dict()), "the memory policy dropped a memory-free parameter"
    assert all(key.startswith(MEMORY_KEY_PREFIXES) for key in added), f"unexpected new parameters: {added}"
    for key in shared:
        assert torch.equal(base.state_dict()[key], memory_policy.state_dict()[key])
    assert memory_policy.hidden_history_span == MEMORY_TOKENS + T_EPISODE
    print("[ok] the memory adds exactly z_init, memory_pos_embed and the writer")


# --------------------------------------------------------------------------------------------------
# B. The two forward paths agree, and the memory is load-bearing
# --------------------------------------------------------------------------------------------------


def test_step_matches_window_with_memory() -> None:
    """Acting path (prefill + KV cache) == batched path (memory rows in the sequence), with a non-trivial Z."""
    policy = _memory_twin(_make_policy(memory_tokens=0, seed=4), seed=5)
    generator = torch.Generator(device=DEVICE).manual_seed(13)
    obs = torch.randn(T_EPISODE, NUM_ENVS, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)

    policy.initialize_state(NUM_ENVS, DEVICE, dtype=torch.float64)
    memory = torch.randn(
        NUM_ENVS, MEMORY_TOKENS, policy.d_model, generator=generator, dtype=torch.float64, device=DEVICE
    )
    _set_memory(policy, memory)
    with torch.no_grad():
        incremental = torch.stack([policy.forward_step(obs[step], commit=True) for step in range(T_EPISODE)])
        batched = policy.forward_sequence(obs, memory=memory)
        windowed = policy.forward_window(obs.transpose(0, 1), memory=memory.unsqueeze(1)).transpose(0, 1)
    error = (incremental - batched).abs().max().item()
    assert error < 1e-6, f"forward_step vs forward_sequence with memory: {error:.3e}"
    assert (incremental - windowed).abs().max().item() < 1e-6, "forward_window disagrees with forward_step"

    # The acting path's H holds the batched pass's memory rows in slots 0..M-1 and its readouts after them.
    hidden, valid = policy.get_episode_hidden()
    assert valid.all(), "H is not fully populated after a whole episode"
    assert (hidden[:, :MEMORY_TOKENS] - policy.memory_readout(memory)).abs().max().item() < 1e-6
    assert (hidden[:, MEMORY_TOKENS:] - incremental.transpose(0, 1)).abs().max().item() < 1e-6
    print(f"[ok] step == sequence == window with a memory: max |dh| = {error:.3e}")


def test_the_memory_changes_the_policy() -> None:
    """``Z`` must actually reach the tokens -- there is no gate to keep it out."""
    policy = _memory_twin(_make_policy(memory_tokens=0, seed=4), seed=5)
    generator = torch.Generator(device=DEVICE).manual_seed(9)
    window = torch.randn(NUM_ENVS, T_EPISODE, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)
    other = torch.randn(
        NUM_ENVS, 1, MEMORY_TOKENS, policy.d_model, generator=generator, dtype=torch.float64, device=DEVICE
    )
    with torch.no_grad():
        with_z_init = policy.forward_window(window)
        with_other = policy.forward_window(window, memory=other)
    assert (with_z_init - with_other).abs().max().item() > 1e-6, "Z did not reach the token stream"
    print("[ok] the token stream reads Z")


def test_segments_are_isolated() -> None:
    """Two segments in one pass: a token only ever sees ITS segment's memory block."""
    policy = _memory_twin(_make_policy(memory_tokens=0, seed=4), seed=5)
    generator = torch.Generator(device=DEVICE).manual_seed(11)
    window = torch.randn(NUM_ENVS, T_EPISODE, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)
    split = T_EPISODE // 2
    segments = torch.zeros(NUM_ENVS, T_EPISODE, dtype=torch.long, device=DEVICE)
    segments[:, split:] = 1
    memory = torch.randn(
        NUM_ENVS, 2, MEMORY_TOKENS, policy.d_model, generator=generator, dtype=torch.float64, device=DEVICE
    )
    late = memory.clone()
    late[:, 1] = torch.randn_like(late[:, 1])
    early = memory.clone()
    early[:, 0] = torch.randn_like(early[:, 0])
    with torch.no_grad():
        both = policy.forward_window(window, segments, memory=memory, memory_segments=segments)
        changed_late = policy.forward_window(window, segments, memory=late, memory_segments=segments)
        changed_early = policy.forward_window(window, segments, memory=early, memory_segments=segments)
    assert torch.equal(both[:, :split], changed_late[:, :split]), "segment 1's memory leaked backwards"
    assert (both[:, split:] - changed_late[:, split:]).abs().max().item() > 1e-6
    assert torch.equal(both[:, split:], changed_early[:, split:]), "segment 0's memory leaked forwards"
    assert (both[:, :split] - changed_early[:, :split]).abs().max().item() > 1e-6

    # A segment's rows must also be exactly what a pass over that segment ALONE produces.
    with torch.no_grad():
        alone = policy.forward_window(window[:, split:], memory=memory[:, 1:2])
    assert (both[:, split:] - alone).abs().max().item() < 1e-6, "a segment is not independent of the other one"
    print("[ok] segments read only their own memory block, and behave as if they were alone")


# --------------------------------------------------------------------------------------------------
# B2. The write is anchored on z_init
# --------------------------------------------------------------------------------------------------


def test_written_memory_is_z_init_at_init() -> None:
    """A freshly built writer writes exactly ``z_init``: ``delta`` is zero, so episode 2 == episode 1."""
    policy = _make_policy(memory_tokens=MEMORY_TOKENS, seed=5, perturb=False)
    generator = torch.Generator(device=DEVICE).manual_seed(101)
    obs = torch.randn(T_EPISODE, NUM_ENVS, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)
    z_init = policy.z_init.detach()
    expanded = z_init.unsqueeze(0).expand(NUM_ENVS, -1, -1)

    policy.initialize_state(NUM_ENVS, DEVICE, dtype=torch.float64)
    with torch.no_grad():
        first = torch.stack([policy.forward_step(obs[step], commit=True) for step in range(T_EPISODE)])
        hidden, valid = policy.get_episode_hidden()
        written, _ = policy.write_memory(hidden, mask=valid)
    assert torch.equal(written, expanded), "the write is not bit-for-bit z_init at init"

    # The acting write feeds the prefill of the next pass, so the whole second episode of the trial has to
    # reproduce the first one frame for frame on identical observations.
    with torch.no_grad():
        dones = torch.ones(NUM_ENVS, dtype=torch.bool, device=DEVICE)
        policy.reset(dones, torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE))
        assert torch.equal(policy.memory, expanded), "the acting path's Z moved off z_init at init"
        second = torch.stack([policy.forward_step(obs[step], commit=True) for step in range(T_EPISODE)])
    error = (second - first).abs().max().item()
    assert error < 1e-12, f"episode 2 differs from episode 1 at init by {error:.3e}"
    print(f"[ok] at init the write is exactly z_init and episode 2 == episode 1: max |dh| = {error:.3e}")


def test_a_nonzero_delta_moves_the_memory_and_the_policy() -> None:
    """Nudge the FF's output layer off zero: ``Z_next`` leaves ``z_init`` and the logits it feeds move."""
    policy = _make_policy(memory_tokens=MEMORY_TOKENS, seed=5, perturb=False)
    generator = torch.Generator(device=DEVICE).manual_seed(103)
    hidden = torch.randn(
        NUM_ENVS, MEMORY_TOKENS + T_EPISODE, policy.d_model, generator=generator, dtype=torch.float64, device=DEVICE
    )
    window = torch.randn(NUM_ENVS, T_EPISODE, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)
    z_init = policy.z_init.detach()

    with torch.no_grad():
        anchored, _ = policy.write_memory(hidden)
        assert torch.equal(anchored, z_init.unsqueeze(0).expand(NUM_ENVS, -1, -1))
        before = policy.action_mean_from_hidden(policy.forward_window(window, memory=anchored.unsqueeze(1)))
        weight = policy.writer.ff[-1].weight
        weight.add_(0.5 * torch.randn(weight.shape, generator=generator, dtype=weight.dtype, device=DEVICE))
        moved, _ = policy.write_memory(hidden)
        after = policy.action_mean_from_hidden(policy.forward_window(window, memory=moved.unsqueeze(1)))
    assert (moved - z_init).abs().max().item() > 1e-6, "delta stayed zero after the nudge"
    assert (after - before).abs().max().item() > 1e-9, "the written memory never reached the actions"
    print("[ok] a nonzero delta writes a memory that is not z_init and that changes the policy")


# --------------------------------------------------------------------------------------------------
# C. Reset / trial bookkeeping
# --------------------------------------------------------------------------------------------------


def test_reset_writes_and_trial_reset_restores() -> None:
    """A done writes ``Z``, clears ``H`` and re-prefills; a trial done puts ``z_init`` back."""
    policy = _memory_twin(_make_policy(memory_tokens=0, seed=4), seed=5)
    generator = torch.Generator(device=DEVICE).manual_seed(17)
    obs = torch.randn(T_EPISODE, NUM_ENVS, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)
    policy.initialize_state(NUM_ENVS, DEVICE, dtype=torch.float64)
    z_init = policy.z_init.detach().double()

    with torch.no_grad():
        # initialize_state already prefilled z_init into the K/V cache and into rows 0..M-1 of H.
        assert policy._history_valid[:, :MEMORY_TOKENS].all(), "the memory rows of H were not prefilled"
        prefilled = policy.memory_readout(z_init.expand(NUM_ENVS, -1, -1))
        assert (policy._hidden_history[:, :MEMORY_TOKENS] - prefilled).abs().max().item() < 1e-9
        for step in range(T_EPISODE):
            policy.forward_step(obs[step], commit=True)
        hidden, valid = policy.get_episode_hidden()
        assert valid.all(), "H does not hold the memory rows plus the whole episode"
        assert not hidden.requires_grad, "the cached H must be detached"

        expected, _ = policy.write_memory(hidden.clone(), mask=valid.clone())

        # Env 0 finishes its episode; the others keep going. The trial does not end, so Z is kept.
        dones = torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE)
        dones[0] = True
        policy.reset(dones, torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE))

    assert (policy.memory[0] - z_init).abs().max().item() > 1e-6, "the done env's Z was not written"
    assert torch.allclose(policy.memory[0], expected[0]), "the acting write is not G(H)"
    assert torch.equal(policy.memory[1:], z_init.expand(NUM_ENVS - 1, -1, -1)), "an unfinished env's Z moved"
    assert not policy._history_valid[0, MEMORY_TOKENS:].any(), "the done env's episode rows were not cleared"
    assert policy._history_valid[0, :MEMORY_TOKENS].all(), "the done env's memory rows were not re-prefilled"
    assert policy._history_valid[1, MEMORY_TOKENS:][:T_EPISODE].all(), "an unfinished env's H was cleared"
    assert policy.episode_index_in_trial.tolist() == [1, 0, 0]
    assert policy._positions[0].item() == 0 and policy._positions[1].item() == T_EPISODE

    # The prefill followed the write: rows 0..M-1 of the fresh H are the readouts of the NEW Z ...
    written = policy.memory.clone()
    assert (policy._hidden_history[0, :MEMORY_TOKENS] - policy.memory_readout(written[:1])[0]).abs().max() < 1e-9
    # ... and so are the cached keys, so the next step attends to the new memory, not the stale one.
    with torch.no_grad():
        after_write = policy.forward_step(obs[0], commit=False)
        _set_memory(policy, z_init.expand(NUM_ENVS, -1, -1).contiguous())
        under_z_init = policy.forward_step(obs[0], commit=False)
    assert (after_write[0] - under_z_init[0]).abs().max().item() > 1e-6, "the step reused the stale Z"

    # A trial done restores z_init (and only for the environments named), and re-prefills them.
    with torch.no_grad():
        _set_memory(policy, written)
        trial_dones = torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE)
        trial_dones[0] = True
        policy.reset(torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE), trial_dones)
    assert torch.equal(policy.memory[0], z_init), "the trial boundary did not restore z_init"
    assert torch.equal(policy.memory[1], written[1]), "a trial-free environment's Z was reset"
    assert policy.episode_index_in_trial.tolist() == [0, 0, 0]
    assert (
        policy._hidden_history[0, :MEMORY_TOKENS] - policy.memory_readout(z_init.unsqueeze(0))[0]
    ).abs().max() < 1e-9
    print("[ok] done -> write + clear + prefill, trial done -> z_init + prefill")


def test_trial_dones_default_to_dones() -> None:
    """``reset(dones)`` alone must never let a memory survive an episode -- the safe fallback."""
    policy = _memory_twin(_make_policy(memory_tokens=0, seed=4), seed=5)
    generator = torch.Generator(device=DEVICE).manual_seed(19)
    num_steps = 2 * T_EPISODE
    obs = torch.randn(num_steps, NUM_ENVS, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)
    dones = _episode_dones(num_steps)
    z_init = policy.z_init.detach().double()

    with torch.no_grad():
        _roll(policy, obs, dones)  # trial_dones=None => every episode is its own trial
        fallback = policy.memory.clone()
        _roll(policy, obs, dones, trial_dones=torch.zeros_like(dones))  # memory persists across the episodes
        persisted = policy.memory.clone()
    assert torch.equal(fallback, z_init.expand(NUM_ENVS, -1, -1)), "memory survived a done without trial_dones"
    assert (persisted - z_init).abs().max().item() > 1e-6, "memory did not persist inside a trial"
    print("[ok] trial_dones=None behaves as trial_dones=dones")


# --------------------------------------------------------------------------------------------------
# D. Gradient flow
# --------------------------------------------------------------------------------------------------


def test_grad_flow_through_the_writer() -> None:
    """A loss on the second segment trains the writer, the memory embedding, the trunk and ``z_init``.

    Not the cached ``H``: the storage snapshots it detached and it must stay out of the graph.
    """
    policy = _memory_twin(_make_policy(memory_tokens=0, seed=4), seed=5)
    policy.zero_grad(set_to_none=True)
    generator = torch.Generator(device=DEVICE).manual_seed(23)
    num_steps = 2 * T_EPISODE
    window = torch.randn(NUM_ENVS, num_steps, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)
    segments = torch.zeros(NUM_ENVS, num_steps, dtype=torch.long, device=DEVICE)
    segments[:, T_EPISODE:] = 1

    # The cached H of the source episode: a leaf that the update path is contractually required to feed in
    # DETACHED (the storage snapshots it at collection time), so no gradient may ever reach it.
    source_hidden = torch.randn(
        NUM_ENVS,
        MEMORY_TOKENS + T_EPISODE,
        policy.d_model,
        generator=generator,
        dtype=torch.float64,
        device=DEVICE,
        requires_grad=True,
    )
    written, _ = policy.write_memory(source_hidden.detach())
    memory = torch.stack([policy.initial_memory(NUM_ENVS), written], dim=1)  # [B, 2, M, d]

    hidden = policy.forward_window(window, segments, memory=memory, memory_segments=segments)
    policy.action_mean_from_hidden(hidden[:, T_EPISODE:]).sum().backward(retain_graph=True)

    trained = {
        "memory_pos_embed": policy.memory_pos_embed,
        "writer.attn.q_proj.weight": policy.writer.attn.q_proj.weight,
        "writer.attn.v_proj.weight": policy.writer.attn.v_proj.weight,
        "writer.ff.0.weight": policy.writer.ff[0].weight,
        # The FF's output layer: its own weight is trained even from a zero start, because its INPUT is nonzero.
        "writer.ff.2.weight": policy.writer.ff[-1].weight,
        "writer.ff.2.bias": policy.writer.ff[-1].bias,
        # ``z_init`` is the writer's anchor, so a WRITTEN segment trains it too (not just source-free ones).
        "z_init": policy.z_init,
        # The trunk itself, which now processes the memory rows as ordinary tokens.
        "blocks.0.attn.k_proj.weight": policy.blocks[0].attn.k_proj.weight,
        "blocks.1.ff.0.weight": policy.blocks[1].ff[0].weight,
    }
    for name, parameter in trained.items():
        assert parameter.grad is not None, f"{name} got no gradient"
        assert parameter.grad.abs().max().item() > 0.0, f"{name}'s gradient is exactly zero"
    assert source_hidden.grad is None, "a gradient reached the cached source episode"

    # ... and a source-free segment reaches z_init directly, the way it always did.
    policy.zero_grad(set_to_none=True)
    policy.action_mean_from_hidden(hidden[:, :T_EPISODE]).sum().backward()
    assert policy.z_init.grad is not None and policy.z_init.grad.abs().max().item() > 0.0, (
        "a source-free segment did not train z_init"
    )
    print("[ok] the writer, memory_pos_embed, the trunk and z_init are trained through the memory rows; H stays out")


def test_gradient_reaches_the_memory_rows() -> None:
    """The trunk really backpropagates INTO the memory rows: a leaf ``Z`` gets a gradient."""
    policy = _memory_twin(_make_policy(memory_tokens=0, seed=4), seed=5)
    generator = torch.Generator(device=DEVICE).manual_seed(29)
    window = torch.randn(NUM_ENVS, T_EPISODE, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)
    memory = torch.randn(
        NUM_ENVS,
        1,
        MEMORY_TOKENS,
        policy.d_model,
        generator=generator,
        dtype=torch.float64,
        device=DEVICE,
        requires_grad=True,
    )
    policy.action_mean_from_hidden(policy.forward_window(window, memory=memory)).sum().backward()
    assert memory.grad is not None and memory.grad.abs().max().item() > 0.0, "no gradient reached Z"
    print("[ok] gradient flows through the trunk into the memory rows")


if __name__ == "__main__":
    test_memory_off_builds_nothing()
    test_memory_only_adds_memory_parameters()
    test_step_matches_window_with_memory()
    test_the_memory_changes_the_policy()
    test_written_memory_is_z_init_at_init()
    test_a_nonzero_delta_moves_the_memory_and_the_policy()
    test_segments_are_isolated()
    test_reset_writes_and_trial_reset_restores()
    test_trial_dones_default_to_dones()
    test_grad_flow_through_the_writer()
    test_gradient_reaches_the_memory_rows()
