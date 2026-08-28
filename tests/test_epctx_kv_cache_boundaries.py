# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""KV-cache rollout path vs. no-cache batched path across STAGGERED episode and trial boundaries.

Self-contained: pure torch, no Isaac Sim. Run with::

    source /home/yandabao/miniforge3/etc/profile.d/conda.sh && conda activate patlab
    python -m pytest tests/test_epctx_kv_cache_boundaries.py -q

Setting: 4 environments whose episodes end at different steps (and whose lengths change from episode to
episode), ``K = 3`` sub-episodes per trial, ``M = 2`` memory tokens, both ``L >= T`` and a sliding ``L = 5``.
The acting path is driven exactly as :class:`~rsl_rl.algorithms.EpisodeContextPPO.process_env_step` drives it
(``forward_step`` -> ``get_episode_hidden`` -> ``reset(dones, trial_dones)``); the reference is a from-scratch
per-environment, per-episode :meth:`forward_sequence` with the memory chain rebuilt by hand through the batched
writer (``z_init`` at a trial start, ``G(H_prev)`` otherwise) -- what the PPO update reconstructs from storage.
Two negative controls (skip the cache flush / skip the trial flush at ONE boundary) prove the comparison is
sensitive to a flush bug, and a float32 run through the real storage + generator covers ``K = 3`` end to end.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import EpisodeContextPPO
from rsl_rl.modules import ActorCriticEpisodeContext

DEVICE = "cpu"
OBS_DIM = 5
CRITIC_OBS_DIM = 7
ACTION_DIM = 4
NUM_ENVS = 4
T_EPISODE = 12  # max_episode_length: every episode below is shorter
MEMORY_TOKENS = 2
EPISODES_PER_TRIAL = 3
NUM_STEPS = 60
# Per-environment episode lengths (cycled). Different per env AND different from one episode to the next.
EPISODE_LENGTHS = (
    (6, 9, 7, 11, 8, 5, 10, 4),
    (9, 7, 11, 6, 12, 4, 8, 3),
    (7, 11, 6, 9, 5, 12, 6, 4),
    (11, 6, 9, 7, 10, 3, 12, 2),
)
CONTEXTS = (("L = T (full episode)", T_EPISODE), ("sliding L = 5", 5))
TOL64 = 1e-9


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _sample_obs(generator: torch.Generator | None = None, dtype: torch.dtype = torch.float32) -> TensorDict:
    if generator is None:
        policy_obs = torch.zeros(NUM_ENVS, OBS_DIM, device=DEVICE, dtype=dtype)
        critic_obs = torch.zeros(NUM_ENVS, CRITIC_OBS_DIM, device=DEVICE, dtype=dtype)
    else:
        policy_obs = torch.randn(NUM_ENVS, OBS_DIM, generator=generator, device=DEVICE, dtype=dtype)
        critic_obs = torch.randn(NUM_ENVS, CRITIC_OBS_DIM, generator=generator, device=DEVICE, dtype=dtype)
    return TensorDict({"policy": policy_obs, "critic": critic_obs}, batch_size=[NUM_ENVS], device=DEVICE)


def _make_policy(
    context_length: int, dtype: torch.dtype = torch.float64, seed: int = 3
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
        memory_tokens=MEMORY_TOKENS,
        episodes_per_trial=EPISODES_PER_TRIAL,
    )
    # Off the initialization: the trunk is near-identity and the writer's delta is exactly zero at init, so an
    # unperturbed policy would make both the context and the memory comparison vacuous.
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if name in ("std", "log_std"):
                continue
            parameter.add_(0.1 * torch.randn_like(parameter))
    return policy.to(dtype).eval()


def _schedule(num_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
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
            trial_dones[step, env] = closed % EPISODES_PER_TRIAL == 0
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


def _reference_means(
    policy: ActorCriticEpisodeContext, obs: torch.Tensor, dones: torch.Tensor, trial_dones: torch.Tensor
) -> torch.Tensor:
    """Action means ``[S, N, A]`` from the NO-CACHE path, episode by episode, memory chain rebuilt by hand.

    Every episode is one :meth:`forward_sequence` over its own frames (positions ``0 .. len-1``, start flag on
    row 0) with the memory ``Z`` its trial has accumulated: ``z_init`` for the first episode of a trial, else
    ``G(H_prev)`` where ``H_prev = [memory_readout(Z_prev) | h_0 .. h_{len-1}]`` zero-padded to ``M + T`` with a
    validity mask -- exactly the ``H`` the storage snapshots and :meth:`memory_from_prefix` writes from.
    """
    num_steps, num_envs = dones.shape
    num_memory, span = policy.num_memory_tokens, policy.hidden_history_span
    means = torch.zeros(num_steps, num_envs, ACTION_DIM, dtype=obs.dtype, device=DEVICE)
    with torch.no_grad():
        z_init = policy.z_init.detach().unsqueeze(0)  # [1, M, d]
        for env in range(num_envs):
            memory = z_init
            for start, end, trial_end in _episodes(dones, trial_dones, env):
                hidden = policy.forward_sequence(obs[start : end + 1, env : env + 1], memory=memory)  # [S_e, 1, d]
                means[start : end + 1, env] = policy.actor(hidden.squeeze(1))
                length = end + 1 - start
                episode_hidden = torch.zeros(1, span, policy.d_model, dtype=obs.dtype, device=DEVICE)
                valid = torch.zeros(1, span, dtype=torch.bool, device=DEVICE)
                episode_hidden[0, :num_memory] = policy.memory_readout(memory)[0]
                episode_hidden[0, num_memory : num_memory + length] = hidden.squeeze(1)
                valid[0, : num_memory + length] = True
                memory = z_init if trial_end else policy.write_memory(episode_hidden, mask=valid)[0]
    return means


def _roll(
    policy: ActorCriticEpisodeContext,
    obs: torch.Tensor,
    dones: torch.Tensor,
    trial_dones: torch.Tensor,
    tamper: tuple[str, int, int] | None = None,
) -> torch.Tensor:
    """Drive the acting path exactly as ``EpisodeContextPPO.process_env_step`` does; return the means ``[S, N, A]``.

    ``tamper = (kind, env, step)`` sabotages ONE boundary: ``"cache"`` skips the KV-cache flush of ``env`` at
    ``step`` (its slots stay reachable and its step counter keeps counting, everything else -- the memory write,
    the H clear, the prefill -- still happens); ``"trial"`` skips the trial flush (``Z`` is carried over instead
    of restored to ``z_init``). With no tamper the reset invariants are asserted at every boundary.
    """
    num_steps, num_envs = dones.shape
    num_memory = policy.num_memory_tokens
    policy.initialize_state(num_envs, DEVICE, dtype=obs.dtype)
    z_init = policy.z_init.detach()
    means = []
    smallest_write = float("inf")
    with torch.no_grad():
        for step in range(num_steps):
            hidden = policy.forward_step(obs[step], commit=True)  # ppo.act -> policy.act
            means.append(policy.actor(hidden))
            done, trial = dones[step], trial_dones[step].clone()
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                policy.get_episode_hidden(done_ids)  # what the storage snapshots (a read; must not disturb)
            index_before = policy.episode_index_in_trial.clone()
            positions_before = policy._positions.clone()
            cache_positions_before = policy._cache_positions.clone()

            if tamper is not None and tamper[2] == step and tamper[0] == "trial":
                trial[tamper[1]] = False
            policy.reset(done, trial_dones=trial)
            if tamper is not None and tamper[2] == step and tamper[0] == "cache":
                env = tamper[1]
                policy._positions[env] = positions_before[env]
                policy._cache_positions[env] = cache_positions_before[env]
            if tamper is not None:
                continue

            # -- reset invariants (clean run only) --
            trial_ids = trial.nonzero(as_tuple=False).squeeze(-1)
            live = ~done
            assert (policy._cache_positions[done_ids] == -1).all(), f"step {step}: cache not flushed for {done_ids}"
            assert (policy._positions[done_ids] == 0).all(), f"step {step}: positions not zeroed for {done_ids}"
            assert not policy._history_valid[done_ids, num_memory:].any(), f"step {step}: H rows not cleared"
            assert policy._history_valid[done_ids, :num_memory].all(), f"step {step}: memory rows not prefilled"
            assert torch.equal(policy._positions[live], positions_before[live]), "a live env's step counter moved"
            assert torch.equal(policy._cache_positions[live], cache_positions_before[live]), "a live env's cache moved"
            if trial_ids.numel() > 0:
                expected = z_init.unsqueeze(0).expand(trial_ids.numel(), -1, -1)
                assert torch.equal(policy.memory[trial_ids], expected), f"step {step}: trial flush left Z != z_init"
                assert (policy.episode_index_in_trial[trial_ids] == 0).all()
            written_ids = done_ids[~trial[done_ids]]
            if written_ids.numel() > 0:
                assert torch.equal(policy.episode_index_in_trial[written_ids], index_before[written_ids] + 1)
                moved = (policy.memory[written_ids] - z_init).abs().amax(dim=(1, 2))
                smallest_write = min(smallest_write, moved.min().item())
    if tamper is None:
        assert smallest_write > 1e-6, f"a memory write left Z at z_init ({smallest_write:.3e}); memory is vacuous"
        print(f"    smallest |Z - z_init| after a write: {smallest_write:.3e}")
    return torch.stack(means)


def _first_rows_after(flags: torch.Tensor) -> torch.Tensor:
    """Boolean ``[S, N]``: the row right after a ``True`` in ``flags`` (the first step of the next episode)."""
    after = torch.zeros_like(flags)
    after[1:] = flags[:-1]
    return after


def _report(label: str, delta: torch.Tensor, dones: torch.Tensor, trial_dones: torch.Tensor) -> None:
    after_episode = _first_rows_after(dones)
    after_trial = _first_rows_after(trial_dones)
    per_env = ", ".join(f"env{e}={delta[:, e].max().item():.2e}" for e in range(delta.shape[1]))
    print(
        f"[{label}] max |d mean| overall {delta.max().item():.3e}"
        f" | first step after an episode reset {delta[after_episode].max().item():.3e}"
        f" ({int(after_episode.sum())} rows)"
        f" | first step after a trial flush {delta[after_trial].max().item():.3e}"
        f" ({int(after_trial.sum())} rows) | {per_env}"
    )


# --------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------


def test_kv_cache_matches_no_cache_at_staggered_boundaries() -> None:
    """Rollout path (KV cache + resets + writes) == from-scratch batched path, every env, every step, K = 3."""
    dones, trial_dones = _schedule(NUM_STEPS)
    assert (dones.sum(dim=0) >= 2 * EPISODES_PER_TRIAL).all(), "every env needs at least two full trials"
    generator = torch.Generator(device=DEVICE).manual_seed(101)
    obs = torch.randn(NUM_STEPS, NUM_ENVS, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)

    for label, context_length in CONTEXTS:
        policy = _make_policy(context_length)
        acting = _roll(policy, obs, dones, trial_dones)
        reference = _reference_means(policy, obs, dones, trial_dones)
        delta = (acting - reference).abs().amax(dim=-1)  # [S, N]
        _report(label, delta, dones, trial_dones)
        assert delta.max().item() < TOL64, f"KV cache vs no-cache ({label}): {delta.max().item():.3e}"
        assert acting.abs().max().item() > 1e-3, "the comparison is vacuous"
        # The memory is load-bearing: cutting the chain (every episode from z_init) must change the reference,
        # and so must the context (a context-free trunk would pass any cache test).
        cut = _reference_means(policy, obs, dones, torch.ones_like(trial_dones))
        assert (cut - reference).abs().max().item() > 1e-4, "cutting the memory chain changed nothing"


def test_skipping_the_cache_flush_is_detected() -> None:
    """Negative control: leave ONE env's KV cache un-flushed at ONE episode end -> its deltas blow up."""
    dones, trial_dones = _schedule(NUM_STEPS)
    generator = torch.Generator(device=DEVICE).manual_seed(101)
    obs = torch.randn(NUM_STEPS, NUM_ENVS, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)
    env = 1
    step = int(dones[:, env].nonzero()[0])  # env 1's first done (step 8)
    others = [e for e in range(NUM_ENVS) if e != env]

    for label, context_length in CONTEXTS:
        policy = _make_policy(context_length)
        reference = _reference_means(policy, obs, dones, trial_dones)
        acting = _roll(policy, obs, dones, trial_dones, tamper=("cache", env, step))
        delta = (acting - reference).abs().amax(dim=-1)
        before = delta[: step + 1].max().item()
        after = delta[step + 1 :, env].max().item()
        first = delta[step + 1, env].item()
        untouched = delta[:, others].max().item()
        print(
            f"[negative control, cache flush skipped, {label}] env {env} step {step}: before {before:.2e},"
            f" first step after {first:.2e}, max after {after:.2e}; other envs {untouched:.2e}"
        )
        assert before < TOL64, "the tamper leaked backwards in time"
        assert first > 1e-3 and after > 1e-3, f"a skipped cache flush went undetected ({label})"
        assert untouched < TOL64, "a skipped flush on one env disturbed another"


def test_skipping_the_trial_flush_is_detected() -> None:
    """Negative control: carry ``Z`` across ONE trial end instead of restoring ``z_init`` -> deltas blow up."""
    dones, trial_dones = _schedule(NUM_STEPS)
    generator = torch.Generator(device=DEVICE).manual_seed(101)
    obs = torch.randn(NUM_STEPS, NUM_ENVS, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)
    env = 2
    step = int(trial_dones[:, env].nonzero()[0])  # env 2's first trial end (step 23)
    others = [e for e in range(NUM_ENVS) if e != env]

    for label, context_length in CONTEXTS:
        policy = _make_policy(context_length)
        reference = _reference_means(policy, obs, dones, trial_dones)
        acting = _roll(policy, obs, dones, trial_dones, tamper=("trial", env, step))
        delta = (acting - reference).abs().amax(dim=-1)
        before = delta[: step + 1].max().item()
        after = delta[step + 1 :, env].max().item()
        first = delta[step + 1, env].item()
        untouched = delta[:, others].max().item()
        print(
            f"[negative control, trial flush skipped, {label}] env {env} step {step}: before {before:.2e},"
            f" first step after {first:.2e}, max after {after:.2e}; other envs {untouched:.2e}"
        )
        assert before < TOL64, "the tamper leaked backwards in time"
        assert first > 1e-5 and after > 1e-5, f"a skipped trial flush went undetected ({label})"
        assert untouched < TOL64, "a skipped flush on one env disturbed another"


def test_storage_reconstruction_matches_acting_k3() -> None:
    """The same statement through the REAL storage + generator (float32, K = 3, windows straddling boundaries).

    The pipeline canary in ``test_epctx_memory_pipeline.py`` covers ``K = 2`` only; with ``K = 3`` a segment
    of episode 2 reads ``G(H_1)`` where ``H_1`` itself ran under ``Z_1 = G(H_0)`` -- exact as long as the
    parameters are the acting ones, which ``learning_rate=0`` guarantees.
    """
    steps_per_env, num_rollouts = 8, 8
    dones, trial_dones = _schedule(steps_per_env * num_rollouts)
    for label, context_length in CONTEXTS:
        policy = _make_policy(context_length, dtype=torch.float32, seed=7)
        ppo = EpisodeContextPPO(
            policy,
            num_learning_epochs=1,
            num_mini_batches=1,
            learning_rate=0.0,
            schedule="fixed",
            desired_kl=None,
            device=DEVICE,
        )
        ppo.init_storage("rl", NUM_ENVS, steps_per_env, _sample_obs(), [ACTION_DIM])
        generator = torch.Generator(device=DEVICE).manual_seed(5)
        obs = _sample_obs(generator)
        acting: list[torch.Tensor] = []
        max_error, saw_third_episode = 0.0, False
        for rollout in range(num_rollouts):
            with torch.no_grad():
                for k in range(steps_per_env):
                    step = rollout * steps_per_env + k
                    ppo.act(obs)
                    acting.append(policy.action_mean.clone())
                    next_obs = _sample_obs(generator)
                    rewards = torch.randn(NUM_ENVS, generator=generator, device=DEVICE)
                    ppo.process_env_step(next_obs, rewards, dones[step], {"trial_done": trial_dones[step]})
                    obs = next_obs
                ppo.compute_returns(obs)
                saw_third_episode |= bool((ppo.storage.row_episode_index == EPISODES_PER_TRIAL - 1).any())
                batch = next(ppo.storage.recurrent_mini_batch_generator(num_mini_batches=1, num_epochs=1))
                policy.act(batch[0], masks=batch[9], hidden_state=batch[8][0])
                window = torch.stack(acting[rollout * steps_per_env :])
                max_error = max(max_error, (policy.action_mean - window).abs().max().item())
            loss_dict = ppo.update()  # consumes the generator and clear()s the storage (commits the sources)
            assert abs(loss_dict["ratio_mean"] - 1.0) < 1e-5, f"ratio_mean = {loss_dict['ratio_mean']}"
        assert saw_third_episode, "no row was acted in episode 2 of a trial; K = 3 was not exercised"
        print(f"[storage, {label}] max |d mean| acting vs generator window over {num_rollouts} rollouts: {max_error:.3e}")
        assert max_error < 1e-4, f"storage reconstruction vs acting ({label}): {max_error:.3e}"


if __name__ == "__main__":
    test_kv_cache_matches_no_cache_at_staggered_boundaries()
    test_skipping_the_cache_flush_is_detected()
    test_skipping_the_trial_flush_is_detected()
    test_storage_reconstruction_matches_acting_k3()
    print("all KV-cache boundary tests passed")
