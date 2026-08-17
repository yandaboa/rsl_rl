# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end tests for the cross-episode memory THROUGH collection, storage and the real PPO update.

Self-contained: pure torch, no Isaac Sim / SimulationApp needed. Run with::

    source /home/yandabao/miniforge3/etc/profile.d/conda.sh && conda activate patlab
    python -m pytest tests/test_epctx_memory_pipeline.py -q

``tests/test_epctx_trial_memory.py`` pins the model in isolation (the ``z_init`` anchor of the write, writer/read
semantics, reset bookkeeping). This file pins the plumbing that carries a memory from the environment step where
it is written to the minibatch where it is read:

* ``test_epoch_zero_canary_with_memory`` -- the load-bearing one. The acting path builds ``Z`` incrementally
  (``Z_1 = G(H_0)`` at the episode boundary, where ``H_0`` is the whole pass: the ``M`` memory rows plus the
  episode's readouts); the update path rebuilds it from the cached ``H`` the storage snapshotted. With unchanged
  parameters the two must produce the same log-probs, i.e. the PPO ratio is 1 at epoch 0 -- across episode
  boundaries AND trial boundaries inside a window.
* ``test_segment_sources_are_the_right_episodes`` -- with several episodes per window, each segment must read the
  episode that ended immediately before it, and the segment after a trial end must read ``z_init``.
* ``test_source_survives_clear`` -- a source written in rollout ``k`` is what rollout ``k+1``'s first segment
  reads (the persistent, clear()-surviving slot).
* ``test_memory_tokens_zero_is_the_stock_step`` -- with ``memory_tokens=0`` the ``process_env_step`` override is
  the stock one, transition for transition.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import EpisodeContextPPO
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules import ActorCriticEpisodeContext

DEVICE = "cpu"
OBS_DIM = 5
CRITIC_OBS_DIM = 7
ACTION_DIM = 4
NUM_ENVS = 6
T_EPISODE = 6
STEPS_PER_ENV = 6
MEMORY_TOKENS = 4
EPISODES_PER_TRIAL = 2
# Per-environment episode length. Env 0 ends three episodes inside one 6-step window (and one of them closes a
# trial), which is the case the sparse snapshot bookkeeping exists for.
EPISODE_LENGTHS = (2, 3, 4, 6, 3, 5)
GAMMA = 0.99
LAM = 0.95


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _sample_obs(num_envs: int, generator: torch.Generator | None = None) -> TensorDict:
    if generator is None:
        policy_obs = torch.zeros(num_envs, OBS_DIM, device=DEVICE)
        critic_obs = torch.zeros(num_envs, CRITIC_OBS_DIM, device=DEVICE)
    else:
        policy_obs = torch.randn(num_envs, OBS_DIM, generator=generator, device=DEVICE)
        critic_obs = torch.randn(num_envs, CRITIC_OBS_DIM, generator=generator, device=DEVICE)
    return TensorDict({"policy": policy_obs, "critic": critic_obs}, batch_size=[num_envs], device=DEVICE)


def _make_policy(
    memory_tokens: int = MEMORY_TOKENS,
    seed: int = 0,
    critic_design: str = "privileged",
) -> ActorCriticEpisodeContext:
    torch.manual_seed(seed)
    policy = ActorCriticEpisodeContext(
        obs=_sample_obs(NUM_ENVS),
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
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
        critic_design=critic_design,
        memory_tokens=memory_tokens,
        episodes_per_trial=EPISODES_PER_TRIAL,
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
        learning_rate=0.0,  # keep the parameters frozen so every minibatch sees the acting parameters
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


def _trial_schedule(num_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Dones and trial dones of ``num_steps`` steps: episodes of :data:`EPISODE_LENGTHS`, ``K = 2`` per trial."""
    dones = torch.zeros(num_steps, NUM_ENVS, dtype=torch.bool, device=DEVICE)
    trial_dones = torch.zeros(num_steps, NUM_ENVS, dtype=torch.bool, device=DEVICE)
    for env, length in enumerate(EPISODE_LENGTHS):
        step, closed = length - 1, 0
        while step < num_steps:
            dones[step, env] = True
            closed += 1
            trial_dones[step, env] = closed % EPISODES_PER_TRIAL == 0
            step += length
    return dones, trial_dones


def _episode_labels(dones: torch.Tensor, trial_dones: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Episode step and episode-index-in-trial of every step of the schedule, ``[S, N]`` each."""
    num_steps = dones.shape[0]
    positions = torch.zeros(num_steps, NUM_ENVS, dtype=torch.long, device=DEVICE)
    episode_index = torch.zeros(num_steps, NUM_ENVS, dtype=torch.long, device=DEVICE)
    for step in range(1, num_steps):
        done = dones[step - 1]
        positions[step] = torch.where(done, torch.zeros_like(positions[step - 1]), positions[step - 1] + 1)
        after_done = torch.where(done, episode_index[step - 1] + 1, episode_index[step - 1])
        episode_index[step] = torch.where(trial_dones[step - 1], torch.zeros_like(after_done), after_done)
    return positions, episode_index


def _collect(
    ppo: EpisodeContextPPO,
    dones: torch.Tensor,
    trial_dones: torch.Tensor,
    offset: int,
    generator: torch.Generator,
    sink: list | None = None,
) -> None:
    """One rollout through ``PPO.act`` / ``process_env_step``, publishing ``trial_done`` as the env would.

    ``sink`` collects ``(actor observation, action)`` per step, which is everything a from-scratch replay of the
    whole history needs.
    """
    obs = _sample_obs(NUM_ENVS, generator)
    with torch.no_grad():
        for step in range(STEPS_PER_ENV):
            actions = ppo.act(obs)
            if sink is not None:
                sink.append((obs["policy"].clone(), actions.clone()))
            next_obs = _sample_obs(NUM_ENVS, generator)
            rewards = torch.randn(NUM_ENVS, generator=generator, device=DEVICE)
            extras = {"trial_done": trial_dones[offset + step]}
            ppo.process_env_step(next_obs, rewards, dones[offset + step], extras)
            obs = next_obs
        ppo.compute_returns(obs)


def _replay_log_probs(
    policy: ActorCriticEpisodeContext,
    history: list,
    dones: torch.Tensor,
    trial_dones: torch.Tensor,
) -> torch.Tensor:
    """Log-probs of the whole history, recomputed FROM SCRATCH episode by episode, ``[S, N]``.

    Nothing of the collection machinery is reused: each episode is one batched
    :meth:`ActorCriticEpisodeContext.forward_sequence` over its own frames, and the memory chain is rebuilt by
    hand -- ``z_init`` for the first episode of a trial, ``G(H_prev)`` for the second (``K = 2``), with ``H``
    assembled the way a pass produces it: the ``M`` memory rows first (:meth:`memory_readout`), then the
    episode's readouts. If the windowed update path agrees with this, the ring buffer, the segment bookkeeping
    and the cached-``H`` writer are all reproducing what the acting path actually did.
    """
    num_steps = len(history)
    obs = torch.stack([frame for frame, _ in history])  # [S, N, obs]
    actions = torch.stack([action for _, action in history])  # [S, N, A]
    log_probs = torch.zeros(num_steps, NUM_ENVS, device=DEVICE)
    num_memory = policy.num_memory_tokens
    span = policy.hidden_history_span
    with torch.no_grad():
        for env in range(NUM_ENVS):
            memory = policy.z_init.detach().unsqueeze(0)  # [1, M, d]
            start = 0
            for step in range(num_steps):
                if not bool(dones[step, env]) and step != num_steps - 1:
                    continue
                episode = obs[start : step + 1, env : env + 1]  # [S_e, 1, obs]
                hidden = policy.forward_sequence(episode, memory=memory)  # [S_e, 1, d]
                policy.update_distribution_from_hidden(hidden.squeeze(1))
                log_probs[start : step + 1, env] = policy.get_actions_log_prob(actions[start : step + 1, env])
                # H of this pass: the M memory rows it ran with, then its readouts, zero-padded to M + T.
                length = step + 1 - start
                episode_hidden = torch.zeros(1, span, policy.d_model, device=DEVICE)
                valid = torch.zeros(1, span, dtype=torch.bool, device=DEVICE)
                episode_hidden[0, :num_memory] = policy.memory_readout(memory)[0]
                episode_hidden[0, num_memory : num_memory + length] = hidden.squeeze(1)
                valid[0, : num_memory + length] = True
                if bool(trial_dones[step, env]):
                    memory = policy.z_init.detach().unsqueeze(0)
                else:
                    memory, _ = policy.write_memory(episode_hidden, mask=valid)
                start = step + 1
    return log_probs


def _record_snapshots(ppo: EpisodeContextPPO) -> list[dict]:
    """Wrap the storage's snapshot push so a test can compare against exactly what was handed over."""
    storage = ppo.storage
    stock_push = storage.push_episode_hidden
    recorded: list[dict] = []

    def recording_push(env_ids, hidden, valid, trial_end, _stock=stock_push, _sink=recorded):
        _sink.append({
            "step": storage.step,
            "env_ids": env_ids.clone(),
            "hidden": hidden.clone(),
            "valid": valid.clone(),
            "trial_end": trial_end.clone(),
        })
        return _stock(env_ids, hidden, valid, trial_end)

    storage.push_episode_hidden = recording_push
    return recorded


# --------------------------------------------------------------------------------------------------
# A. Reconstruction canary with a live memory
# --------------------------------------------------------------------------------------------------


def test_epoch_zero_canary_with_memory() -> None:
    """The update path's memory must reproduce the acting path's, so the epoch-0 PPO ratio is still 1."""
    num_steps = 3 * STEPS_PER_ENV
    dones, trial_dones = _trial_schedule(num_steps)
    for critic_design in ("privileged", "shared_trunk"):
        policy = _make_policy(critic_design=critic_design)
        ppo = _make_ppo(policy)
        generator = torch.Generator(device=DEVICE).manual_seed(11)

        for rollout in range(3):
            _collect(ppo, dones, trial_dones, rollout * STEPS_PER_ENV, generator)
            stored_log_prob = ppo.storage.actions_log_prob.clone()

            max_error, rows, wrote_memory = 0.0, 0, False
            for batch in ppo.storage.recurrent_mini_batch_generator(num_mini_batches=2, num_epochs=1):
                obs_batch, actions_batch, prefix = batch[0], batch[1], batch[8][0]
                assert prefix.source_hidden is not None, "the generator dropped the memory sources"
                wrote_memory = wrote_memory or bool(prefix.segment_has_source.any())
                policy.act(obs_batch, masks=batch[9], hidden_state=prefix)
                log_prob = policy.get_actions_log_prob(actions_batch)
                max_error = max(max_error, (log_prob - batch[5].squeeze(-1)).abs().max().item())
                rows += int(log_prob.numel())
            assert rows == STEPS_PER_ENV * NUM_ENVS
            assert wrote_memory, "no segment ever had a source; the canary would be memory-free"
            assert max_error < 1e-5, f"reconstructed log-probs drifted by {max_error:.3e} ({critic_design})"

            # ... and the same statement through the real, unmodified PPO update.
            loss_dict = ppo.update()
            assert abs(loss_dict["ratio_mean"] - 1.0) < 1e-5, f"ratio_mean = {loss_dict['ratio_mean']}"
            assert abs(loss_dict["ratio_min"] - 1.0) < 1e-4 and abs(loss_dict["ratio_max"] - 1.0) < 1e-4, (
                f"ratio range [{loss_dict['ratio_min']}, {loss_dict['ratio_max']}] is not ~1 at epoch 0"
            )
            for key, value in loss_dict.items():
                assert value == value and abs(value) < float("inf"), f"{key} is not finite: {value}"
            assert stored_log_prob.abs().sum() > 0.0
            ppo.commit_obs_normalization(ppo.storage.collected_observations)
            print(
                f"[ok] epoch-0 canary with memory ({critic_design}), rollout {rollout}: max |d logp| = {max_error:.3e}"
            )


def test_windowed_update_matches_a_from_scratch_replay() -> None:
    """The generator's windows must reproduce a full, episode-by-episode recompute of the whole history.

    Independent of the acting path (which the canary above compares against): every episode is re-run in one
    batched pass with a memory chain rebuilt by hand. It covers windows that straddle an episode boundary, a
    trial boundary and a rollout boundary at once.
    """
    num_steps = 2 * STEPS_PER_ENV
    dones, trial_dones = _trial_schedule(num_steps)
    policy = _make_policy(seed=29)
    ppo = _make_ppo(policy)
    generator = torch.Generator(device=DEVICE).manual_seed(53)
    history: list = []

    for rollout in range(2):
        _collect(ppo, dones, trial_dones, rollout * STEPS_PER_ENV, generator, sink=history)
        replayed = _replay_log_probs(policy, history, dones, trial_dones)[rollout * STEPS_PER_ENV :]

        with torch.no_grad():
            batch = next(ppo.storage.recurrent_mini_batch_generator(num_mini_batches=1, num_epochs=1))
            policy.act(batch[0], masks=batch[9], hidden_state=batch[8][0])
            windowed = policy.get_actions_log_prob(batch[1])
        error = (windowed - replayed).abs().max().item()
        assert error < 1e-4, f"the windowed update and the from-scratch replay disagree by {error:.3e}"
        assert replayed.abs().sum().item() > 0.0
        # Vacuity guard: replaying the same history with the memory chain CUT (every episode starting from
        # z_init) must give a different answer, otherwise the agreement above says nothing about the memory.
        cut = _replay_log_probs(policy, history, dones, torch.ones_like(trial_dones))[rollout * STEPS_PER_ENV :]
        assert (cut - replayed).abs().max().item() > 1e-4, "cutting the memory chain changed nothing"
        ppo.update()
        ppo.commit_obs_normalization(ppo.storage.collected_observations)
        print(f"[ok] rollout {rollout}: windowed update == from-scratch replay, max |d logp| = {error:.3e}")


def test_the_memory_is_load_bearing_in_the_update() -> None:
    """The canary above would pass trivially if the source data were ignored: corrupt it and the batch must move."""
    num_steps = 2 * STEPS_PER_ENV
    dones, trial_dones = _trial_schedule(num_steps)
    policy = _make_policy(seed=3)
    ppo = _make_ppo(policy)
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    _collect(ppo, dones, trial_dones, 0, generator)
    ppo.update()
    _collect(ppo, dones, trial_dones, STEPS_PER_ENV, generator)

    batch = next(ppo.storage.recurrent_mini_batch_generator(num_mini_batches=1, num_epochs=1))
    prefix = batch[8][0]
    assert bool(prefix.segment_has_source.any()), "no source in this minibatch; the test would be vacuous"
    with torch.no_grad():
        reference = policy.act(batch[0], hidden_state=prefix).clone()
        # The corruption has to be non-affine: ``H`` reaches ``Z`` only through the writer's LayerNorms, so a
        # constant shift (or a rescaling) of every feature is exactly invariant and would look like a leak.
        noise = torch.randn(prefix.source_hidden.shape, generator=generator, device=DEVICE)
        wrecked = type(prefix)(
            obs=prefix.obs,
            positions=prefix.positions,
            window_positions=prefix.window_positions,
            memory_segments=prefix.memory_segments,
            source_hidden=prefix.source_hidden + noise,
            source_valid=prefix.source_valid,
            segment_has_source=prefix.segment_has_source,
        )
        perturbed = policy.act(batch[0], hidden_state=wrecked)
        blind = type(prefix)(
            obs=prefix.obs,
            positions=prefix.positions,
            window_positions=prefix.window_positions,
            memory_segments=prefix.memory_segments,
            source_hidden=prefix.source_hidden,
            source_valid=prefix.source_valid,
            segment_has_source=torch.zeros_like(prefix.segment_has_source),
        )
        without_source = policy.act(batch[0], hidden_state=blind)
    assert not torch.allclose(reference, perturbed), "the cached H is not read by the update path"
    assert not torch.allclose(reference, without_source), "dropping every source changed nothing"
    print("[ok] the update path genuinely writes the cached H into the memory it reads")


# --------------------------------------------------------------------------------------------------
# B. Segment / source correctness
# --------------------------------------------------------------------------------------------------


def test_segment_sources_are_the_right_episodes() -> None:
    """Several episodes inside one window: each segment reads the episode that ended right before it."""
    dones, trial_dones = _trial_schedule(STEPS_PER_ENV)
    positions, episode_index = _episode_labels(dones, trial_dones)
    policy = _make_policy(seed=7)
    ppo = _make_ppo(policy)
    recorded = _record_snapshots(ppo)
    generator = torch.Generator(device=DEVICE).manual_seed(31)
    _collect(ppo, dones, trial_dones, 0, generator)

    # The row labels must be the acting policy's own episode index.
    assert torch.equal(ppo.storage.row_episode_index, episode_index), "the stored episode index is wrong"
    # Env 0 (episodes of 2 steps) must have closed three episodes, one of them a trial, inside this window.
    env_zero = [
        (entry["step"], bool(entry["trial_end"][(entry["env_ids"] == 0).nonzero()].item()))
        for entry in recorded
        if bool((entry["env_ids"] == 0).any())
    ]
    assert [step for step, _ in env_zero] == [1, 3, 5], f"env 0's dones landed at {env_zero}"
    assert [end for _, end in env_zero] == [False, True, False], "the K=2 trial boundary was not recorded"

    batch = next(ppo.storage.recurrent_mini_batch_generator(num_mini_batches=1, num_epochs=1))
    prefix = batch[8][0]
    num_prefix = prefix.obs.shape[0]
    assert torch.equal(prefix.memory_segments[:num_prefix], torch.zeros_like(prefix.memory_segments[:num_prefix]))

    snapshot_of = {
        (entry["step"], int(env)): index for index, entry in enumerate(recorded) for env in entry["env_ids"].tolist()
    }
    for env in range(NUM_ENVS):
        segments = prefix.memory_segments[num_prefix:, env]
        expected = (positions[:, env] == 0).long().cumsum(dim=0)
        assert torch.equal(segments, expected), f"env {env}: segments {segments.tolist()} != {expected.tolist()}"

        for segment in range(int(segments.max().item()) + 1):
            rows = (segments == segment).nonzero(as_tuple=False).squeeze(-1)
            if rows.numel() == 0:  # segment 0 can be prefix-only when the window opens on an episode start
                continue
            first = int(rows[0].item())
            expects_source = bool(episode_index[first, env].item() > 0)
            assert bool(prefix.segment_has_source[env, segment]) == expects_source, (
                f"env {env} segment {segment}: has_source should be {expects_source}"
            )
            if not expects_source or first == 0:
                continue  # a segment opening at row 0 sources from the (empty, first-rollout) persistent slot
            entry = recorded[snapshot_of[first - 1, env]]
            row = int((entry["env_ids"] == env).nonzero().item())
            assert torch.equal(prefix.source_hidden[env, segment], entry["hidden"][row]), (
                f"env {env} segment {segment} read the wrong episode's H"
            )
            assert torch.equal(prefix.source_valid[env, segment], entry["valid"][row])
    print("[ok] every segment sources from the episode that ended immediately before it; trials cut the chain")


def test_trial_boundary_cuts_the_source() -> None:
    """The segment after a trial end must read ``z_init``, and the one after a plain done must not."""
    dones, trial_dones = _trial_schedule(STEPS_PER_ENV)
    _, episode_index = _episode_labels(dones, trial_dones)
    policy = _make_policy(seed=13)
    ppo = _make_ppo(policy)
    generator = torch.Generator(device=DEVICE).manual_seed(37)
    _collect(ppo, dones, trial_dones, 0, generator)

    prefix = next(ppo.storage.recurrent_mini_batch_generator(num_mini_batches=1, num_epochs=1))[8][0]
    num_prefix = prefix.obs.shape[0]
    # Env 0: rows 0-1 episode 0 (no source), rows 2-3 episode 1 (source), rows 4-5 episode 0 again (trial reset).
    # This is the very first rollout, so every environment opens ON an episode start: segment 0 holds the (inert,
    # zero-filled) prefix alone and the window's own segments are numbered from 1.
    segments = prefix.memory_segments[num_prefix:, 0].tolist()
    assert segments == [1, 1, 2, 2, 3, 3], f"env 0 segments: {segments}"
    assert episode_index[:, 0].tolist() == [0, 0, 1, 1, 0, 0]
    assert prefix.segment_has_source[0].tolist() == [False, False, True, False], (
        f"env 0 has_source: {prefix.segment_has_source[0].tolist()}"
    )

    # And it is visible in the memory the policy builds: the trial-reset segment gets exactly z_init.
    memory = policy.memory_from_prefix(prefix)
    z_init = policy.z_init.detach()
    assert torch.equal(memory[0, 3], z_init), "the segment after a trial end did not read z_init"
    assert torch.equal(memory[0, 1], z_init), "the first episode of a trial did not read z_init"
    assert (memory[0, 2] - z_init).abs().max().item() > 1e-6, "the written segment was not written"
    print("[ok] a trial boundary restores z_init for the segment that follows it")


# --------------------------------------------------------------------------------------------------
# C. Persistence across clear()
# --------------------------------------------------------------------------------------------------


def test_source_survives_clear() -> None:
    """A source established in rollout k must be what rollout k+1's opening segment reads."""
    num_steps = 2 * STEPS_PER_ENV
    dones, trial_dones = _trial_schedule(num_steps)
    _, episode_index = _episode_labels(dones, trial_dones)
    policy = _make_policy(seed=17)
    ppo = _make_ppo(policy)
    recorded = _record_snapshots(ppo)
    generator = torch.Generator(device=DEVICE).manual_seed(41)

    _collect(ppo, dones, trial_dones, 0, generator)
    last_of_rollout = {}
    for entry in recorded:  # chronological, so the last write per environment wins -- as in _commit_sources
        for row, env in enumerate(entry["env_ids"].tolist()):
            last_of_rollout[env] = (entry["hidden"][row].clone(), bool(entry["trial_end"][row]))
    ppo.update()  # ends in storage.clear(), which folds the snapshots into the persistent slot
    assert not ppo.storage._snapshot_envs, "the rollout-local snapshots were not released"
    for env, (hidden, trial_end) in last_of_rollout.items():
        assert torch.equal(ppo.storage.source_hidden[env], hidden), f"env {env}'s persistent source is stale"
        assert bool(ppo.storage.has_source[env]) == (not trial_end)

    _collect(ppo, dones, trial_dones, STEPS_PER_ENV, generator)
    prefix = next(ppo.storage.recurrent_mini_batch_generator(num_mini_batches=1, num_epochs=1))[8][0]
    num_prefix = prefix.obs.shape[0]
    checked = 0
    for env in range(NUM_ENVS):
        segments = prefix.memory_segments[num_prefix:, env]
        opening = int(segments[0].item())  # the segment the window opens in
        expects_source = bool(episode_index[STEPS_PER_ENV, env].item() > 0)
        assert bool(prefix.segment_has_source[env, opening]) == expects_source
        if not expects_source:
            continue
        hidden, _ = last_of_rollout[env]
        assert torch.equal(prefix.source_hidden[env, opening], hidden), (
            f"env {env}'s opening segment did not read the source written in the previous rollout"
        )
        checked += 1
    assert checked > 0, "no environment carried a source across the rollout boundary; the test would be vacuous"
    print(f"[ok] {checked} environments read a source that was written in the previous rollout")


def test_rollout_without_any_done() -> None:
    """A window entirely inside one episode: one segment, sourced purely from the persistent slot.

    With ``T = 80`` and 32-step rollouts this is the COMMON case, not an edge case -- most rollouts of a real run
    contain no episode boundary for most environments, and some contain none at all.
    """
    num_steps = 2 * STEPS_PER_ENV
    dones, trial_dones = _trial_schedule(num_steps)
    quiet = torch.zeros_like(dones)  # second rollout: nothing ever terminates
    policy = _make_policy(seed=31)
    ppo = _make_ppo(policy)
    generator = torch.Generator(device=DEVICE).manual_seed(59)

    _collect(ppo, dones, trial_dones, 0, generator)
    source_before = ppo.storage.source_hidden.clone()
    ppo.update()
    assert not torch.equal(ppo.storage.source_hidden, source_before), "the first rollout wrote no source at all"
    established = ppo.storage.source_hidden.clone()
    has_source = ppo.storage.has_source.clone()

    _collect(ppo, quiet, quiet, STEPS_PER_ENV, generator)
    assert not ppo.storage._snapshot_envs, "no done should have produced no snapshot"
    prefix = next(ppo.storage.recurrent_mini_batch_generator(num_mini_batches=1, num_epochs=1))[8][0]
    num_prefix = prefix.obs.shape[0]
    for env in range(NUM_ENVS):
        segments = prefix.memory_segments[num_prefix:, env]
        # One episode, one segment. Its id is 1 rather than 0 for an environment whose previous episode ended
        # exactly on the rollout boundary (row 0 then opens a segment), which sources from the same slot.
        assert int(segments.min()) == int(segments.max()), f"env {env} was split into segments {segments.tolist()}"
        segment = int(segments[0])
        assert bool(prefix.segment_has_source[env, segment]) == bool(has_source[env])
        assert torch.equal(prefix.source_hidden[env, segment], established[env]), (
            f"env {env}: the done-free window lost its persistent source"
        )
    ppo.update()  # and the fold-in is a no-op, not a crash, when there is nothing to fold
    assert torch.equal(ppo.storage.source_hidden, established)
    print("[ok] a rollout without a single done keeps reading the source established before it")


# --------------------------------------------------------------------------------------------------
# D. Gradient flow
# --------------------------------------------------------------------------------------------------


def test_generator_gradients_reach_the_memory_modules() -> None:
    """A PPO-style loss on a minibatch straight off the generator trains the writer, z_init and the trunk."""
    num_steps = 2 * STEPS_PER_ENV
    dones, trial_dones = _trial_schedule(num_steps)
    policy = _make_policy(seed=19, critic_design="shared_trunk")
    ppo = _make_ppo(policy)
    generator = torch.Generator(device=DEVICE).manual_seed(43)
    _collect(ppo, dones, trial_dones, 0, generator)
    ppo.update()
    _collect(ppo, dones, trial_dones, STEPS_PER_ENV, generator)

    batch = next(ppo.storage.recurrent_mini_batch_generator(num_mini_batches=1, num_epochs=1))
    prefix = batch[8][0]
    assert not prefix.source_hidden.requires_grad, "the storage handed out a source that is already in a graph"
    # Make it a leaf: if any gradient reached the cached H, this would pick it up.
    prefix.source_hidden.requires_grad_(True)
    policy.zero_grad(set_to_none=True)

    policy.act(batch[0], masks=batch[9], hidden_state=prefix)
    ratio = torch.exp(policy.get_actions_log_prob(batch[1]) - batch[5].squeeze(-1))
    value = policy.evaluate(batch[0])
    loss = -(ratio * batch[3].squeeze(-1)).mean() + (value - batch[4]).pow(2).mean()
    loss.backward()

    trained = {
        "z_init": policy.z_init,
        "memory_pos_embed": policy.memory_pos_embed,
        "writer.attn.q_proj.weight": policy.writer.attn.q_proj.weight,
        "writer.ff.0.weight": policy.writer.ff[0].weight,
        "blocks.0.attn.k_proj.weight": policy.blocks[0].attn.k_proj.weight,
    }
    for name, parameter in trained.items():
        assert parameter.grad is not None, f"{name} got no gradient"
        assert parameter.grad.abs().max().item() > 0.0, f"{name}'s gradient is exactly zero"
    assert prefix.source_hidden.grad is None, "a gradient reached the cached source episode"
    print("[ok] the generator's minibatch trains the writer, z_init and the trunk; the cached H stays out")


# --------------------------------------------------------------------------------------------------
# E. memory_tokens = 0 is the stock path
# --------------------------------------------------------------------------------------------------


def test_memory_tokens_zero_is_the_stock_step() -> None:
    """Without a memory, ``process_env_step`` must be the stock one and the generator must carry no memory."""
    dones, trial_dones = _trial_schedule(STEPS_PER_ENV)
    stored = []
    for use_override in (True, False):
        policy = _make_policy(memory_tokens=0, seed=23)
        ppo = _make_ppo(policy)
        assert not ppo.storage.has_memory
        generator = torch.Generator(device=DEVICE).manual_seed(47)
        obs = _sample_obs(NUM_ENVS, generator)
        with torch.no_grad():
            for step in range(STEPS_PER_ENV):
                ppo.act(obs)
                next_obs = _sample_obs(NUM_ENVS, generator)
                rewards = torch.randn(NUM_ENVS, generator=generator, device=DEVICE)
                extras = {"trial_done": trial_dones[step], "time_outs": dones[step]}
                # ``use_override`` False calls the stock implementation, bypassing the subclass entirely.
                step_fn = ppo.process_env_step if use_override else PPO.process_env_step.__get__(ppo, type(ppo))
                step_fn(next_obs, rewards, dones[step], extras)
                obs = next_obs
            ppo.compute_returns(obs)
        stored.append({
            "rewards": ppo.storage.rewards.clone(),
            "returns": ppo.storage.returns.clone(),
            "advantages": ppo.storage.advantages.clone(),
            "values": ppo.storage.values.clone(),
            "actions": ppo.storage.actions.clone(),
            "log_prob": ppo.storage.actions_log_prob.clone(),
            "dones": ppo.storage.dones.clone(),
            "trial_dones": ppo.storage.trial_dones.clone(),
            "positions": ppo.storage.frame_positions.clone(),
        })
        prefix = next(ppo.storage.recurrent_mini_batch_generator(num_mini_batches=2, num_epochs=1))[8][0]
        assert prefix.source_hidden is None and prefix.segment_has_source is None
        assert prefix.memory is None and prefix.memory_segments is None
        assert not hasattr(ppo.storage, "row_episode_index")

    with_override, stock = stored
    for key, value in with_override.items():
        assert torch.equal(value, stock[key]), f"the override changed '{key}' with memory_tokens=0"
    print("[ok] memory_tokens=0: process_env_step and the generator are the stock ones, transition for transition")


if __name__ == "__main__":
    test_epoch_zero_canary_with_memory()
    test_the_memory_is_load_bearing_in_the_update()
    test_segment_sources_are_the_right_episodes()
    test_trial_boundary_cuts_the_source()
    test_source_survives_clear()
    test_rollout_without_any_done()
    test_generator_gradients_reach_the_memory_modules()
    test_memory_tokens_zero_is_the_stock_step()
    print("all episode-context memory pipeline tests passed")
