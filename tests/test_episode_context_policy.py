# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the single-episode context transformer policy and its frame ring buffer.

Self-contained: pure torch, no Isaac Sim / SimulationApp needed. Run with::

    source /home/yandabao/miniforge3/etc/profile.d/conda.sh && conda activate patlab
    python -m pytest tests/test_episode_context_policy.py -q

The load-bearing tests are

* ``test_step_matches_window``: the acting path (one frame per step, rolling KV cache) and the update path
  (one causal pass over ``[prefix | window]``) must produce the same ``h`` -- across an episode boundary inside
  the window and with the prefix truncated at an episode start. Everything the PPO reconstruction canary claims
  rests on that equality.
* ``test_ring_buffer_sufficiency``: with ``T = 80``, ``W = 32`` and ``L = 80`` every window row's reconstructed
  context must equal the ground truth (its own frames since its episode start, capped at ``L``), including the
  worst case of a window that starts on episode step 79.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.modules import ActorCriticEpisodeContext, EpisodeContextPrefix
from rsl_rl.storage import EpisodeContextRolloutStorage
from rsl_rl.storage.rollout_storage import RolloutStorage

DEVICE = "cpu"
OBS_DIM = 5
ACTION_DIM = 4
NUM_ENVS = 3
T_EPISODE = 6  # small T so the tests stay fast; the shapes are the same at T = 80


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _make_policy(
    context_length: int = T_EPISODE,
    seed: int = 0,
    noise_std_type: str = "scalar",
    num_envs: int = NUM_ENVS,
    obs_normalization: bool = False,
) -> ActorCriticEpisodeContext:
    """A small float64 policy. Double precision makes "tight tolerance" mean something."""
    torch.manual_seed(seed)
    sample_obs = TensorDict({"policy": torch.zeros(num_envs, OBS_DIM)}, batch_size=[num_envs], device=DEVICE)
    policy = ActorCriticEpisodeContext(
        obs=sample_obs,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=ACTION_DIM,
        actor_obs_normalization=obs_normalization,
        context_length=context_length,
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
    # Move the LayerNorms/heads off their initialization so the tests are not measuring a near-identity network.
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if name in ("std", "log_std"):
                continue
            parameter.add_(0.1 * torch.randn_like(parameter))
    return policy.double().eval()


def _positions_from_dones(dones: torch.Tensor) -> torch.Tensor:
    """Episode step of every row, ``[S, N]``, from the per-step done flags (a done ends the row's episode)."""
    num_steps, num_envs = dones.shape
    positions = torch.zeros(num_steps, num_envs, dtype=torch.long, device=dones.device)
    step = torch.zeros(num_envs, dtype=torch.long, device=dones.device)
    for t in range(num_steps):
        positions[t] = step
        step = (step + 1) * (~dones[t].bool()).long()
    return positions


def _roll_incrementally(policy: ActorCriticEpisodeContext, obs: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
    """Step the acting path one frame at a time and collect ``h`` ``[S, N, d]``."""
    policy.initialize_state(obs.shape[1], DEVICE, dtype=obs.dtype)
    hidden = []
    for step in range(obs.shape[0]):
        hidden.append(policy.forward_step(obs[step], commit=True))
        policy.reset(dones[step])
    return torch.stack(hidden)


def _window_hidden(
    policy: ActorCriticEpisodeContext, obs: torch.Tensor, positions: torch.Tensor, window_start: int
) -> torch.Tensor:
    """The update path: one causal pass over ``[prefix | window]``, returning ``h`` of the window only."""
    prefix_start = max(0, window_start - policy.context_prefix_length)
    prefix = EpisodeContextPrefix(
        obs=obs[prefix_start:window_start],
        positions=positions[prefix_start:window_start],
        window_positions=positions[window_start:],
    )
    return policy.forward_window(obs[window_start:], prefix)


# --------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------


def test_step_matches_window() -> None:
    """Acting one frame at a time must reproduce the batched ``[prefix | window]`` forward."""
    num_steps = 3 * T_EPISODE
    generator = torch.Generator(device=DEVICE).manual_seed(11)
    obs = torch.randn(num_steps, NUM_ENVS, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)

    # Desynchronized environments: env e ends its first episode e steps early, so the windows below contain
    # episode boundaries at different rows per environment.
    dones = torch.zeros(num_steps, NUM_ENVS, dtype=torch.bool, device=DEVICE)
    for env in range(NUM_ENVS):
        step = T_EPISODE - 1 - env
        while step < num_steps:
            dones[step, env] = True
            step += T_EPISODE
    positions = _positions_from_dones(dones)

    for context_length, label in ((T_EPISODE, "full episode (L = T)"), (3, "windowed (L = 3)")):
        policy = _make_policy(context_length=context_length)
        with torch.no_grad():
            incremental = _roll_incrementally(policy, obs, dones)
            # A window in the middle of the run: it has a full prefix and contains episode boundaries.
            middle = _window_hidden(policy, obs, positions, window_start=2 * T_EPISODE - 2)
            # ... and one at the very beginning, where the prefix is truncated by the start of the run.
            early = _window_hidden(policy, obs, positions, window_start=2)

        middle_error = (incremental[2 * T_EPISODE - 2 :] - middle).abs().max().item()
        early_error = (incremental[2:] - early).abs().max().item()
        assert middle.shape == incremental[2 * T_EPISODE - 2 :].shape
        assert middle_error < 1e-10, f"step vs window mismatch ({label}): {middle_error:.3e}"
        assert early_error < 1e-10, f"step vs window mismatch at a truncated prefix ({label}): {early_error:.3e}"
        # The test would be vacuous if the window were context-free
        assert incremental.abs().max().item() > 1e-3
        print(f"[ok] step == window, {label}: max |dh| = {max(middle_error, early_error):.3e}")


def test_window_is_episode_local_and_span_limited() -> None:
    """A row's ``h`` must depend on its own episode's frames within reach -- and on nothing else.

    "Within reach" is ``num_layers * (L - 1)`` frames, not ``L - 1``: every block attends ``L - 1`` rows back, so
    a two-block trunk sees ``2 * (L - 1)``. That depth factor is exactly what sizes the storage's prefix
    (:attr:`ActorCriticEpisodeContext.context_prefix_length`), so it is asserted here rather than assumed.
    """
    num_steps = 2 * T_EPISODE
    generator = torch.Generator(device=DEVICE).manual_seed(5)
    obs = torch.randn(num_steps, NUM_ENVS, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)
    dones = torch.zeros(num_steps, NUM_ENVS, dtype=torch.bool, device=DEVICE)
    dones[T_EPISODE - 1] = True
    dones[2 * T_EPISODE - 1] = True
    positions = _positions_from_dones(dones)

    context_length = 2
    policy = _make_policy(context_length=context_length)
    reach = policy.num_layers * (context_length - 1)
    assert policy.context_prefix_length == min(T_EPISODE - 1, reach)
    query = T_EPISODE + 4  # episode step 4 of the second episode, so the reach fits inside the episode
    with torch.no_grad():
        reference = policy.forward_sequence(obs, positions=positions)
        for perturbed_step in range(num_steps):
            noisy = obs.clone()
            noisy[perturbed_step] += 7.0
            perturbed = policy.forward_sequence(noisy, positions=positions)
            distance = query - perturbed_step
            # Causal, never across the episode boundary at T_EPISODE, never further back than the reach.
            inside = 0 <= distance <= reach and perturbed_step >= T_EPISODE
            changed = not torch.equal(reference[query], perturbed[query])
            assert changed == inside, (
                f"h[{query}] {'changed' if changed else 'did not change'} when step {perturbed_step} moved;"
                f" expected {'a change' if inside else 'no change'}"
            )
    print(f"[ok] the query row sees exactly its own episode's last {reach + 1} frames (L={context_length}, 2 layers)")


def test_forward_sequence_matches_forward_step() -> None:
    """The whole-episode helper the BC script uses agrees with the acting path."""
    policy = _make_policy(context_length=T_EPISODE)
    generator = torch.Generator(device=DEVICE).manual_seed(17)
    obs = torch.randn(T_EPISODE, NUM_ENVS, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)
    dones = torch.zeros(T_EPISODE, NUM_ENVS, dtype=torch.bool, device=DEVICE)
    dones[-1] = True
    with torch.no_grad():
        incremental = _roll_incrementally(policy, obs, dones)
        batched = policy.forward_sequence(obs)
    error = (incremental - batched).abs().max().item()
    assert error < 1e-10, f"forward_sequence vs forward_step: {error:.3e}"

    # A ragged (early terminated) episode: garbage in the padded rows must not reach the valid ones.
    valid = 3
    mask = torch.ones(T_EPISODE, NUM_ENVS, dtype=torch.bool, device=DEVICE)
    mask[valid:, 1] = False
    padded = obs.clone()
    padded[valid:, 1] = 123.0
    with torch.no_grad():
        masked = policy.forward_sequence(padded, mask=mask)
        clean = policy.forward_sequence(obs, mask=mask)
    assert torch.equal(masked[:valid, 1], clean[:valid, 1]), "padding leaked into the valid rows"
    assert torch.equal(masked[:, [0, 2]], clean[:, [0, 2]]), "env 1's padding leaked into the other environments"
    assert (masked[:valid, 1] - incremental[:valid, 1]).abs().max().item() < 1e-10
    print(f"[ok] forward_sequence == forward_step: max |dh| = {error:.3e}, padding inert")


def test_forward_window_batch_first_segments() -> None:
    """The batch-first ``forward_window(obs [B, S, d], seg_mask)`` convention used by the BC script.

    ``seg_mask`` holds an integer segment id per position: a position attends only inside its own segment, so
    right padding with an id of its own is inert, and a second episode in the same row starts a fresh context.
    """
    policy = _make_policy(context_length=T_EPISODE)
    generator = torch.Generator(device=DEVICE).manual_seed(29)
    num_steps = 2 * T_EPISODE
    obs = torch.randn(NUM_ENVS, num_steps, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)

    with torch.no_grad():
        # One segment == one episode starting at step 0, and it agrees with the time-major helper.
        single = policy.forward_window(obs[:, :T_EPISODE], torch.zeros(NUM_ENVS, T_EPISODE, dtype=torch.long))
        reference = policy.forward_sequence(obs[:, :T_EPISODE].transpose(0, 1)).transpose(0, 1)
        assert single.shape == (NUM_ENVS, T_EPISODE, policy.d_model)
        assert torch.equal(single, reference), "batch-first and time-major disagree on a single episode"
        # ``seg_mask=None`` means the same thing.
        assert torch.equal(policy.forward_window(obs[:, :T_EPISODE]), single)

        # Two segments in one row: the second must be identical to running those frames on their own.
        segments = torch.zeros(NUM_ENVS, num_steps, dtype=torch.long)
        segments[:, T_EPISODE:] = 1
        both = policy.forward_window(obs, segments)
        second = policy.forward_window(obs[:, T_EPISODE:], torch.zeros(NUM_ENVS, T_EPISODE, dtype=torch.long))
        # Tolerance, not bit-identity: a 12-row pass and a 6-row pass do the same arithmetic in a different
        # kernel/reduction order. Bit-identity is asserted below, where both sides have the same shape.
        first_error = (both[:, :T_EPISODE] - single).abs().max().item()
        second_error = (both[:, T_EPISODE:] - second).abs().max().item()
        assert first_error < 1e-12, f"the second segment leaked backwards: {first_error:.3e}"
        assert second_error < 1e-12, f"the second segment saw the first one: {second_error:.3e}"

        # Right padding (its own id) leaves the real rows bit-identical whatever it contains.
        padded = obs.clone()
        padded[:, T_EPISODE:] = 999.0
        garbage = policy.forward_window(padded, segments)
        assert torch.equal(garbage[:, :T_EPISODE], both[:, :T_EPISODE]), "padding leaked into the valid rows"
    print("[ok] batch-first forward_window(obs, seg_mask): segments isolate, padding inert")


def test_cache_reset_isolates_environments() -> None:
    """Resetting env 1 must clear only env 1's context, and leave the others bit-identical."""
    policy = _make_policy(context_length=T_EPISODE)
    generator = torch.Generator(device=DEVICE).manual_seed(23)
    obs = torch.randn(4, NUM_ENVS, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)
    dones = torch.zeros(4, NUM_ENVS, dtype=torch.bool, device=DEVICE)
    dones[1, 1] = True  # env 1 starts a new episode at row 2

    with torch.no_grad():
        rolled = _roll_incrementally(policy, obs, dones)
        rolled_positions = policy.positions.tolist()
        # Env 1's rows 2..3 must equal a fresh two-step rollout of the same frames.
        fresh = _roll_incrementally(policy, obs[2:], torch.zeros(2, NUM_ENVS, dtype=torch.bool, device=DEVICE))
    assert torch.equal(rolled[2:, 1], fresh[:, 1]), "the reset did not clear env 1's context"
    assert not torch.equal(rolled[2:, 0], fresh[:, 0]), "env 0 was reset too (its context should have survived)"
    assert rolled_positions == [4, 2, 4]
    print("[ok] per-environment reset: only the done environment lost its context")


def test_ring_buffer_sufficiency() -> None:
    """T = 80, W = 32, L = 80: every window row's reconstructed context must equal the ground truth."""
    episode_length, window, context_length = 80, 32, 80
    num_envs, num_rollouts = 16, 6
    obs_dim = 3
    storage = EpisodeContextRolloutStorage(
        "rl",
        num_envs,
        window,
        TensorDict({"policy": torch.zeros(num_envs, obs_dim)}, batch_size=[num_envs], device=DEVICE),
        [ACTION_DIM],
        device=DEVICE,
        actor_obs_groups=["policy"],
        context_length=context_length,
        max_episode_length=episode_length,
        num_layers=4,
    )
    # L = T, so a row's context is capped by its own episode: 79 frames of history, 112 ring slots.
    assert storage.ring_size == window + min(context_length, episode_length) == 112
    assert storage.prefix_length == 79

    # Environment e ends its first episode ``2e + 1`` steps early, so the environments are permanently phase
    # shifted and the window start lands on many different episode steps across the run -- including 79, the
    # tightest one (env 15 at the 5th rollout), which is the case the ring buffer is sized for.
    total_steps = num_rollouts * window
    dones_all = torch.zeros(total_steps, num_envs, dtype=torch.bool, device=DEVICE)
    for env in range(num_envs):
        boundary = episode_length - 1 - (2 * env + 1)
        while boundary < total_steps:
            dones_all[boundary, env] = True
            boundary += episode_length
    positions_all = _positions_from_dones(dones_all)

    generator = torch.Generator(device=DEVICE).manual_seed(3)
    history_obs = []
    worst_case_seen = False

    for rollout in range(num_rollouts):
        for _ in range(window):
            global_step = len(history_obs)
            obs = torch.randn(num_envs, obs_dim, generator=generator, device=DEVICE)
            dones = dones_all[global_step]
            history_obs.append(obs)

            transition = RolloutStorage.Transition()
            transition.observations = TensorDict({"policy": obs}, batch_size=[num_envs], device=DEVICE)
            transition.actions = torch.zeros(num_envs, ACTION_DIM, device=DEVICE)
            transition.rewards = torch.zeros(num_envs, device=DEVICE)
            transition.dones = dones
            transition.values = torch.zeros(num_envs, 1, device=DEVICE)
            transition.actions_log_prob = torch.zeros(num_envs, device=DEVICE)
            transition.action_mean = torch.zeros(num_envs, ACTION_DIM, device=DEVICE)
            transition.action_sigma = torch.ones(num_envs, ACTION_DIM, device=DEVICE)
            storage.add_transitions(transition)

        # -- the claim: the [prefix | window] slice reconstructs every row's true context --
        prefix_obs, prefix_positions, window_positions = storage.context_slice()
        sequence = torch.cat([prefix_obs, torch.stack(history_obs[-window:])], dim=0)
        positions = torch.cat([prefix_positions, window_positions], dim=0)
        prefix_length = storage.prefix_length
        first_global = len(history_obs) - window

        truth_obs = torch.stack(history_obs)
        truth_pos = positions_all[: len(history_obs)]
        for row in range(window):
            global_step = first_global + row
            index = prefix_length + row
            assert torch.equal(positions[index], truth_pos[global_step]), "window episode steps are wrong"
            reach = truth_pos[global_step].clamp(max=context_length - 1)
            for env in range(num_envs):
                back = int(reach[env].item())
                expected = truth_obs[global_step - back : global_step + 1, env]
                got = sequence[index - back : index + 1, env]
                assert torch.equal(got, expected), (
                    f"rollout {rollout}, row {row}, env {env}: the reconstructed context of episode step"
                    f" {int(truth_pos[global_step, env].item())} does not match the collected frames"
                )
                assert torch.equal(
                    positions[index - back : index + 1, env], truth_pos[global_step - back : global_step + 1, env]
                )
                if row == 0 and back == episode_length - 1:
                    worst_case_seen = True  # a window starting on episode step 79: the tightest ring demand
        storage.clear()

    assert worst_case_seen, "the worst case (window starting at episode step 79) never occurred; test is vacuous"
    print(f"[ok] ring buffer: {num_rollouts} rollouts x {window} rows x {num_envs} envs reconstructed exactly")


# --------------------------------------------------------------------------------------------------
# Shared-trunk critic (critic_design="shared_trunk")
# --------------------------------------------------------------------------------------------------


def _make_shared_policy(
    detach_critic_trunk: bool = False,
    privileged_group: bool = False,
    seed: int = 0,
    context_length: int = T_EPISODE,
) -> ActorCriticEpisodeContext:
    """A shared-trunk-critic policy. ``privileged_group`` adds a critic group that must be IGNORED."""
    torch.manual_seed(seed)
    tensors = {"policy": torch.zeros(NUM_ENVS, OBS_DIM)}
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    if privileged_group:
        tensors["critic"] = torch.zeros(NUM_ENVS, OBS_DIM + 2)
        obs_groups["critic"] = ["critic"]
    policy = ActorCriticEpisodeContext(
        obs=TensorDict(tensors, batch_size=[NUM_ENVS], device=DEVICE),
        obs_groups=obs_groups,
        num_actions=ACTION_DIM,
        context_length=context_length,
        d_model=32,
        num_layers=2,
        num_heads=4,
        max_episode_length=T_EPISODE,
        ff_mult=2,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        critic_design="shared_trunk",
        detach_critic_trunk=detach_critic_trunk,
        init_noise_std=0.5,
    )
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if name in ("std", "log_std"):
                continue
            parameter.add_(0.1 * torch.randn_like(parameter))
    return policy.double().eval()


def _obs_dict(frames: torch.Tensor, privileged: torch.Tensor | None = None) -> TensorDict:
    tensors = {"policy": frames}
    if privileged is not None:
        tensors["critic"] = privileged
    return TensorDict(tensors, batch_size=list(frames.shape[:-1]), device=DEVICE)


def test_critic_design_routing() -> None:
    """The value head's input is the critic observation when privileged and ``d_model`` when shared."""
    privileged = _make_policy()  # default design
    assert privileged.critic_design == "privileged"
    assert privileged.critic[0].in_features == privileged.num_critic_obs == OBS_DIM

    shared = _make_shared_policy(privileged_group=True)
    assert shared.critic_design == "shared_trunk"
    assert shared.critic[0].in_features == shared.d_model
    assert shared.critic[-1].out_features == 1
    # The head must still be reachable as ``critic.*`` (uwlab's critic-warmup / strict-load filters key on it).
    assert any(name.startswith("critic.") for name, _ in shared.named_parameters())
    # A critic group that this environment does not publish is tolerated (it is never read).
    absent = ActorCriticEpisodeContext(
        obs=TensorDict({"policy": torch.zeros(NUM_ENVS, OBS_DIM)}, batch_size=[NUM_ENVS], device=DEVICE),
        obs_groups={"policy": ["policy"], "critic": ["does_not_exist"]},
        num_actions=ACTION_DIM,
        d_model=16,
        num_layers=1,
        num_heads=2,
        max_episode_length=T_EPISODE,
        ff_mult=2,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        critic_design="shared_trunk",
        critic_obs_normalization=True,
    )
    assert absent.critic[0].in_features == absent.d_model
    assert not absent.critic_obs_normalization, "a shared-trunk critic must not normalize an unused observation"
    absent.update_normalization(TensorDict({"policy": torch.zeros(2, OBS_DIM)}, batch_size=[2], device=DEVICE))
    print("[ok] critic_design routes the value head's input (privileged obs vs d_model trunk readout)")


def test_shared_trunk_value_uses_the_actor_context() -> None:
    """Values must come off the actor's ``h`` -- context-dependent, privileged-observation-independent."""
    policy = _make_shared_policy(privileged_group=True, seed=3)
    num_steps = 2 * T_EPISODE
    generator = torch.Generator(device=DEVICE).manual_seed(17)
    frames = torch.randn(num_steps, NUM_ENVS, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)
    dones = torch.zeros(num_steps, NUM_ENVS, dtype=torch.bool, device=DEVICE)
    dones[T_EPISODE - 1] = True
    positions = _positions_from_dones(dones)

    with torch.no_grad():
        # -- collection: act() then evaluate() on the same frame, exactly as ppo.act does --
        policy.initialize_state(NUM_ENVS, DEVICE, dtype=frames.dtype)
        step_values, step_hidden = [], []
        for step in range(num_steps):
            obs = _obs_dict(frames[step], torch.randn(NUM_ENVS, OBS_DIM + 2, generator=generator, dtype=torch.float64))
            policy.act(obs)
            hidden = policy._last_hidden.clone()
            value = policy.evaluate(obs)
            assert torch.allclose(value, policy.critic(hidden)), "evaluate() did not read the acting path's h_t"
            # The privileged group is pure noise and must not move the value. Compared with the cache bypassed,
            # so that the equality is a statement about the forward and not about the cache.
            other = _obs_dict(
                frames[step], torch.randn(NUM_ENVS, OBS_DIM + 2, generator=generator, dtype=torch.float64)
            )
            assert torch.equal(
                policy.evaluate(obs, use_cached_hidden=False), policy.evaluate(other, use_cached_hidden=False)
            ), "the shared-trunk critic read the privileged obs"
            step_values.append(value)
            step_hidden.append(hidden)
            policy.reset(dones[step])
        step_values = torch.stack(step_values)

        # -- update: one batched [prefix | window] pass; PPO calls act() first, then evaluate() --
        window_start = T_EPISODE - 2
        prefix_start = max(0, window_start - policy.context_prefix_length)
        prefix = EpisodeContextPrefix(
            obs=frames[prefix_start:window_start],
            positions=positions[prefix_start:window_start],
            window_positions=positions[window_start:],
        )
        window_obs = _obs_dict(frames[window_start:])
        policy.act(window_obs, hidden_state=prefix)
        window_values = policy.evaluate(window_obs)
        assert window_values.shape == (num_steps - window_start, NUM_ENVS, 1)
        error = (window_values - step_values[window_start:]).abs().max().item()
        assert error < 1e-10, f"batched values disagree with the acting path's: {error:.3e}"

        # Context-dependence: corrupting the prefix must move the values of the rows that reach into it.
        wrecked = EpisodeContextPrefix(
            obs=prefix.obs + 5.0, positions=prefix.positions, window_positions=prefix.window_positions
        )
        policy.act(window_obs, hidden_state=wrecked)
        assert not torch.allclose(window_values, policy.evaluate(window_obs)), (
            "corrupting the prefix left the values unchanged; the critic is not conditioned on the context"
        )

        # -- bootstrap: an observation that never went through act() is peeked, not committed --
        policy.reset(dones[-1])
        cache_before = [cache.clone() for cache in policy._key_cache]
        positions_before = policy._positions.clone()
        boot_obs = _obs_dict(torch.randn(NUM_ENVS, OBS_DIM, generator=generator, dtype=torch.float64))
        boot_value = policy.evaluate(boot_obs)
        assert boot_value.shape == (NUM_ENVS, 1)
        assert torch.equal(policy._positions, positions_before), "the bootstrap value advanced the episode step"
        for before, after in zip(cache_before, policy._key_cache):
            assert torch.equal(before, after), "the bootstrap value was committed to the KV cache"
        assert torch.allclose(boot_value, policy.critic(policy.forward_step(boot_obs, commit=False)))
    print("[ok] shared-trunk values: acting h_t on collection, batched h on update, peeked h on the bootstrap")


def test_shared_trunk_value_gradients_reach_the_trunk() -> None:
    """``detach_critic_trunk`` is the switch: the value loss shapes the trunk only when it is ``False``."""
    generator = torch.Generator(device=DEVICE).manual_seed(23)
    frames = torch.randn(T_EPISODE, NUM_ENVS, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)
    positions = torch.arange(T_EPISODE, device=DEVICE).unsqueeze(1).expand(T_EPISODE, NUM_ENVS)
    trunk_names = ("token_embed", "blocks", "final_norm", "pos_embed", "start_embed")

    for detach in (False, True):
        policy = _make_shared_policy(detach_critic_trunk=detach, seed=7)
        policy.train()
        prefix = EpisodeContextPrefix(obs=frames[:0], positions=positions[:0], window_positions=positions)
        obs = _obs_dict(frames)
        policy.zero_grad(set_to_none=True)
        policy.act(obs, hidden_state=prefix)  # what ppo.py does right before evaluate()
        value_loss = policy.evaluate(obs).pow(2).mean()
        value_loss.backward()

        trunk_grad = 0.0
        for name, parameter in policy.named_parameters():
            if name.startswith(trunk_names):
                trunk_grad += 0.0 if parameter.grad is None else parameter.grad.abs().sum().item()
        head_grad = sum(
            parameter.grad.abs().sum().item()
            for name, parameter in policy.named_parameters()
            if name.startswith("critic.") and parameter.grad is not None
        )
        assert head_grad > 0.0, "the value loss did not even reach the value head"
        if detach:
            assert trunk_grad == 0.0, f"detach_critic_trunk=True still leaked {trunk_grad:.3e} into the trunk"
        else:
            assert trunk_grad > 0.0, "detach_critic_trunk=False did not train the trunk on the value loss"
        print(f"[ok] detach_critic_trunk={detach}: trunk grad = {trunk_grad:.3e}, head grad = {head_grad:.3e}")


def test_privileged_evaluate_is_unchanged() -> None:
    """The default design is untouched: a context-free MLP on the critic observation, no trunk involvement."""
    policy = _make_policy(seed=4)
    generator = torch.Generator(device=DEVICE).manual_seed(31)
    frames = torch.randn(NUM_ENVS, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)
    obs = _obs_dict(frames)
    with torch.no_grad():
        policy.initialize_state(NUM_ENVS, DEVICE, dtype=frames.dtype)
        reference = policy.evaluate(obs)
        assert torch.equal(reference, policy.critic(frames)), "the privileged critic stopped being a plain MLP"
        # It is context-free: acting (which moves h_t) must not move the value of the same observation.
        policy.act(obs)
        assert torch.equal(reference, policy.evaluate(obs))
    try:
        policy.value_from_hidden(torch.zeros(NUM_ENVS, policy.d_model, dtype=torch.float64))
    except RuntimeError:
        pass
    else:
        raise AssertionError("value_from_hidden() must refuse to fabricate a value for a privileged critic")
    print("[ok] critic_design='privileged' is byte-unchanged behavior")


if __name__ == "__main__":
    test_step_matches_window()
    test_window_is_episode_local_and_span_limited()
    test_forward_sequence_matches_forward_step()
    test_forward_window_batch_first_segments()
    test_cache_reset_isolates_environments()
    test_ring_buffer_sufficiency()
    test_critic_design_routing()
    test_shared_trunk_value_uses_the_actor_context()
    test_shared_trunk_value_gradients_reach_the_trunk()
    test_privileged_evaluate_is_unchanged()
    print("all episode-context policy tests passed")
