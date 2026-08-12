# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the hierarchical trial-memory transformer policy.

Self-contained: pure torch, no Isaac Sim / SimulationApp needed. Run with::

    source /home/yandabao/miniforge3/etc/profile.d/conda.sh && conda activate patlab
    PYTHONPATH=/home/yandabao/rsl_rl-wt/meta-memory:$PYTHONPATH python tests/test_actor_critic_trial_memory.py

The load-bearing test is ``test_incremental_matches_batched``: the acting path (one token per step, KV cached)
and the training path (one batched forward over a padded episode) must produce the same ``h``. Everything the PPO
reconstruction canary claims rests on that equality.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.modules import ActorCriticTrialMemory

DEVICE = "cpu"
OBS_DIM = 5
ACTION_DIM = 4
NUM_ENVS = 3
T_EPISODE = 6  # small T so the tests stay fast; the shapes are the same at T = 80
NUM_MEMORY = 3


def _make_policy(
    attention_window: int | None = None,
    seed: int = 0,
    noise_std_type: str = "scalar",
    init_noise_std: float = 1.0,
    perturb: bool = True,
) -> ActorCriticTrialMemory:
    """A small float64 policy. Double precision makes "tight tolerance" mean something."""
    torch.manual_seed(seed)
    sample_obs = TensorDict({"policy": torch.zeros(NUM_ENVS, OBS_DIM)}, batch_size=[NUM_ENVS], device=DEVICE)
    policy = ActorCriticTrialMemory(
        obs=sample_obs,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=ACTION_DIM,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_memory_tokens=NUM_MEMORY,
        attention_window=attention_window,
        max_episode_length=T_EPISODE,
        ff_mult=2,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        noise_std_type=noise_std_type,
        init_noise_std=init_noise_std,
    )
    # Randomize the LayerNorms so that they are not the identity and the test can actually see a difference.
    # (``perturb=False`` keeps the fresh init, which the gSDE effective-std identity is stated against.)
    if perturb:
        for parameter in policy.parameters():
            with torch.no_grad():
                parameter.add_(0.1 * torch.randn_like(parameter))
    return policy.double().eval()


def _make_episode(num_steps: int, seed: int = 1) -> dict[str, torch.Tensor]:
    """A synthetic episode of ``num_steps`` tokens (use ``T + 1`` to include the terminal token)."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64, device=DEVICE)  # noqa: E731
    return {
        "obs": randn(num_steps, NUM_ENVS, OBS_DIM),
        "prev_actions": randn(num_steps, NUM_ENVS, ACTION_DIM),
        "prev_rewards": randn(num_steps, NUM_ENVS, 1),
        "prev_dones": (randn(num_steps, NUM_ENVS, 1) > 1.5).double(),
    }


def _roll_incrementally(policy: ActorCriticTrialMemory, episode: dict[str, torch.Tensor]) -> torch.Tensor:
    """Step the acting path one token at a time and collect ``h`` ``[S, N, d]``."""
    num_steps = episode["obs"].shape[0]
    policy.initialize_state(NUM_ENVS, DEVICE)
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


def _roll_incrementally_with_distribution(
    policy: ActorCriticTrialMemory, episode: dict[str, torch.Tensor], actions: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Acting path, but also reading off the action distribution at every step.

    This is exactly what ``act`` does (``forward_step`` -> ``_update_distribution``), only with the sampled action
    replaced by a fixed one so that the log-probs are comparable against the batched path.
    """
    num_steps = episode["obs"].shape[0]
    policy.initialize_state(NUM_ENVS, DEVICE)
    means, stds, log_probs, entropies = [], [], [], []
    for step in range(num_steps):
        hidden = policy.forward_step(
            episode["obs"][step],
            prev_actions=episode["prev_actions"][step],
            prev_rewards=episode["prev_rewards"][step],
            prev_dones=episode["prev_dones"][step],
            commit=True,
        )
        policy.update_distribution_from_hidden(hidden)
        means.append(policy.action_mean.clone())
        stds.append(policy.action_std.clone())
        entropies.append(policy.entropy.clone())
        log_probs.append(policy.get_actions_log_prob(actions[step]).clone())
    return {
        "mean": torch.stack(means),
        "std": torch.stack(stds),
        "entropy": torch.stack(entropies),
        "log_prob": torch.stack(log_probs),
    }


def _batched_distribution(
    policy: ActorCriticTrialMemory, episode: dict[str, torch.Tensor], actions: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Run the PPO reconstruction path: one batched forward, then one distribution over ``[S, B, ...]``."""
    hidden = _roll_batched(policy, episode)
    policy.update_distribution_from_hidden(hidden)
    return {
        "mean": policy.action_mean.clone(),
        "std": policy.action_std.clone(),
        "entropy": policy.entropy.clone(),
        "log_prob": policy.get_actions_log_prob(actions).clone(),
    }


def _roll_batched(
    policy: ActorCriticTrialMemory, episode: dict[str, torch.Tensor], mask: torch.Tensor | None = None
) -> torch.Tensor:
    return policy.forward_sequence(
        episode["obs"],
        episode["prev_actions"],
        episode["prev_rewards"],
        episode["prev_dones"],
        memory=policy.initial_memory(NUM_ENVS).double(),
        mask=mask,
    )


# --------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------


def test_incremental_matches_batched() -> None:
    """Acting one token at a time must reproduce the single batched forward, for full and windowed attention."""
    episode = _make_episode(T_EPISODE + 1)  # T action tokens + 1 terminal token

    for window, label in ((None, "full (W >= T)"), (3, "windowed (W = 3)")):
        policy = _make_policy(attention_window=window)
        with torch.no_grad():
            incremental = _roll_incrementally(policy, episode)
            batched = _roll_batched(policy, episode)
        error = (incremental - batched).abs().max().item()
        assert incremental.shape == batched.shape == (T_EPISODE + 1, NUM_ENVS, policy.d_model)
        assert error < 1e-10, f"incremental vs batched mismatch ({label}): {error:.3e}"
        print(f"[ok] incremental == batched, {label}: max |dh| = {error:.3e}")

    # ... and the heads agree too, so acted and reconstructed actions/values match.
    policy = _make_policy()
    with torch.no_grad():
        incremental = _roll_incrementally(policy, episode)
        batched = _roll_batched(policy, episode)
        mean_error = (policy.action_mean_from_hidden(incremental) - policy.action_mean_from_hidden(batched)).abs().max()
        value_error = (policy.value_from_hidden(incremental) - policy.value_from_hidden(batched)).abs().max()
    assert mean_error.item() < 1e-10 and value_error.item() < 1e-10
    print(f"[ok] heads agree: |d_mean| = {mean_error.item():.3e}, |d_value| = {value_error.item():.3e}")


def test_incremental_matches_batched_with_ragged_mask() -> None:
    """A padded (early-terminated) episode: valid rows must match the unpadded incremental rollout bit-for-bit."""
    valid_len = 4
    policy = _make_policy()
    episode = _make_episode(T_EPISODE + 1, seed=7)

    with torch.no_grad():
        incremental = _roll_incrementally(policy, episode)

    # Env 1 ends after ``valid_len`` steps; garbage in the padded region must not leak into the valid rows.
    mask = torch.ones(T_EPISODE + 1, NUM_ENVS, dtype=torch.bool, device=DEVICE)
    mask[valid_len:, 1] = False
    padded = {key: value.clone() for key, value in episode.items()}
    for key in ("obs", "prev_actions", "prev_rewards", "prev_dones"):
        padded[key][valid_len:, 1] = 123.0

    with torch.no_grad():
        batched = _roll_batched(policy, padded, mask=mask)
        clean = _roll_batched(policy, episode, mask=mask)

    # Same code path, same mask, different garbage in the padded region -> must be bit-identical.
    assert torch.equal(batched[:valid_len, 1], clean[:valid_len, 1]), "padding leaked into the valid rows"
    assert torch.equal(batched[:, [0, 2]], clean[:, [0, 2]]), "env 1's padding leaked into the other environments"
    # And the masked batched forward still agrees with the unpadded incremental rollout.
    error = (batched[:valid_len, 1] - incremental[:valid_len, 1]).abs().max().item()
    other_error = (batched[:, [0, 2]] - incremental[:, [0, 2]]).abs().max().item()
    assert error < 1e-10 and other_error < 1e-10, f"masked batched vs incremental: {error:.3e} / {other_error:.3e}"
    print(f"[ok] ragged episode: padding inert (bit-identical), vs incremental max |dh| = {error:.3e}")


def test_causality_by_perturbation() -> None:
    """Perturbing the input at time ``t`` must leave ``h_s`` bit-identical for every ``s < t``."""
    policy = _make_policy()
    episode = _make_episode(T_EPISODE + 1, seed=3)
    perturb_step = 3

    with torch.no_grad():
        reference = _roll_batched(policy, episode)
        perturbed_episode = {key: value.clone() for key, value in episode.items()}
        perturbed_episode["obs"][perturb_step] += 5.0
        perturbed_episode["prev_actions"][perturb_step] -= 3.0
        perturbed_episode["prev_rewards"][perturb_step] += 7.0
        perturbed = _roll_batched(policy, perturbed_episode)

    assert torch.equal(reference[:perturb_step], perturbed[:perturb_step]), (
        f"h_s for s < {perturb_step} changed -- the causal mask leaks information backwards"
    )
    changed = (reference[perturb_step:] - perturbed[perturb_step:]).abs().max().item()
    assert changed > 1e-3, f"the perturbation had no effect at all ({changed:.3e}); the test would be vacuous"

    # Same statement on the acting path.
    with torch.no_grad():
        reference_inc = _roll_incrementally(policy, episode)
        perturbed_inc = _roll_incrementally(policy, perturbed_episode)
    assert torch.equal(reference_inc[:perturb_step], perturbed_inc[:perturb_step]), "acting path is not causal"
    print(f"[ok] causal: h_{{s<{perturb_step}}} bit-identical, h_{{s>={perturb_step}}} moved by {changed:.3e}")


def test_memory_is_constant_within_an_episode() -> None:
    """``Z`` must not drift while an episode runs; it changes only when the writer is invoked."""
    policy = _make_policy()
    episode = _make_episode(T_EPISODE + 1, seed=4)
    policy.initialize_state(NUM_ENVS, DEVICE)
    initial = policy.memory.clone()

    with torch.no_grad():
        for step in range(T_EPISODE + 1):
            policy.forward_step(
                episode["obs"][step],
                prev_actions=episode["prev_actions"][step],
                prev_rewards=episode["prev_rewards"][step],
                prev_dones=episode["prev_dones"][step],
            )
            assert torch.equal(policy.memory, initial), f"Z drifted at step {step}"

        # The writer, and only the writer, moves Z.
        policy.update_memory()
        after_write = policy.memory.clone()
        assert not torch.allclose(after_write, initial), "the writer did not change Z"

        # An episode boundary keeps Z; a trial boundary restores Z_init.
        policy.reset_episode(torch.ones(NUM_ENVS, dtype=torch.bool, device=DEVICE))
        assert torch.equal(policy.memory, after_write), "reset_episode must retain Z"
        assert policy._positions.sum().item() == 0 and not policy._token_valid.any(), "short-term state not cleared"

        trial_dones = torch.tensor([True, False, False], device=DEVICE)
        policy.reset_trial(trial_dones)
        assert torch.equal(policy.memory[0], policy.z_init.detach().double()), "reset_trial must restore Z_init"
        assert torch.equal(policy.memory[1:], after_write[1:]), "reset_trial must not touch other environments"

    print("[ok] Z is constant within an episode, survives reset_episode, and is restored by reset_trial")


def test_writer_attention_is_m_by_t_plus_one() -> None:
    """The writer reads ``Q = Z`` against ``K = V = H_e``: the attention is ``M x (T + 1)``."""
    policy = _make_policy()
    batch = NUM_ENVS
    memory = policy.initial_memory(batch).double()
    hidden = torch.randn(batch, T_EPISODE + 1, policy.d_model, dtype=torch.float64, device=DEVICE)

    with torch.no_grad():
        new_memory, weights = policy.write_memory(memory, hidden, need_weights=True)

    assert weights is not None, "need_weights=True must return the attention tensor"
    assert weights.shape == (batch, policy.num_heads, NUM_MEMORY, T_EPISODE + 1), (
        f"writer attention must be M x (T + 1), got {tuple(weights.shape)}"
    )
    assert new_memory.shape == (batch, NUM_MEMORY, policy.d_model)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums)), "attention rows must be normalized"

    # The same shape comes out of the acting path's writer call.
    policy.initialize_state(NUM_ENVS, DEVICE)
    episode = _make_episode(T_EPISODE + 1, seed=5)
    with torch.no_grad():
        for step in range(T_EPISODE + 1):
            policy.forward_step(episode["obs"][step], prev_actions=episode["prev_actions"][step])
        acting_weights = policy.update_memory(need_weights=True)
    assert acting_weights.shape == (NUM_ENVS, policy.num_heads, NUM_MEMORY, T_EPISODE + 1)
    print(f"[ok] writer attention shape {tuple(weights.shape)} = [B, heads, M, T+1]")


def test_writer_ignores_padded_rows() -> None:
    """Rows of ``H_e`` past an episode's length must not influence ``Z_{e+1}``."""
    policy = _make_policy()
    batch = NUM_ENVS
    valid_len = 4
    memory = policy.initial_memory(batch).double()
    hidden = torch.randn(batch, T_EPISODE + 1, policy.d_model, dtype=torch.float64, device=DEVICE)
    mask = torch.ones(batch, T_EPISODE + 1, dtype=torch.bool, device=DEVICE)
    mask[1, valid_len:] = False

    with torch.no_grad():
        reference, _ = policy.write_memory(memory, hidden, mask=mask)
        perturbed_hidden = hidden.clone()
        # A *structured* perturbation: the writer LayerNorms its keys, so a constant offset would cancel out.
        perturbed_hidden[1, valid_len:] += 50.0 * torch.randn_like(perturbed_hidden[1, valid_len:])
        perturbed, _ = policy.write_memory(memory, perturbed_hidden, mask=mask)

    assert torch.equal(reference[1], perturbed[1]), "the masked tail of H_e leaked into Z"
    assert torch.equal(reference[0], perturbed[0]) and torch.equal(reference[2], perturbed[2])

    # Without the mask the same perturbation clearly matters, so the assertion above is not vacuous.
    with torch.no_grad():
        unmasked_reference, _ = policy.write_memory(memory, hidden)
        unmasked_perturbed, _ = policy.write_memory(memory, perturbed_hidden)
    delta = (unmasked_reference[1] - unmasked_perturbed[1]).abs().max().item()
    assert delta > 1e-3, f"unmasked writer barely reacted ({delta:.3e}); the masking test would be vacuous"

    # A fully masked episode must not produce NaNs (degenerate but reachable on a zero-length episode).
    empty_mask = mask.clone()
    empty_mask[2] = False
    with torch.no_grad():
        degenerate, _ = policy.write_memory(memory, hidden, mask=empty_mask)
    assert torch.isfinite(degenerate).all(), "a fully masked H_e produced NaNs"
    print(f"[ok] writer masking: padded rows ignored (unmasked they move Z by {delta:.3e})")


def test_gradient_reaches_z_init() -> None:
    """A loss on episode 1 must reach the learned NO_MEMORY tokens."""
    policy = _make_policy()
    episode = _make_episode(T_EPISODE + 1, seed=6)

    memory = policy.initial_memory(NUM_ENVS)
    hidden = policy.forward_sequence(
        episode["obs"], episode["prev_actions"], episode["prev_rewards"], episode["prev_dones"], memory=memory
    )
    loss = policy.value_from_hidden(hidden).square().mean() + policy.action_mean_from_hidden(hidden).square().mean()
    loss.backward()

    assert policy.z_init.grad is not None, "no gradient reached Z_init"
    grad_norm = policy.z_init.grad.norm().item()
    assert grad_norm > 0.0 and torch.isfinite(policy.z_init.grad).all(), f"degenerate Z_init gradient: {grad_norm}"

    # ... and through the writer into the next episode as well.
    policy.zero_grad(set_to_none=True)
    memory = policy.initial_memory(NUM_ENVS)
    hidden = policy.forward_sequence(
        episode["obs"], episode["prev_actions"], episode["prev_rewards"], episode["prev_dones"], memory=memory
    )
    next_memory, _ = policy.write_memory(memory, hidden.transpose(0, 1))
    hidden_2 = policy.forward_sequence(
        episode["obs"], episode["prev_actions"], episode["prev_rewards"], episode["prev_dones"], memory=next_memory
    )
    policy.value_from_hidden(hidden_2).square().mean().backward()
    assert policy.z_init.grad is not None and policy.z_init.grad.norm().item() > 0.0
    assert policy.writer.attn.q_proj.weight.grad is not None, "no gradient reached the writer"
    print(f"[ok] gradient reaches Z_init (|g| = {grad_norm:.3e}) and the writer")


def test_tensordict_obs_and_action_api() -> None:
    """``act`` / ``act_inference`` / ``evaluate`` work off a TensorDict and keep the usual ActorCritic API."""
    policy = _make_policy()
    policy.initialize_state(NUM_ENVS, DEVICE)
    obs = TensorDict(
        {"policy": torch.randn(NUM_ENVS, OBS_DIM, dtype=torch.float64)}, batch_size=[NUM_ENVS], device=DEVICE
    )

    actions = policy.act(obs)
    assert actions.shape == (NUM_ENVS, ACTION_DIM)
    assert policy.action_mean.shape == (NUM_ENVS, ACTION_DIM)
    assert policy.action_std.shape == (NUM_ENVS, ACTION_DIM)
    assert policy.entropy.shape == (NUM_ENVS,)
    assert policy.get_actions_log_prob(actions).shape == (NUM_ENVS,)

    # evaluate() right after act() reuses h_t without appending a second token
    position_before = policy._positions.clone()
    values = policy.evaluate(obs)
    assert values.shape == (NUM_ENVS, 1)
    assert torch.equal(policy._positions, position_before), "evaluate() must not consume a token slot"

    # the bootstrap evaluate() after record_transition peeks without committing
    policy.record_transition(torch.zeros(NUM_ENVS, dtype=torch.float64), torch.zeros(NUM_ENVS, dtype=torch.bool))
    bootstrap = policy.evaluate(obs)
    assert bootstrap.shape == (NUM_ENVS, 1)
    assert torch.equal(policy._positions, position_before), "the bootstrap evaluate() must not commit a token"

    deterministic = policy.act_inference(obs)
    assert deterministic.shape == (NUM_ENVS, ACTION_DIM)
    assert policy._positions.tolist() == [2] * NUM_ENVS, "act/act_inference must each consume exactly one token"

    memory_state, _ = policy.get_hidden_states()
    assert memory_state.shape == (NUM_ENVS, NUM_MEMORY, policy.d_model)
    print("[ok] TensorDict obs + act/act_inference/evaluate API")


def test_terminal_token_for_a_subset_of_envs() -> None:
    """``append_terminal_token(env_ids=...)`` advances only those envs and leaves the others untouched."""
    policy = _make_policy()
    episode = _make_episode(T_EPISODE + 1, seed=8)
    action_steps = {key: value[:T_EPISODE] for key, value in episode.items()}
    with torch.no_grad():
        _roll_incrementally(policy, action_steps)

        cache_before = [cache.clone() for cache in policy._key_cache]
        positions_before = policy._positions.clone()
        env_ids = torch.tensor([1], device=DEVICE)
        policy.append_terminal_token(
            episode["obs"][T_EPISODE],
            prev_actions=episode["prev_actions"][T_EPISODE],
            prev_rewards=episode["prev_rewards"][T_EPISODE],
            prev_dones=episode["prev_dones"][T_EPISODE],
            env_ids=env_ids,
        )

    assert policy._positions.tolist() == [T_EPISODE, T_EPISODE + 1, T_EPISODE], "only env 1 should have advanced"
    for layer, cache in enumerate(policy._key_cache):
        assert torch.equal(cache[0], cache_before[layer][0]) and torch.equal(cache[2], cache_before[layer][2]), (
            "the terminal token of env 1 disturbed another environment's cache"
        )
    assert policy._token_valid[1].sum().item() == T_EPISODE + 1
    assert policy._token_valid[0].sum().item() == T_EPISODE
    assert positions_before.tolist() == [T_EPISODE] * NUM_ENVS
    print("[ok] terminal token can be appended for a subset of environments")


def test_attention_window_restricts_context() -> None:
    """``W < T`` must actually drop old tokens; the default ``W >= T`` must keep them (section 15 ablation knob)."""
    episode = _make_episode(T_EPISODE + 1, seed=9)
    perturbed_episode = {key: value.clone() for key, value in episode.items()}
    perturbed_episode["obs"][0] += 5.0

    sensitivities = {}
    for window, label in ((None, "full"), (2, "W=2")):
        policy = _make_policy(attention_window=window)
        with torch.no_grad():
            reference = _roll_batched(policy, episode)
            perturbed = _roll_batched(policy, perturbed_episode)
        sensitivities[label] = (reference[-1] - perturbed[-1]).abs().max().item()

    assert sensitivities["full"] > 1e-3, "full attention should carry step 0 all the way to the last token"
    assert sensitivities["W=2"] < 1e-12, f"a W=2 window must not see step 0, got {sensitivities['W=2']:.3e}"
    print(f"[ok] attention window: full={sensitivities['full']:.3e}, W=2={sensitivities['W=2']:.3e}")


def test_gsde_incremental_matches_batched() -> None:
    """Under gSDE the acting path and the batched reconstruction path must give the *same* distribution.

    gSDE keys the std on ``h_t``, so a distribution mismatch is no longer just a mean mismatch -- it would also
    corrupt the std and hence the PPO ratio. Tight tolerance, float64.
    """
    episode = _make_episode(T_EPISODE + 1, seed=11)
    actions = _make_episode(T_EPISODE + 1, seed=12)["prev_actions"]  # arbitrary fixed actions [S, N, A]

    for window, label in ((None, "full (W >= T)"), (3, "windowed (W = 3)")):
        policy = _make_policy(attention_window=window, noise_std_type="gsde", init_noise_std=0.5)
        assert policy.log_std.shape == (policy.d_model, ACTION_DIM), "gSDE log_std must be [d_model, num_actions]"
        with torch.no_grad():
            incremental = _roll_incrementally_with_distribution(policy, episode, actions)
            batched = _batched_distribution(policy, episode, actions)

        errors = {key: (incremental[key] - batched[key]).abs().max().item() for key in incremental}
        assert incremental["std"].shape == (T_EPISODE + 1, NUM_ENVS, ACTION_DIM)
        for key, error in errors.items():
            assert error < 1e-10, f"gSDE {key} mismatch, {label}: {error:.3e}"

        # Non-vacuous: the std really is state-dependent, otherwise this would just be testing a constant.
        spread = incremental["std"].std(dim=(0, 1)).max().item()
        assert spread > 1e-3, f"gSDE std is effectively constant ({spread:.3e}); the test would be vacuous"
        print(
            f"[ok] gSDE incremental == batched, {label}: max |d mean| = {errors['mean']:.3e},"
            f" |d std| = {errors['std']:.3e}, |d logp| = {errors['log_prob']:.3e} (std spread {spread:.3e})"
        )


def test_gsde_log_prob_is_reproducible() -> None:
    """Check the canary property: the same actions get the same log-prob on both paths.

    Re-drawing the gSDE exploration matrix in between must not move it either: the exploration matrix feeds only
    ``get_noise``, while the density comes from ``sqrt(h_t^2 @ sigma^2)``. If
    ``sample_weights`` could move a log-prob, every PPO ratio would be wrong the moment the weights were re-drawn.
    """
    policy = _make_policy(noise_std_type="gsde", init_noise_std=0.5, seed=2)
    episode = _make_episode(T_EPISODE + 1, seed=13)
    actions = _make_episode(T_EPISODE + 1, seed=14)["prev_actions"]

    with torch.no_grad():
        acted = _roll_incrementally_with_distribution(policy, episode, actions)["log_prob"]
        # Re-draw the exploration matrix between "acting" and "learning", the worst case for the PPO ratio.
        policy.distribution.sample_weights(policy.log_std)
        reconstructed = _batched_distribution(policy, episode, actions)["log_prob"]
        # ... and once more through the acting path, to show the acting path itself is deterministic in log_prob.
        acted_again = _roll_incrementally_with_distribution(policy, episode, actions)["log_prob"]

    ratio_error = (torch.exp(reconstructed - acted) - 1.0).abs().max().item()
    assert (reconstructed - acted).abs().max().item() < 1e-10, "log-probs drifted across the two paths"
    assert torch.equal(acted, acted_again), "the acting path's log_prob is not reproducible"
    assert ratio_error < 1e-9, f"epoch-0 PPO ratio would be {1.0 + ratio_error:.9f}, not 1"
    assert acted.abs().sum().item() > 0.0, "all log-probs are zero; the test would be vacuous"
    print(f"[ok] gSDE log_prob reproducible across paths and across sample_weights: |ratio - 1| = {ratio_error:.3e}")


def test_gsde_effective_std_scaling() -> None:
    """Check the scaling documented by ``calibrate_gsde_init.py``.

    The realized action std is ``init_noise_std * ||phi||``, and for a fresh readout LayerNorm
    ``||phi|| = sqrt(d_model)``. Here ``phi`` is ``h_t`` (the ``final_norm`` output).
    """
    episode = _make_episode(1, seed=15)
    measured = {}
    for init_noise_std in (1.0, 0.1):
        policy = _make_policy(noise_std_type="gsde", init_noise_std=init_noise_std, perturb=False)
        policy.initialize_state(NUM_ENVS, DEVICE)
        with torch.no_grad():
            hidden = policy.forward_step(episode["obs"][0], prev_actions=episode["prev_actions"][0])
            policy.update_distribution_from_hidden(hidden)
            std = policy.action_std
            feature_norm = hidden.norm(dim=-1)

        # Exact identity (all sigma equal at init): std_j = sigma * ||h||, up to the distribution's 1e-6 epsilon.
        exact = init_noise_std * feature_norm.unsqueeze(-1).expand_as(std)
        exact_error = ((std - exact).abs() / exact).max().item()
        assert exact_error < 1e-5, f"std != init * ||h||: relative error {exact_error:.3e}"

        # ... and for a fresh final_norm, ||h|| == sqrt(d_model), i.e. the sqrt(d_model) amplification the
        # calibration script warns about (a naive init of 1.0 does NOT give an effective std of 1.0).
        reference = policy.d_model**0.5
        norm_error = (feature_norm / reference - 1.0).abs().max().item()
        assert norm_error < 0.02, f"||h|| = {feature_norm.mean().item():.3f} != sqrt(d_model) = {reference:.3f}"
        measured[init_noise_std] = std.mean().item()
        print(
            f"[ok] gSDE effective std: init={init_noise_std} -> std={std.mean().item():.4f}"
            f" (= init * ||h||, ||h|| = {feature_norm.mean().item():.3f} vs sqrt(d_model) = {reference:.3f})"
        )

    # The amplification is linear in init_noise_std, so calibration is a single division.
    assert abs(measured[1.0] / measured[0.1] - 10.0) < 1e-3, "effective std is not linear in init_noise_std"
    assert abs(measured[1.0] / 32**0.5 - 1.0) < 0.02, "init_noise_std=1.0 must give ~sqrt(d_model), not ~1.0"


def test_gsde_checkpoint_type_mismatch_fails_loudly() -> None:
    """A checkpoint saved under one noise type must not load silently into another (the shapes differ)."""
    scalar_policy = _make_policy()
    gsde_policy = _make_policy(noise_std_type="gsde")
    log_policy = _make_policy(noise_std_type="log")

    for source, target, label in (
        (scalar_policy, gsde_policy, "scalar -> gsde"),
        (gsde_policy, scalar_policy, "gsde -> scalar"),
        (gsde_policy, log_policy, "gsde -> log"),
        (log_policy, gsde_policy, "log -> gsde"),
    ):
        for strict in (True, False):
            try:
                target.load_state_dict(source.state_dict(), strict=strict)
            except ValueError:
                continue
            raise AssertionError(f"{label} (strict={strict}) loaded silently instead of raising")

    # The matching case still works, and re-arms the gSDE distribution.
    reloaded = _make_policy(noise_std_type="gsde", seed=5)
    reloaded.load_state_dict(gsde_policy.state_dict())
    assert torch.equal(reloaded.log_std, gsde_policy.log_std)
    episode = _make_episode(1, seed=16)
    reloaded.initialize_state(NUM_ENVS, DEVICE)
    with torch.no_grad():
        actions = reloaded.act(TensorDict({"policy": episode["obs"][0]}, batch_size=[NUM_ENVS], device=DEVICE))
    assert actions.shape == (NUM_ENVS, ACTION_DIM) and torch.isfinite(actions).all()
    print("[ok] mismatched noise_std_type checkpoints raise; a matching gSDE checkpoint reloads and acts")


def test_design_doc_defaults() -> None:
    """The documented configuration (d=256, L=4, heads=8, M=8, W >= T, T=80) instantiates and runs."""
    sample_obs = TensorDict({"policy": torch.zeros(2, 43)}, batch_size=[2], device=DEVICE)
    policy = ActorCriticTrialMemory(
        obs=sample_obs, obs_groups={"policy": ["policy"], "critic": ["policy"]}, num_actions=7
    ).eval()
    assert (policy.d_model, policy.num_layers, policy.num_heads) == (256, 4, 8)
    assert (policy.num_memory_tokens, policy.max_episode_length, policy.max_tokens) == (8, 80, 81)
    assert policy.attention_window >= policy.max_tokens, "W must default to full-episode attention"
    assert policy.z_init.shape == (8, 256)

    batch, num_steps = 2, 81
    with torch.no_grad():
        hidden = policy.forward_sequence(
            torch.randn(num_steps, batch, 43),
            torch.randn(num_steps, batch, 7),
            torch.randn(num_steps, batch),
            torch.zeros(num_steps, batch),
            memory=policy.initial_memory(batch),
        )
        next_memory, weights = policy.write_memory(
            memory=policy.initial_memory(batch), hidden=hidden.transpose(0, 1), need_weights=True
        )
    assert hidden.shape == (81, batch, 256)
    assert next_memory.shape == (batch, 8, 256)
    assert weights.shape == (batch, 8, 8, 81), f"writer attention must be 8 x 81, got {tuple(weights.shape)}"
    params = sum(parameter.numel() for parameter in policy.parameters()) / 1e6
    print(f"[ok] design-doc defaults: h {tuple(hidden.shape)}, writer attn {tuple(weights.shape)}, {params:.2f}M")


if __name__ == "__main__":
    test_incremental_matches_batched()
    test_incremental_matches_batched_with_ragged_mask()
    test_causality_by_perturbation()
    test_memory_is_constant_within_an_episode()
    test_writer_attention_is_m_by_t_plus_one()
    test_writer_ignores_padded_rows()
    test_gradient_reaches_z_init()
    test_tensordict_obs_and_action_api()
    test_terminal_token_for_a_subset_of_envs()
    test_attention_window_restricts_context()
    test_gsde_incremental_matches_batched()
    test_gsde_log_prob_is_reproducible()
    test_gsde_effective_std_scaling()
    test_gsde_checkpoint_type_mismatch_fails_loudly()
    test_design_doc_defaults()
    print("all trial-memory policy tests passed")
