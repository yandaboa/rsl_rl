# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the trial-memory PPO path: episode pairs, the per-epoch memory sweep and gradient routing.

Self-contained: pure torch, no Isaac Sim / SimulationApp needed. Run with::

    source /home/yandabao/miniforge3/etc/profile.d/conda.sh && conda activate patlab
    PYTHONPATH=/home/yandabao/rsl_rl-wt/meta-memory:$PYTHONPATH python tests/test_trial_pair_ppo.py

The training unit is an adjacent episode **pair**: the source episode supplies a differentiable memory context
and the target episode takes the clipped PPO loss. For K = 3 the pairs are ``(None, t1), (t1, t2), (t2, t3)``, so
every episode is a target exactly once.

The load-bearing test is ``test_epoch_zero_canary``: with unchanged parameters, the reconstruction of a rollout
must reproduce the behavior log-probs, i.e. the PPO ratio must be 1. If that fails, every advantage is being
applied to a different policy than the one that acted.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCriticTrialMemory

DEVICE = "cpu"
OBS_DIM = 5
ACTION_DIM = 4
T_EPISODE = 6  # small T so the tests stay fast; the shapes are the same at T = 80
K = 3
T_TRIAL = K * T_EPISODE
NUM_MEMORY = 3
GAMMA = 0.999
LAM = 0.99


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _make_policy(num_envs: int, seed: int = 0, obs_normalization: bool = False) -> ActorCriticTrialMemory:
    torch.manual_seed(seed)
    sample_obs = TensorDict({"policy": torch.zeros(num_envs, OBS_DIM)}, batch_size=[num_envs], device=DEVICE)
    policy = ActorCriticTrialMemory(
        obs=sample_obs,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=ACTION_DIM,
        actor_obs_normalization=obs_normalization,
        d_model=32,
        num_layers=2,
        num_heads=4,
        num_memory_tokens=NUM_MEMORY,
        max_episode_length=T_EPISODE,
        ff_mult=2,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        init_noise_std=0.5,
    )
    # Move the LayerNorms/heads off their initialization so the tests are not measuring a near-identity network.
    # The action-noise parameter is left alone (a negative std would produce NaN log-probs).
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if name in ("std", "log_std"):
                continue
            parameter.add_(0.1 * torch.randn_like(parameter))
    return policy


def _aligned_schedule(num_envs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """K episodes of T_EPISODE steps, all environments in lockstep. Returns (dones, trial_dones, time_outs)."""
    dones = torch.zeros(T_TRIAL, num_envs, dtype=torch.bool, device=DEVICE)
    trial_dones = torch.zeros_like(dones)
    for episode in range(1, K + 1):
        dones[episode * T_EPISODE - 1] = True
    trial_dones[T_TRIAL - 1] = True
    return dones, trial_dones, dones.clone()


def _collect(
    ppo: PPO,
    num_envs: int,
    dones: torch.Tensor,
    trial_dones: torch.Tensor,
    time_outs: torch.Tensor,
    seed: int = 0,
) -> None:
    """Run a full rollout through ``PPO.act`` / ``PPO.process_env_step`` and compute returns."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator, device=DEVICE)  # noqa: E731
    obs = TensorDict({"policy": randn(num_envs, OBS_DIM)}, batch_size=[num_envs], device=DEVICE)
    with torch.no_grad():
        for step in range(dones.shape[0]):
            ppo.act(obs)
            # The observation returned by step() is already the post-reset one for terminated environments.
            next_obs = TensorDict({"policy": randn(num_envs, OBS_DIM)}, batch_size=[num_envs], device=DEVICE)
            extras = {"time_outs": time_outs[step], "trial_done": trial_dones[step]}
            ppo.process_env_step(next_obs, randn(num_envs), dones[step], extras)
            obs = next_obs
        ppo.compute_returns(obs)


def _make_ppo(policy: ActorCriticTrialMemory, num_envs: int, num_steps: int, **kwargs) -> PPO:
    defaults = dict(
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=0.0,  # keep the parameters frozen so every minibatch sees the acting parameters
        schedule="fixed",
        desired_kl=None,
        gamma=GAMMA,
        lam=LAM,
        device=DEVICE,
        defer_obs_normalization=True,
    )
    defaults.update(kwargs)
    ppo = PPO(policy, **defaults)
    sample_obs = TensorDict({"policy": torch.zeros(num_envs, OBS_DIM)}, batch_size=[num_envs], device=DEVICE)
    ppo.init_storage("rl", num_envs, num_steps, sample_obs, [ACTION_DIM])
    return ppo


# --------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------


def test_epoch_zero_canary() -> None:
    """With unchanged parameters, the recomputed log-probs must equal the stored behavior log-probs."""
    num_envs = 4
    for normalization, label in ((False, "no obs normalization"), (True, "deferred obs normalization")):
        policy = _make_policy(num_envs, obs_normalization=normalization)
        ppo = _make_ppo(policy, num_envs, T_TRIAL, num_mini_batches=2)
        dones, trial_dones, time_outs = _aligned_schedule(num_envs)
        _collect(ppo, num_envs, dones, trial_dones, time_outs)

        stored_log_prob = ppo.storage.actions_log_prob.clone()
        parameters_before = [parameter.detach().clone() for parameter in policy.parameters()]

        # Reconstruct exactly the way PPO does, and compare pair by pair against the stored numbers.
        max_error = 0.0
        num_checked = 0
        index = ppo.storage.build_trial_pairs()
        for batch in ppo.storage.trial_pair_mini_batch_generator(policy, num_mini_batches=2, num_epochs=1):
            hidden, _ = ppo.trial_pair_forward(batch)
            distribution = policy.update_distribution_from_hidden(hidden)
            target = batch["target"]
            log_prob = distribution.log_prob(target["actions"]).sum(dim=-1)
            reference = target["old_actions_log_prob"].squeeze(-1)
            mask = target["loss_mask"]
            max_error = max(max_error, (log_prob[mask] - reference[mask]).abs().max().item())
            num_checked += int(mask.sum().item())
        assert num_checked == num_envs * T_TRIAL, f"only {num_checked} of {num_envs * T_TRIAL} steps reconstructed"
        assert max_error < 1e-4, f"reconstructed log-probs drifted by {max_error:.3e} ({label})"

        # ... and the same statement through the real update, read off the ratio diagnostics.
        loss_dict = ppo.update()
        assert abs(loss_dict["ratio_mean"] - 1.0) < 1e-4, f"ratio_mean = {loss_dict['ratio_mean']}"
        assert abs(loss_dict["ratio_min"] - 1.0) < 1e-3 and abs(loss_dict["ratio_max"] - 1.0) < 1e-3, (
            f"ratio range [{loss_dict['ratio_min']}, {loss_dict['ratio_max']}] is not ~1 at epoch 0"
        )
        assert loss_dict["ratio_clip_frac"] == 0.0
        for before, after in zip(parameters_before, policy.parameters()):
            assert torch.equal(before, after), "lr = 0 should have frozen the parameters"
        assert stored_log_prob.abs().sum() > 0.0, "the stored log-probs are all zero; the test would be vacuous"
        print(
            f"[ok] epoch-0 canary ({label}): {index['num_pairs']} pairs, max |d logp| = {max_error:.3e},"
            f" ratio in [{loss_dict['ratio_min']:.6f}, {loss_dict['ratio_max']:.6f}]"
        )


def test_gradient_routing() -> None:
    """The target loss must reach the source trunk and the writer, and must NOT reach the episode before it."""
    num_envs = 1
    policy = _make_policy(num_envs, seed=3)
    ppo = _make_ppo(policy, num_envs, T_TRIAL)
    dones, trial_dones, time_outs = _aligned_schedule(num_envs)
    _collect(ppo, num_envs, dones, trial_dones, time_outs, seed=5)

    # Make the stored observations differentiable leaves: their gradients tell us exactly which episode's trunk
    # the loss reached. (Parameters are shared across episodes, so a parameter gradient could not distinguish.)
    # The rollout sits *after* the carry region, which is empty here but still offsets every row index.
    carry = ppo.storage.carry_steps
    observations = ppo.storage.observations["policy"]
    observations.requires_grad_(True)

    batch = next(ppo.storage.trial_pair_mini_batch_generator(policy, num_mini_batches=1, num_epochs=1))
    hidden, _ = ppo.trial_pair_forward(batch)

    # The pair (t2, t3): target is the third episode, source the second, and the first must be unreachable.
    column = int((batch["positions"] == 2).nonzero(as_tuple=False)[0, 0].item())
    assert batch["source_slots"].numel() == K - 1, "the degenerate pair must be the only one without a source"
    hidden[:, column].square().sum().backward()

    grad = observations.grad
    assert grad is not None, "no gradient reached the stored observations at all"
    episode_grad = [
        grad[carry + episode * T_EPISODE : carry + (episode + 1) * T_EPISODE].abs().sum().item()
        for episode in range(K)
    ]

    assert episode_grad[2] > 0.0, "the target episode's own trunk received no gradient"
    assert episode_grad[1] > 0.0, "the loss did not reach the SOURCE episode's trunk through the memory writer"
    assert episode_grad[0] == 0.0, (
        f"gradient leaked through the stopgrad Zbar_e into episode e-1 ({episode_grad[0]:.3e}); the memory"
        " checkpoint is not detached"
    )
    assert policy.writer.attn.q_proj.weight.grad is not None
    assert policy.writer.attn.q_proj.weight.grad.abs().sum().item() > 0.0, "no gradient reached the memory writer"
    assert policy.z_init.grad is not None and policy.z_init.grad.abs().sum().item() >= 0.0

    # A degenerate pair trains the learned Z_init instead of a writer input.
    # Note: a fresh batch, because the gathers of the previous one were freed by the backward above.
    policy.zero_grad(set_to_none=True)
    observations.grad = None
    batch = next(ppo.storage.trial_pair_mini_batch_generator(policy, num_mini_batches=1, num_epochs=1))
    hidden, _ = ppo.trial_pair_forward(batch)
    first_column = int((batch["positions"] == 0).nonzero(as_tuple=False)[0, 0].item())
    hidden[:, first_column].square().sum().backward()
    assert policy.z_init.grad is not None and policy.z_init.grad.abs().sum().item() > 0.0, (
        "the degenerate (None, t1) pair must still train Z_init"
    )
    assert observations.grad[carry + T_EPISODE :].abs().sum().item() == 0.0, (
        "the first episode's loss reached later episodes"
    )

    print(
        f"[ok] gradient routing: |g| per episode = "
        f"{[f'{value:.3e}' for value in episode_grad]} (episode 0 is exactly zero)"
    )


def test_every_episode_is_a_target_exactly_once() -> None:
    """Over one epoch every valid episode takes exactly one PPO loss -- never zero, never two."""
    num_envs = 2
    policy = _make_policy(num_envs, seed=7)
    ppo = _make_ppo(policy, num_envs, T_TRIAL, num_mini_batches=3)
    dones, trial_dones, time_outs = _aligned_schedule(num_envs)
    _collect(ppo, num_envs, dones, trial_dones, time_outs, seed=11)

    index = ppo.storage.build_trial_pairs()
    assert index["num_pairs"] == num_envs * K, f"expected {num_envs * K} pairs, got {index['num_pairs']}"
    assert index["dropped_episodes"] == 0 and index["dropped_trials"] == 0

    losses_per_episode = torch.zeros(index["num_pairs"], dtype=torch.long, device=DEVICE)
    source_uses = torch.zeros(index["num_pairs"], dtype=torch.long, device=DEVICE)
    for batch in ppo.storage.trial_pair_mini_batch_generator(policy, num_mini_batches=3, num_epochs=1):
        losses_per_episode[batch["episode_ids"]] += 1
        source_uses[batch["episode_ids"][batch["source_slots"]] - 1] += 1
        # Every step of the target episode is covered by the loss mask exactly once
        assert bool(batch["target"]["loss_mask"].sum(dim=0).eq(T_EPISODE).all()), "targets are not full episodes"

    assert torch.equal(losses_per_episode, torch.ones_like(losses_per_episode)), (
        f"episodes taking != 1 loss: {(losses_per_episode != 1).nonzero(as_tuple=False).flatten().tolist()}"
    )
    # The last episode of a trial is never a source; every other episode is a source exactly once.
    positions = index["ep_pos"]
    expected_sources = (positions < K - 1).long()
    assert torch.equal(source_uses, expected_sources), (
        f"source usage {source_uses.tolist()} != expected {expected_sources.tolist()}"
    )
    print(f"[ok] {index['num_pairs']} episodes, each a target exactly once and a source at most once")


def test_desynchronized_environments() -> None:
    """A genuine terminal resets one environment early; pairs must still be grouped per trial, not per index."""
    num_envs = 2
    dones = torch.zeros(T_TRIAL, num_envs, dtype=torch.bool, device=DEVICE)
    trial_dones = torch.zeros_like(dones)
    time_outs = torch.zeros_like(dones)

    # env 0: the nominal K = 3 x 6-step trial
    for episode in range(1, K + 1):
        dones[episode * T_EPISODE - 1, 0] = True
    trial_dones[T_TRIAL - 1, 0] = True
    time_outs[:, 0] = dones[:, 0]

    # env 1: an "abnormal_robot" terminal at step 1 (a real terminal, so no timeout bootstrap) ends the trial
    # early; the trial that starts at step 2 is still open when the rollout ends and must be DEFERRED -- kept
    # in the buffer for the next rollout, not thrown away.
    dones[1, 1] = True
    trial_dones[1, 1] = True
    dones[7, 1] = True
    dones[13, 1] = True

    policy = _make_policy(num_envs, seed=13)
    ppo = _make_ppo(policy, num_envs, T_TRIAL)
    carry = ppo.storage.carry_steps
    assert carry == T_TRIAL, "the trial-memory path must reserve a carry region by default"
    _collect(ppo, num_envs, dones, trial_dones, time_outs, seed=17)

    index = ppo.storage.build_trial_pairs()
    assert index["num_pairs"] == K + 1, f"expected {K + 1} pairs (env 0's trial + env 1's short trial)"
    assert index["dropped_episodes"] == 0 and index["dropped_trials"] == 0, (
        f"nothing may be dropped: {index['dropped_episodes']} eps / {index['dropped_trials']} trials"
    )
    assert index["deferred_trials"] == 1 and index["deferred_episodes"] == 3, (
        f"env 1's open trial must be deferred, got {index['deferred_trials']} trials"
    )
    assert index["envs_without_data"] == 0
    assert index["ep_env"].tolist() == [0, 0, 0, 1]
    assert index["ep_start"].tolist() == [carry, carry + T_EPISODE, carry + 2 * T_EPISODE, carry]
    assert index["ep_len"].tolist() == [T_EPISODE, T_EPISODE, T_EPISODE, 2]
    assert index["ep_pos"].tolist() == [0, 1, 2, 0], "env 1's short trial must start a fresh trial at position 0"
    assert index["ep_lag"].tolist() == [0, 0, 0, 0], "everything here was collected under the current policy"

    # The pairing must connect the right spans: the source of env 0's second episode is its first episode.
    batch = next(ppo.storage.trial_pair_mini_batch_generator(policy, num_mini_batches=1, num_epochs=1))
    column = int((batch["episode_ids"] == 1).nonzero(as_tuple=False)[0, 0].item())
    slot = int((batch["source_slots"] == column).nonzero(as_tuple=False)[0, 0].item())
    source_obs = batch["source"]["obs"]["policy"][:, slot]
    stored = ppo.storage.observations["policy"][carry : carry + T_EPISODE, 0]
    assert torch.equal(source_obs[:T_EPISODE], stored), "the source episode does not carry env 0's first episode"
    # Source sequences carry T + 1 tokens (the terminal token) whose observation slot is zeros by default.
    assert source_obs.shape[0] == T_EPISODE + 1
    assert torch.equal(source_obs[T_EPISODE], torch.zeros(OBS_DIM, device=DEVICE)), (
        "the terminal token must not carry the post-reset observation"
    )
    # The target episode carries acting steps only (no terminal token, it takes no loss).
    assert batch["target"]["obs"]["policy"].shape[0] == T_EPISODE

    # env 1's surviving episode is a degenerate pair with a 2-step target
    env1_column = int((batch["episode_ids"] == 3).nonzero(as_tuple=False)[0, 0].item())
    assert int(batch["positions"][env1_column].item()) == 0
    assert int(batch["target"]["loss_mask"][:, env1_column].sum().item()) == 2

    # ... and PPO still runs end to end on the ragged rollout
    deferred_obs = ppo.storage.observations["policy"][carry + 2 : carry + T_TRIAL, 1].clone()
    loss_dict = ppo.update()
    assert abs(loss_dict["ratio_mean"] - 1.0) < 1e-4, f"ratio_mean = {loss_dict['ratio_mean']} on a ragged rollout"
    assert loss_dict["pool_dropped_episodes"] == 0.0 and loss_dict["pool_envs_without_data"] == 0.0

    # After the update the storage carried env 1's open trial (steps 2..17) into the carry region, and env 0 --
    # which closed its trial exactly at the rollout end -- carried nothing.
    assert ppo.storage.carry_len.tolist() == [0, T_TRIAL - 2]
    assert ppo.storage.at_trial_start.tolist() == [True, True]
    kept = ppo.storage.observations["policy"][carry - (T_TRIAL - 2) : carry, 1]
    assert torch.equal(kept, deferred_obs), "the deferred trial's observations did not survive the roll-over"
    print("[ok] desynchronized environments: 4 pairs kept, env 1's open trial deferred (not dropped), spans correct")


if __name__ == "__main__":
    test_epoch_zero_canary()
    test_gradient_routing()
    test_every_episode_is_a_target_exactly_once()
    test_desynchronized_environments()
    print("all trial-pair PPO tests passed")
