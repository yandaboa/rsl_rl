# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end tests for the episode-context policy through the REAL (unmodified) PPO update.

Self-contained: pure torch, no Isaac Sim / SimulationApp needed. Run with::

    source /home/yandabao/miniforge3/etc/profile.d/conda.sh && conda activate patlab
    python -m pytest tests/test_episode_context_ppo.py -q

The load-bearing test is ``test_epoch_zero_canary``: with unchanged parameters, re-inferring a rollout from the
stored frames must reproduce the behavior log-probs, i.e. the PPO ratio must be 1. If that fails, the window
slicing is wrong and every advantage is being applied to a different policy than the one that acted.

``test_ppo_stays_byte_identical`` guards the premise of the whole exercise: ``ppo.py``, ``rollout_storage.py``
and the runner are untouched relative to feee047 (the commit that produced run 225077).
"""

from __future__ import annotations

import subprocess
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import EpisodeContextPPO
from rsl_rl.modules import ActorCriticEpisodeContext
from rsl_rl.storage import EpisodeContextRolloutStorage

DEVICE = "cpu"
OBS_DIM = 5
CRITIC_OBS_DIM = 7
ACTION_DIM = 4
NUM_ENVS = 6
T_EPISODE = 6
STEPS_PER_ENV = 8  # deliberately not a divisor of T: every window straddles episode boundaries
GAMMA = 0.99
LAM = 0.95
BASE_COMMIT = "feee047"


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
    seed: int = 0,
    context_length: int = T_EPISODE,
    obs_normalization: bool = False,
    noise_std_type: str = "scalar",
    critic_design: str = "privileged",
    detach_critic_trunk: bool = False,
) -> ActorCriticEpisodeContext:
    torch.manual_seed(seed)
    policy = ActorCriticEpisodeContext(
        obs=_sample_obs(NUM_ENVS),
        obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=ACTION_DIM,
        actor_obs_normalization=obs_normalization,
        critic_obs_normalization=obs_normalization,
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
        critic_design=critic_design,
        detach_critic_trunk=detach_critic_trunk,
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
# Tests
# --------------------------------------------------------------------------------------------------


def test_epoch_zero_canary() -> None:
    """With unchanged parameters, the re-inferred log-probs (and values) must equal the stored ones."""
    dones = _episode_schedule(3 * STEPS_PER_ENV)
    for normalization, label in ((False, "no obs normalization"), (True, "deferred obs normalization")):
        policy = _make_policy(obs_normalization=normalization, noise_std_type="gsde")
        ppo = _make_ppo(policy)
        generator = torch.Generator(device=DEVICE).manual_seed(4)

        # Three consecutive rollouts: the first has no history at all, the later ones re-infer their window
        # from a prefix that was collected in a PREVIOUS rollout -- the case the ring buffer exists for.
        for rollout in range(3):
            _collect(ppo, dones, rollout * STEPS_PER_ENV, generator)
            stored_log_prob = ppo.storage.actions_log_prob.clone()
            stored_values = ppo.storage.values.clone()
            stored_actions = ppo.storage.actions.clone()
            parameters_before = [parameter.detach().clone() for parameter in policy.parameters()]

            # Reconstruct exactly the way PPO does, straight off the generator it consumes.
            max_error = 0.0
            max_value_error = 0.0
            rows = 0
            for batch in ppo.storage.recurrent_mini_batch_generator(num_mini_batches=2, num_epochs=1):
                obs_batch, actions_batch = batch[0], batch[1]
                prefix = batch[8][0]
                policy.act(obs_batch, masks=batch[9], hidden_state=prefix)
                log_prob = policy.get_actions_log_prob(actions_batch)
                reference = batch[5].squeeze(-1)
                max_error = max(max_error, (log_prob - reference).abs().max().item())
                value = policy.evaluate(obs_batch)
                max_value_error = max(max_value_error, (value - batch[2]).abs().max().item())
                rows += int(log_prob.numel())
            assert rows == STEPS_PER_ENV * NUM_ENVS, f"{rows} rows re-inferred, expected {STEPS_PER_ENV * NUM_ENVS}"
            assert max_error < 1e-5, f"reconstructed log-probs drifted by {max_error:.3e} ({label})"
            assert max_value_error < 1e-6, f"reconstructed values drifted by {max_value_error:.3e} ({label})"

            # ... and the same statement through the real, unmodified PPO update.
            loss_dict = ppo.update()
            assert abs(loss_dict["ratio_mean"] - 1.0) < 1e-5, f"ratio_mean = {loss_dict['ratio_mean']} ({label})"
            assert abs(loss_dict["ratio_min"] - 1.0) < 1e-4 and abs(loss_dict["ratio_max"] - 1.0) < 1e-4, (
                f"ratio range [{loss_dict['ratio_min']}, {loss_dict['ratio_max']}] is not ~1 at epoch 0 ({label})"
            )
            assert loss_dict["ratio_clip_frac"] == 0.0
            for key, value in loss_dict.items():
                assert value == value and abs(value) < float("inf"), f"{key} is not finite: {value}"
            # (``max_value_error`` above is the value-side canary: the context-free critic re-evaluates to the
            # stored values exactly. The value LOSS is against the returns, so it is legitimately non-zero.)
            for before, after in zip(parameters_before, policy.parameters()):
                assert torch.equal(before, after), "lr = 0 should have frozen the parameters"
            assert stored_log_prob.abs().sum() > 0.0, "the stored log-probs are all zero; the test would be vacuous"
            assert stored_actions.abs().sum() > 0.0
            assert stored_values.abs().sum() > 0.0
            ppo.commit_obs_normalization(ppo.storage.collected_observations)
            print(
                f"[ok] epoch-0 canary ({label}), rollout {rollout}: max |d logp| = {max_error:.3e},"
                f" ratio in [{loss_dict['ratio_min']:.9f}, {loss_dict['ratio_max']:.9f}]"
            )


def test_prefix_actually_carries_history() -> None:
    """The canary would be vacuous if the window were context-free: the prefix must be non-empty and load-bearing."""
    dones = _episode_schedule(2 * STEPS_PER_ENV)
    policy = _make_policy(seed=2)
    ppo = _make_ppo(policy)
    generator = torch.Generator(device=DEVICE).manual_seed(9)
    _collect(ppo, dones, 0, generator)
    ppo.update()
    _collect(ppo, dones, STEPS_PER_ENV, generator)

    storage: EpisodeContextRolloutStorage = ppo.storage
    assert storage.prefix_length == policy.context_prefix_length > 0
    batch = next(storage.recurrent_mini_batch_generator(num_mini_batches=2, num_epochs=1))
    prefix = batch[8][0]
    assert prefix.obs.shape == (storage.prefix_length, NUM_ENVS // 2, OBS_DIM)
    assert prefix.window_positions.shape == (STEPS_PER_ENV, NUM_ENVS // 2)
    # Some window row must genuinely reach into the prefix (i.e. its episode started in the previous rollout).
    assert int(prefix.window_positions[0].max().item()) > 0

    with torch.no_grad():
        reference = policy.act(batch[0], hidden_state=prefix).clone()
        wrecked = type(prefix)(
            obs=prefix.obs + 5.0, positions=prefix.positions, window_positions=prefix.window_positions
        )
        perturbed = policy.act(batch[0], hidden_state=wrecked)
    assert not torch.allclose(reference, perturbed), "corrupting the prefix changed nothing; it is not being read"
    print("[ok] the prefix is non-empty, reaches into the previous rollout and is read by the trunk")


def test_every_row_is_trained_exactly_once() -> None:
    """One epoch must cover every (step, environment) row exactly once, with no padding and no drops."""
    dones = _episode_schedule(STEPS_PER_ENV)
    policy = _make_policy(seed=5)
    ppo = _make_ppo(policy, num_mini_batches=3)
    generator = torch.Generator(device=DEVICE).manual_seed(1)
    _collect(ppo, dones, 0, generator)

    seen = torch.zeros(STEPS_PER_ENV, NUM_ENVS, device=DEVICE)
    for batch in ppo.storage.recurrent_mini_batch_generator(num_mini_batches=3, num_epochs=1):
        actions = batch[1]
        assert actions.shape[0] == STEPS_PER_ENV
        # Locate the environment chunk by matching the stored actions (the generator hands out slices).
        for column in range(actions.shape[1]):
            matches = (ppo.storage.actions == actions[:, column : column + 1]).all(dim=-1).all(dim=0)
            found = matches.nonzero(as_tuple=False)
            assert found.numel() == 1, "could not uniquely locate a minibatch column in the storage"
            seen[:, int(found.item())] += 1.0
    assert torch.equal(seen, torch.ones_like(seen)), f"row coverage is not exactly one: {seen}"
    print(f"[ok] every one of {STEPS_PER_ENV * NUM_ENVS} rows is trained exactly once per epoch")


def test_shared_trunk_epoch_zero_value_canary() -> None:
    """With a shared trunk, epoch-0 values re-inferred INSIDE the real PPO update must equal the stored ones.

    Same statement as the ratio canary, on the value side: the acting path reads ``h_t`` off the KV cache while
    the update reads it off one batched ``[prefix | window]`` pass, so if the two disagreed every advantage would
    be measured against a value the behavior policy never produced. The values are captured by wrapping
    ``policy.evaluate`` -- ``ppo.py`` itself is untouched and is what calls it.
    """
    dones = _episode_schedule(2 * STEPS_PER_ENV)
    policy = _make_policy(critic_design="shared_trunk", noise_std_type="gsde")
    assert policy.critic[0].in_features == policy.d_model
    ppo = _make_ppo(policy)
    generator = torch.Generator(device=DEVICE).manual_seed(13)

    for rollout in range(2):
        _collect(ppo, dones, rollout * STEPS_PER_ENV, generator)
        stored_values = ppo.storage.values.clone()

        # Record what the REAL update path feeds into the value loss, in minibatch order.
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
        re_inferred = torch.cat(recorded, dim=1)  # the generator hands out contiguous environment chunks
        assert re_inferred.shape == stored_values.shape
        value_error = (re_inferred - stored_values).abs().max().item()
        assert value_error < 1e-5, f"epoch-0 values drifted by {value_error:.3e} (rollout {rollout})"
        assert stored_values.abs().max().item() > 1e-4, "the stored values are ~0; the canary would be vacuous"
        # The ratio canary must still hold with the actor and critic sharing one trunk pass.
        assert abs(loss_dict["ratio_mean"] - 1.0) < 1e-5, f"ratio_mean = {loss_dict['ratio_mean']}"
        for key, value in loss_dict.items():
            assert value == value and abs(value) < float("inf"), f"{key} is not finite: {value}"
        ppo.commit_obs_normalization(ppo.storage.collected_observations)
        print(f"[ok] shared-trunk epoch-0 value canary, rollout {rollout}: max |dV| = {value_error:.3e}")


def test_shared_trunk_bootstrap_value_is_not_committed() -> None:
    """``compute_returns`` evaluates a frame that never went through ``act()``; it must peek, not commit.

    If the bootstrap frame entered the KV cache, the next rollout's first token would attend to a duplicate of
    itself and the epoch-0 reconstruction would break -- which is exactly what the canary above would then fail
    on, so this test pins the mechanism directly.
    """
    dones = _episode_schedule(STEPS_PER_ENV)
    policy = _make_policy(seed=6, critic_design="shared_trunk")
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
        cache_before = [cache.clone() for cache in policy._key_cache]
        cache_positions_before = policy._cache_positions.clone()
        ppo.compute_returns(obs)

    assert torch.equal(policy.positions, positions_before), "the bootstrap advanced the acting path's episode step"
    assert torch.equal(policy._cache_positions, cache_positions_before)
    for before, after in zip(cache_before, policy._key_cache):
        assert torch.equal(before, after), "the bootstrap frame was committed to the KV cache"
    assert ppo.storage.returns.abs().sum().item() > 0.0
    print("[ok] the shared-trunk bootstrap value peeks the last frame without committing it")


def test_shared_trunk_value_loss_trains_the_trunk_through_ppo() -> None:
    """Through the real ``update()``: the trunk moves under the value loss iff ``detach_critic_trunk=False``.

    The actor's own losses are switched off (zero advantages -> zero surrogate gradient, zero entropy
    coefficient) so that any trunk movement can only have come through the value head.
    """
    dones = _episode_schedule(STEPS_PER_ENV)
    for detach in (False, True):
        policy = _make_policy(seed=8, critic_design="shared_trunk", detach_critic_trunk=detach)
        ppo = _make_ppo(policy, learning_rate=1e-2, value_loss_coef=1.0, entropy_coef=0.0)
        generator = torch.Generator(device=DEVICE).manual_seed(2)
        _collect(ppo, dones, 0, generator)
        ppo.storage.advantages.zero_()  # d surrogate / d ratio = -advantage = 0
        trunk_before = {
            name: parameter.detach().clone()
            for name, parameter in policy.named_parameters()
            if name.startswith(("token_embed", "blocks", "final_norm", "pos_embed", "start_embed"))
        }
        ppo.update()

        moved = max(
            (parameter - trunk_before[name]).abs().max().item()
            for name, parameter in policy.named_parameters()
            if name in trunk_before
        )
        head_moved = max(
            parameter.grad.abs().max().item()
            for name, parameter in policy.named_parameters()
            if name.startswith("critic.") and parameter.grad is not None
        )
        assert head_moved > 0.0, "the value loss never reached the value head"
        if detach:
            assert moved == 0.0, f"detach_critic_trunk=True moved the trunk by {moved:.3e}"
        else:
            assert moved > 0.0, "detach_critic_trunk=False left the trunk untouched by the value loss"
        print(f"[ok] real PPO update, detach_critic_trunk={detach}: max |d trunk| = {moved:.3e}")


def test_ppo_stays_byte_identical() -> None:
    """The known-good PPO, storage and runner must be untouched: all new behavior lives in new files."""
    protected = [
        "rsl_rl/algorithms/ppo.py",
        "rsl_rl/storage/rollout_storage.py",
        "rsl_rl/runners/on_policy_runner.py",
    ]
    diff = subprocess.run(
        ["git", "diff", "--stat", BASE_COMMIT, "--", *protected],
        capture_output=True,
        text=True,
        cwd=__file__.rsplit("/tests/", 1)[0],
    )
    assert diff.returncode == 0, f"git diff failed: {diff.stderr}"
    assert diff.stdout.strip() == "", f"protected files changed since {BASE_COMMIT}:\n{diff.stdout}"
    print(f"[ok] ppo.py / rollout_storage.py / on_policy_runner.py are byte-identical to {BASE_COMMIT}")


if __name__ == "__main__":
    test_epoch_zero_canary()
    test_prefix_actually_carries_history()
    test_every_row_is_trained_exactly_once()
    test_shared_trunk_epoch_zero_value_canary()
    test_shared_trunk_bootstrap_value_is_not_committed()
    test_shared_trunk_value_loss_trains_the_trunk_through_ppo()
    test_ppo_stays_byte_identical()
    print("all episode-context PPO tests passed")
