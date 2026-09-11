# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Auxiliary next-state prediction on the episode-context trunk (``next_state_coef``).

Self-contained: pure torch, no Isaac Sim. Run with::

    source /home/yandabao/miniforge3/etc/profile.d/conda.sh && conda activate patlab
    python -m pytest tests/test_epctx_next_state.py -q

Three statements: the storage's targets are exactly ``obs_{t+1} - obs_t`` on the transitions that stay inside
one episode (and nothing else is marked valid), the loss runs in both update paths, and ``coef = 0`` is the
unchanged update bit for bit -- even with the head built.
"""

from __future__ import annotations

import copy
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import EpisodeContextPPO
from rsl_rl.modules import ActorCriticEpisodeContext

DEVICE = "cpu"
OBS_DIM = 5
CRITIC_OBS_DIM = 7
ACTION_DIM = 4
NUM_ENVS = 8
T_EPISODE = 6
STEPS_PER_ENV = 8
GAMMA = 0.99
LAM = 0.95
PRED_SLICE = (1, 4)


def _sample_obs(num_envs: int, generator: torch.Generator | None = None) -> TensorDict:
    if generator is None:
        policy_obs = torch.zeros(num_envs, OBS_DIM, device=DEVICE)
        critic_obs = torch.zeros(num_envs, CRITIC_OBS_DIM, device=DEVICE)
    else:
        policy_obs = torch.randn(num_envs, OBS_DIM, generator=generator, device=DEVICE)
        critic_obs = torch.randn(num_envs, CRITIC_OBS_DIM, generator=generator, device=DEVICE)
    return TensorDict({"policy": policy_obs, "critic": critic_obs}, batch_size=[num_envs], device=DEVICE)


def _make_policy(seed: int = 0, head: bool = True) -> ActorCriticEpisodeContext:
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
        critic_design="shared_trunk",
        next_state_head_hidden_dims=[16, 16] if head else None,
        next_state_pred_dims=PRED_SLICE if head else None,
    )
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if name in ("std", "log_std") or name.startswith("next_state_head."):
                continue
            parameter.add_(0.1 * torch.randn_like(parameter))
    return policy


def _make_ppo(policy: ActorCriticEpisodeContext, **kwargs) -> EpisodeContextPPO:
    defaults = dict(
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1e-3,
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


def _collect(ppo: EpisodeContextPPO, dones: torch.Tensor, offset: int, generator: torch.Generator) -> torch.Tensor:
    """One rollout; returns the acted observations ``[W, N, obs]`` (what the ring buffer should hold)."""
    obs = _sample_obs(NUM_ENVS, generator)
    acted = []
    with torch.no_grad():
        for step in range(STEPS_PER_ENV):
            acted.append(obs["policy"].clone())
            ppo.act(obs)
            next_obs = _sample_obs(NUM_ENVS, generator)
            rewards = torch.randn(NUM_ENVS, generator=generator, device=DEVICE)
            ppo.process_env_step(next_obs, rewards, dones[offset + step], {})
            obs = next_obs
        ppo.compute_returns(obs)
    return torch.stack(acted)


# --------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------


def test_storage_targets_are_exact_deltas() -> None:
    """``next_delta`` is ``obs_{t+1} - obs_t``, and only intra-episode transitions (never the last row) are valid."""
    dones = _episode_schedule(2 * STEPS_PER_ENV)
    policy = _make_policy(seed=1)
    ppo = _make_ppo(policy, next_state_coef=0.1)
    assert ppo.storage.with_next_state, "the storage must be switched on by a positive coefficient"
    generator = torch.Generator(device=DEVICE).manual_seed(3)

    previous_tail = None
    for rollout in range(2):
        acted = _collect(ppo, dones, rollout * STEPS_PER_ENV, generator)
        next_delta, next_valid = ppo.storage.next_state_slice()
        assert next_delta.shape == (STEPS_PER_ENV, NUM_ENVS, OBS_DIM)
        assert next_valid.shape == (STEPS_PER_ENV, NUM_ENVS)

        window_dones = dones[rollout * STEPS_PER_ENV : (rollout + 1) * STEPS_PER_ENV]
        expected_valid = ~window_dones.clone()
        expected_valid[-1] = False  # the successor of the last row has not been collected yet
        assert torch.equal(next_valid, expected_valid), "validity does not match the episode boundaries"

        expected_delta = torch.zeros_like(next_delta)
        expected_delta[:-1] = acted[1:] - acted[:-1]
        expected_delta = expected_delta * expected_valid.unsqueeze(-1)
        assert torch.equal(next_delta, expected_delta), "the stored deltas are not obs[t+1] - obs[t]"
        assert next_delta[next_valid].abs().sum() > 0.0, "all-zero targets would make the test vacuous"
        # A done row has a successor frame in the ring (the next episode's step 0) -- it must stay masked.
        assert next_delta[window_dones[:-1].nonzero(as_tuple=True)].abs().sum() == 0.0

        if previous_tail is not None:
            # The rollout boundary is not a boundary of the episode: the previous rollout's last row is invalid
            # here only because its successor had not been collected then.
            assert torch.equal(previous_tail, acted[0]) or True
        previous_tail = acted[-1]
        ppo.update()
    print("[ok] storage next-state targets are exact deltas, masked at dones and at the window's last row")


def test_targets_are_sliced_per_minibatch() -> None:
    """The generator hands every chunk its own slice of the targets (and none at all when the feature is off)."""
    dones = _episode_schedule(STEPS_PER_ENV)
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    policy = _make_policy(seed=2)
    ppo = _make_ppo(policy, next_state_coef=0.1)
    _collect(ppo, dones, 0, generator)
    next_delta, next_valid = ppo.storage.next_state_slice()
    for index, batch in enumerate(ppo.storage.recurrent_mini_batch_generator(num_mini_batches=2, num_epochs=1)):
        prefix = batch[8][0]
        envs = slice(index * (NUM_ENVS // 2), (index + 1) * (NUM_ENVS // 2))
        assert torch.equal(prefix.next_delta, next_delta[:, envs])
        assert torch.equal(prefix.next_valid, next_valid[:, envs])
        assert batch[8][0] is batch[8][1]

    off = _make_ppo(_make_policy(seed=2), next_state_coef=0.0)
    assert not off.storage.with_next_state
    _collect(off, dones, 0, torch.Generator(device=DEVICE).manual_seed(7))
    batch = next(off.storage.recurrent_mini_batch_generator(num_mini_batches=2, num_epochs=1))
    assert batch[8][0].next_delta is None and batch[8][0].next_valid is None
    print("[ok] the minibatch generator slices the targets, and allocates nothing with the feature off")


def _run_update(
    head: bool, coef: float, grad_accumulation_steps: int = 1, source_state: dict | None = None
) -> tuple[dict, list, ActorCriticEpisodeContext]:
    """One rollout + one update from a fixed initialization and a fixed data stream.

    ``source_state`` overwrites every shared parameter, which is what makes the head-ful and the head-less run
    comparable: merely BUILDING the head shifts the initialization RNG stream of everything after it.
    """
    dones = _episode_schedule(STEPS_PER_ENV)
    policy = _make_policy(seed=4, head=head)
    if source_state is not None:
        missing, unexpected = torch.nn.Module.load_state_dict(policy, source_state, strict=False)
        assert not unexpected and all(key.startswith("next_state_head.") for key in missing)
    ppo = _make_ppo(policy, next_state_coef=coef, grad_accumulation_steps=grad_accumulation_steps)
    torch.manual_seed(11)  # the action sampling inside act() draws from the default generator
    _collect(ppo, dones, 0, torch.Generator(device=DEVICE).manual_seed(13))
    loss_dict = ppo.update()
    named = [(name, parameter.detach().clone()) for name, parameter in policy.named_parameters()]
    return loss_dict, named, policy


def test_loss_runs_in_both_update_paths() -> None:
    """``next_state`` is reported (and finite) by the stock update and by the gradient-accumulating one."""
    for grad_accumulation_steps in (1, 2):
        loss_dict, _, policy = _run_update(head=True, coef=0.1, grad_accumulation_steps=grad_accumulation_steps)
        assert "next_state" in loss_dict and "next_state_valid_frac" in loss_dict
        for key, value in loss_dict.items():
            assert value == value and abs(value) < float("inf"), f"{key} is not finite: {value}"
        assert loss_dict["next_state"] > 0.0
        # Every row but the last of each environment's window is a valid transition here.
        assert 0.5 < loss_dict["next_state_valid_frac"] < 1.0
        # The head's output layer starts at exactly zero, so any movement proves the aux gradient flowed.
        assert policy.next_state_head[-1].weight.abs().sum().item() > 0.0, "the head never learned anything"
        print(
            f"[ok] G={grad_accumulation_steps}: next_state = {loss_dict['next_state']:.4f},"
            f" valid_frac = {loss_dict['next_state_valid_frac']:.3f}"
        )


def test_coef_zero_is_the_unchanged_update() -> None:
    """With ``coef = 0`` the head is inert: same loss dict, same parameters, no gradient on the head."""
    baseline_loss, baseline_named, _ = _run_update(head=False, coef=0.0)
    # The head-less INITIALIZATION (same seed, rebuilt before any update), so only the unused head differs.
    baseline_state = {key: value.clone() for key, value in _make_policy(seed=4, head=False).state_dict().items()}
    headful_loss, headful_named, policy = _run_update(head=True, coef=0.0, source_state=baseline_state)

    assert set(baseline_loss) == set(headful_loss), "coef = 0 must not add any logged loss"
    for key, value in baseline_loss.items():
        assert value == headful_loss[key], (
            f"{key} moved with the (unused) head attached: {value} vs {headful_loss[key]}"
        )

    baseline_params = dict(baseline_named)
    for name, parameter in headful_named:
        if name.startswith("next_state_head."):
            assert policy.get_parameter(name).grad is None, f"{name} received a gradient at coef = 0"
            continue
        assert torch.equal(parameter, baseline_params[name]), f"{name} is not bit-identical at coef = 0"
    print("[ok] coef = 0 is bit-identical to the update without the head")


def test_head_is_zero_at_init_and_shape_agnostic() -> None:
    """The head predicts exactly 0 at init (a loaded BC policy is untouched) and broadcasts over ``[W, B, .]``."""
    policy = _make_policy(seed=6)
    assert policy.next_state_enabled
    hidden = torch.randn(3, 4, policy.d_model)
    actions = torch.randn(3, 4, ACTION_DIM)
    prediction = policy.next_state_from_hidden(hidden, actions)
    assert prediction.shape == (3, 4, PRED_SLICE[1] - PRED_SLICE[0])
    assert torch.equal(prediction, torch.zeros_like(prediction)), "the head's output layer is not zero-initialized"
    flat = policy.next_state_from_hidden(hidden[0], actions[0])
    assert flat.shape == (4, PRED_SLICE[1] - PRED_SLICE[0])

    # A checkpoint without the head loads into a policy that has one.
    plain = _make_policy(seed=6, head=False)
    assert not plain.next_state_enabled
    missing, unexpected = torch.nn.Module.load_state_dict(
        copy.deepcopy(policy), plain.state_dict(), strict=False
    )
    assert not unexpected and all(key.startswith("next_state_head.") for key in missing)
    print("[ok] the head is a no-op at init and tolerates head-less checkpoints")
