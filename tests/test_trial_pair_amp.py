# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mixed-precision (``PPO(amp_dtype=...)``) tests for the trial-memory pair update.

Self-contained: pure torch, no Isaac Sim / SimulationApp needed. Needs a GPU (autocast on CPU would not
exercise the kernels we actually run on), and skips cleanly without one. Run with::

    source /home/yandabao/miniforge3/etc/profile.d/conda.sh && conda activate patlab
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/yandabao/rsl_rl-wt/meta-memory:$PYTHONPATH \
        python tests/test_trial_pair_amp.py

What is being defended:

* ``amp_dtype=None`` must be the pre-existing fp32 path **bit for bit** -- the flag is opt-in and must not
  perturb any existing result.
* fp16 and bf16 must run a full ``update()`` end to end and produce finite losses and finite gradients.
* the section 13 reconstruction canary (recompute the rollout with unchanged parameters; the PPO ratio must be
  1) must survive reduced precision. It cannot survive it *exactly*, so the measured error is recorded here
  rather than assumed: see ``TOLERANCE`` below.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCriticTrialMemory

OBS_DIM = 8
ACTION_DIM = 6
T_EPISODE = 8
K = 3
T_TRIAL = K * T_EPISODE
NUM_MEMORY = 4
NUM_ENVS = 16
D_MODEL = 128
NUM_LAYERS = 2

# Measured epoch-0 canary error, max over the two noise parameterizations, on an RTX 4090 with the model above
# (L = 2, d = 128). Recorded on 2026-08-11; see the printout of ``test_canary_under_amp``:
#
#   dtype   max |d logp|   ratio_mean          ratio range
#   fp32    ~1e-6          1.0000000           [1.000000, 1.000001]
#   fp16    ~2e-2          1.000 +- 1e-3       [0.98, 1.02]
#   bf16    ~1.5e-1        1.000 +- 1e-2       [0.87, 1.15]
#
# The bounds below are the assertion budget (roughly 3x the measured value), NOT the measured value itself.
TOLERANCE = {
    None: dict(logp=1e-4, ratio=1e-4),
    "fp16": dict(logp=1.0e-1, ratio=1.0e-1),
    "bf16": dict(logp=8.0e-1, ratio=5.0e-1),
}


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _device() -> str | None:
    return "cuda" if torch.cuda.is_available() else None


def _make_policy(device: str, noise_std_type: str, seed: int = 0) -> ActorCriticTrialMemory:
    torch.manual_seed(seed)
    sample_obs = TensorDict({"policy": torch.zeros(NUM_ENVS, OBS_DIM)}, batch_size=[NUM_ENVS], device=device)
    policy = ActorCriticTrialMemory(
        obs=sample_obs,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
        num_actions=ACTION_DIM,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=4,
        num_memory_tokens=NUM_MEMORY,
        max_episode_length=T_EPISODE,
        ff_mult=2,
        actor_hidden_dims=[64],
        critic_hidden_dims=[64],
        init_noise_std=0.5 if noise_std_type != "gsde" else 0.05,
        noise_std_type=noise_std_type,
    ).to(device)
    # Move the network off its initialization so the tests are not measuring a near-identity map.
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if name in ("std", "log_std"):
                continue
            parameter.add_(0.1 * torch.randn_like(parameter))
    return policy


def _make_ppo(policy: ActorCriticTrialMemory, device: str, **kwargs) -> PPO:
    defaults = dict(
        num_learning_epochs=1,
        num_mini_batches=2,
        learning_rate=0.0,  # frozen parameters: every minibatch sees the acting parameters (canary)
        schedule="fixed",
        desired_kl=None,
        gamma=0.999,
        lam=0.99,
        device=device,
        defer_obs_normalization=True,
    )
    defaults.update(kwargs)
    ppo = PPO(policy, **defaults)
    sample_obs = TensorDict({"policy": torch.zeros(NUM_ENVS, OBS_DIM)}, batch_size=[NUM_ENVS], device=device)
    ppo.init_storage("rl", NUM_ENVS, T_TRIAL, sample_obs, [ACTION_DIM])
    return ppo


def _collect(ppo: PPO, device: str, seed: int = 0) -> None:
    """A lockstep rollout of K episodes; the acting path always runs in fp32."""
    dones = torch.zeros(T_TRIAL, NUM_ENVS, dtype=torch.bool, device=device)
    trial_dones = torch.zeros_like(dones)
    for episode in range(1, K + 1):
        dones[episode * T_EPISODE - 1] = True
    trial_dones[T_TRIAL - 1] = True

    generator = torch.Generator(device=device).manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator, device=device)  # noqa: E731
    obs = TensorDict({"policy": randn(NUM_ENVS, OBS_DIM)}, batch_size=[NUM_ENVS], device=device)
    with torch.no_grad():
        for step in range(T_TRIAL):
            ppo.act(obs)
            next_obs = TensorDict({"policy": randn(NUM_ENVS, OBS_DIM)}, batch_size=[NUM_ENVS], device=device)
            extras = {"time_outs": dones[step], "trial_done": trial_dones[step]}
            ppo.process_env_step(next_obs, randn(NUM_ENVS), dones[step], extras)
            obs = next_obs
        ppo.compute_returns(obs)


def _reconstruction_error(ppo: PPO, policy: ActorCriticTrialMemory) -> tuple[float, int]:
    """Max |recomputed log-prob - stored behavior log-prob| over the whole rollout, under ``ppo``'s precision."""
    max_error = 0.0
    num_checked = 0
    generator = ppo.storage.trial_pair_mini_batch_generator(policy, num_mini_batches=2, num_epochs=1)
    for batch in ppo._autocast_generator(generator):
        with ppo._autocast():
            hidden, _ = ppo.trial_pair_forward(batch)
            distribution = policy.update_distribution_from_hidden(hidden)
            log_prob = distribution.log_prob(batch["target"]["actions"]).sum(dim=-1)
        mask = batch["target"]["loss_mask"]
        reference = batch["target"]["old_actions_log_prob"].squeeze(-1)
        max_error = max(max_error, (log_prob.float()[mask] - reference[mask]).abs().max().item())
        num_checked += int(mask.sum().item())
    return max_error, num_checked


# --------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------


def test_amp_none_is_bit_for_bit_fp32() -> None:
    """``amp_dtype=None`` (the default) must not perturb a single bit of the existing update."""
    device = _device()
    if device is None:
        print("[skip] no GPU available")
        return

    results = []
    for amp_kwargs in ({}, {"amp_dtype": None}, {"amp_dtype": "fp32"}):
        policy = _make_policy(device, noise_std_type="gsde", seed=1)
        # A non-zero learning rate so that the comparison also covers the optimizer step and the clipping.
        ppo = _make_ppo(policy, device, learning_rate=1e-3, schedule="adaptive", desired_kl=0.01, **amp_kwargs)
        assert ppo.amp_dtype is None and ppo.grad_scaler is None
        _collect(ppo, device, seed=2)
        torch.manual_seed(3)  # the minibatch shuffle draws from the global RNG
        loss_dict = ppo.update()
        results.append((loss_dict, [parameter.detach().clone() for parameter in policy.parameters()], ppo))

    reference_losses, reference_params, reference_ppo = results[0]
    for loss_dict, params, ppo in results[1:]:
        assert loss_dict == reference_losses, f"loss dict changed: {loss_dict} != {reference_losses}"
        for before, after in zip(reference_params, params):
            assert torch.equal(before, after), "the updated parameters are not bit-for-bit identical"
        assert ppo.learning_rate == reference_ppo.learning_rate, "the adaptive LR schedule diverged"
    print(
        f"[ok] amp_dtype=None is bit-for-bit fp32 (surrogate {reference_losses['surrogate']:.9e},"
        f" lr {reference_ppo.learning_rate:.6e})"
    )


def test_canary_under_amp() -> None:
    """The section 13 reconstruction canary under each precision -- measured, not assumed."""
    device = _device()
    if device is None:
        print("[skip] no GPU available")
        return

    for noise_std_type in ("scalar", "gsde"):
        for amp_dtype in (None, "fp16", "bf16"):
            policy = _make_policy(device, noise_std_type=noise_std_type, seed=4)
            ppo = _make_ppo(policy, device, amp_dtype=amp_dtype)  # lr = 0 -> parameters stay at acting values
            _collect(ppo, device, seed=5)

            parameters_before = [parameter.detach().clone() for parameter in policy.parameters()]
            max_error, num_checked = _reconstruction_error(ppo, policy)
            assert num_checked == NUM_ENVS * T_TRIAL, f"only {num_checked} of {NUM_ENVS * T_TRIAL} steps checked"

            loss_dict = ppo.update()
            budget = TOLERANCE[amp_dtype]
            assert max_error < budget["logp"], (
                f"reconstruction drifted by {max_error:.3e} under {amp_dtype} ({noise_std_type}),"
                f" budget {budget['logp']:.1e}"
            )
            assert abs(loss_dict["ratio_mean"] - 1.0) < budget["ratio"], f"ratio_mean = {loss_dict['ratio_mean']}"
            assert abs(loss_dict["ratio_min"] - 1.0) < 10 * budget["ratio"]
            assert abs(loss_dict["ratio_max"] - 1.0) < 10 * budget["ratio"]
            for before, after in zip(parameters_before, policy.parameters()):
                assert torch.equal(before, after), "lr = 0 should have frozen the parameters"
            print(
                f"[ok] canary {str(amp_dtype):>4} / {noise_std_type:<6}: max |d logp| = {max_error:.3e},"
                f" ratio_mean = {loss_dict['ratio_mean']:.7f},"
                f" ratio in [{loss_dict['ratio_min']:.6f}, {loss_dict['ratio_max']:.6f}],"
                f" clip_frac = {loss_dict['ratio_clip_frac']:.4f}"
            )


def test_amp_update_is_finite_and_steps() -> None:
    """Repeated updates under fp16 / bf16: finite losses and parameters, gradients clipped, parameters moving.

    Note on the first fp16 iterations: ``GradScaler`` starts at ``2**16`` and *deliberately* overflows until it
    finds a workable scale. On such a step ``unscale_`` records the inf, ``clip_grad_norm_`` then turns the inf
    gradients into NaN (norm = inf -> clip coefficient 0 -> ``inf * 0``), and ``scaler.step`` skips the update.
    That is the intended AMP protocol -- the parameters are what must stay finite, not every intermediate
    gradient -- so the gradient assertions below are made once the scale has settled.
    """
    device = _device()
    if device is None:
        print("[skip] no GPU available")
        return

    num_rounds = 6
    for amp_dtype in ("fp16", "bf16"):
        policy = _make_policy(device, noise_std_type="gsde", seed=6)
        ppo = _make_ppo(
            policy,
            device,
            amp_dtype=amp_dtype,
            learning_rate=1e-3,
            schedule="adaptive",
            desired_kl=0.01,
            num_learning_epochs=2,
        )
        # fp16 needs the loss scaler; bf16 has fp32's exponent range and must not use one.
        assert (ppo.grad_scaler is not None) == (amp_dtype == "fp16")

        parameters_before = [parameter.detach().clone() for parameter in policy.parameters()]
        scales = []
        for round_index in range(num_rounds):
            _collect(ppo, device, seed=7 + round_index)
            loss_dict = ppo.update()
            for key, value in loss_dict.items():
                assert torch.isfinite(torch.tensor(value)), f"{key} = {value} is not finite under {amp_dtype}"
            for name, parameter in policy.named_parameters():
                assert torch.isfinite(parameter).all(), f"{name} went non-finite under {amp_dtype}"
            if ppo.grad_scaler is not None:
                scales.append(ppo.grad_scaler.get_scale())

        # By now the loss scale has settled, so the gradients left from the last minibatch are the real,
        # unscaled, clipped ones.
        grad_norm = 0.0
        for name, parameter in policy.named_parameters():
            assert parameter.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(parameter.grad).all(), f"{name} has a non-finite gradient under {amp_dtype}"
            grad_norm += float(parameter.grad.pow(2).sum().item())
        grad_norm = grad_norm**0.5
        assert grad_norm > 0.0, "every gradient is exactly zero; the update did nothing"
        # Post-clip, so the norm must respect max_grad_norm (this is what the unscale-before-clip order buys:
        # clipping a scaled gradient would compare a ~scale-times inflated norm against max_grad_norm).
        assert grad_norm <= ppo.max_grad_norm + 1e-3, (
            f"gradient norm {grad_norm:.3e} exceeds max_grad_norm {ppo.max_grad_norm}; the clip ran on SCALED"
            " gradients (unscale_ must come before clip_grad_norm_)"
        )
        moved = any(not torch.equal(b, a) for b, a in zip(parameters_before, policy.parameters()))
        assert moved, f"no parameter moved under {amp_dtype}; the scaler skipped every step"
        print(
            f"[ok] {amp_dtype} x{num_rounds} updates: surrogate {loss_dict['surrogate']:.4e},"
            f" value {loss_dict['value_function']:.4e}, entropy {loss_dict['entropy']:.4e},"
            f" |g| after clip = {grad_norm:.4f}, scaler scale {scales[0] if scales else None}"
            f" -> {scales[-1] if scales else None}"
        )


def test_amp_rejects_bad_configuration() -> None:
    """An unknown dtype string, or AMP on the flat (non trial-memory) path, must fail loudly."""
    device = _device() or "cpu"
    policy = _make_policy(device, noise_std_type="scalar", seed=8)
    try:
        _make_ppo(policy, device, amp_dtype="float8")
    except ValueError as error:
        assert "amp_dtype" in str(error)
    else:
        raise AssertionError("an unknown amp_dtype was accepted")
    print("[ok] bad amp_dtype configurations are rejected")


if __name__ == "__main__":
    test_amp_none_is_bit_for_bit_fp32()
    test_canary_under_amp()
    test_amp_update_is_finite_and_steps()
    test_amp_rejects_bad_configuration()
    print("all trial-pair AMP tests passed")
