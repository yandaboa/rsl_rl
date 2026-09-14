# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``num_loops`` (weight-shared looped trunk) and ``loop_input_injection`` on :class:`EpisodeContextModel`.

Pure torch, no Isaac. Run with::

    source /home/yandabao/miniforge3/etc/profile.d/conda.sh && conda activate patlab3
    python -m pytest tests/test_epctx_num_loops.py -q
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models import EpisodeContextModel

from tests.test_episode_context_policy import (
    ACTION_DIM,
    DEVICE,
    NUM_ENVS,
    OBS_DIM,
    T_EPISODE,
    _positions_from_dones,
    _roll_incrementally,
    _window_hidden,
)


def _make(seed: int = 0, **kwargs) -> EpisodeContextModel:
    torch.manual_seed(seed)
    sample_obs = TensorDict({"policy": torch.zeros(NUM_ENVS, OBS_DIM)}, batch_size=[NUM_ENVS], device=DEVICE)
    policy = EpisodeContextModel(
        obs=sample_obs,
        obs_groups={"actor": ["policy"], "critic": ["policy"]},
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
        **kwargs,
    )
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if name in ("std", "log_std"):
                continue
            parameter.add_(0.1 * torch.randn_like(parameter))
    return policy.double().eval()


def _rollout() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_steps = 3 * T_EPISODE
    generator = torch.Generator(device=DEVICE).manual_seed(11)
    obs = torch.randn(num_steps, NUM_ENVS, OBS_DIM, generator=generator, dtype=torch.float64, device=DEVICE)
    dones = torch.zeros(num_steps, NUM_ENVS, dtype=torch.bool, device=DEVICE)
    for env in range(NUM_ENVS):
        step = T_EPISODE - 1 - env
        while step < num_steps:
            dones[step, env] = True
            step += T_EPISODE
    return obs, dones, _positions_from_dones(dones)


def test_num_loops_one_is_baseline() -> None:
    """``num_loops=1`` (the default) must be byte-identical to a model built without the kwarg."""
    obs, dones, _ = _rollout()
    base = _make()
    looped = _make(num_loops=1, loop_input_injection=True)  # injection is a no-op with a single pass
    assert base.depth == base.num_layers == 2 and looped.depth == 2
    for (name_a, p_a), (name_b, p_b) in zip(base.named_parameters(), looped.named_parameters()):
        assert name_a == name_b and torch.equal(p_a, p_b), name_a
    with torch.no_grad():
        h_base = _roll_incrementally(base, obs, dones)
        h_loop = _roll_incrementally(looped, obs, dones)
    assert torch.equal(h_base, h_loop)


def test_num_loops_three_runs_and_differs() -> None:
    """Looping shares weights (same parameter count), deepens the trunk, changes the output, and keeps the
    acting path (per-depth KV caches) equal to the batched window path."""
    obs, dones, positions = _rollout()
    base = _make()
    looped = _make(num_loops=3)
    assert looped.depth == 6 and len(looped.blocks) == 2
    assert sum(p.numel() for p in looped.parameters()) == sum(p.numel() for p in base.parameters())
    with torch.no_grad():
        h_base = _roll_incrementally(base, obs, dones)
        h_loop = _roll_incrementally(looped, obs, dones)
        assert len(looped._key_cache) == 6 and len(looped._value_cache) == 6
        assert (h_base - h_loop).abs().max().item() > 1e-3
        # Acting path == update path with the looped depth.
        window = _window_hidden(looped, obs, positions, window_start=2 * T_EPISODE - 2)
    error = (h_loop[2 * T_EPISODE - 2 :] - window).abs().max().item()
    assert error < 1e-10, f"step vs window mismatch with num_loops=3: {error:.3e}"
    # Distribution / value heads work on the looped readout.
    with torch.no_grad():
        looped.initialize_state(NUM_ENVS, DEVICE, dtype=torch.float64)
        actions = looped.act(TensorDict({"policy": obs[0]}, batch_size=[NUM_ENVS], device=DEVICE))
    assert actions.shape == (NUM_ENVS, ACTION_DIM) and torch.isfinite(actions).all()


def test_loop_input_injection_toggle() -> None:
    """Injection re-adds the trunk input at the start of every pass after the first: same parameters, different
    output; the acting and window paths still agree."""
    obs, dones, positions = _rollout()
    plain = _make(num_loops=2)
    injected = _make(num_loops=2, loop_input_injection=True)
    for (name_a, p_a), (_, p_b) in zip(plain.named_parameters(), injected.named_parameters()):
        assert torch.equal(p_a, p_b), name_a
    with torch.no_grad():
        h_plain = _roll_incrementally(plain, obs, dones)
        h_inj = _roll_incrementally(injected, obs, dones)
        window = _window_hidden(injected, obs, positions, window_start=2)
    assert (h_plain - h_inj).abs().max().item() > 1e-3
    error = (h_inj[2:] - window).abs().max().item()
    assert error < 1e-10, f"step vs window mismatch with loop_input_injection: {error:.3e}"


def test_num_loops_validation() -> None:
    try:
        _make(num_loops=0)
    except AssertionError:
        return
    raise AssertionError("num_loops=0 must be rejected")
