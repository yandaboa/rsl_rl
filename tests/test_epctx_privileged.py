# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Privileged encoder on the episode-context trunk (``privileged_group``).

Self-contained: pure torch, no Isaac Sim. Run with::

    source /home/yandabao/miniforge3/etc/profile.d/conda.sh && conda activate patlab
    python -m pytest tests/test_epctx_privileged.py -q

Four statements: ``privileged_group=None`` is the unchanged policy (same state-dict keys, same outputs), a
privileged policy acts EXACTLY like the non-privileged one at init (zero projection), a no-privileged
checkpoint loads with ``strict=False`` and misses only the ``privileged_*`` tensors, and the encoder receives
gradient after one loss step.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import EpisodeContextPPO
from rsl_rl.modules import ActorCriticEpisodeContext

DEVICE = "cpu"
OBS_DIM = 5
PRIV_DIM = 6
CRITIC_OBS_DIM = 7
ACTION_DIM = 4
NUM_ENVS = 8
T_EPISODE = 6
STEPS_PER_ENV = 8


def _sample_obs(num_envs: int, generator: torch.Generator | None = None) -> TensorDict:
    def draw(dim: int) -> torch.Tensor:
        if generator is None:
            return torch.zeros(num_envs, dim, device=DEVICE)
        return torch.randn(num_envs, dim, generator=generator, device=DEVICE)

    return TensorDict(
        {"policy": draw(OBS_DIM), "privileged": draw(PRIV_DIM), "critic": draw(CRITIC_OBS_DIM)},
        batch_size=[num_envs],
        device=DEVICE,
    )


def _make_policy(seed: int = 0, privileged: bool = True) -> ActorCriticEpisodeContext:
    torch.manual_seed(seed)
    policy = ActorCriticEpisodeContext(
        obs=_sample_obs(NUM_ENVS),
        obs_groups={"policy": ["policy", "privileged"] if privileged else ["policy"], "critic": ["critic"]},
        num_actions=ACTION_DIM,
        actor_obs_normalization=True,
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
        privileged_group="privileged" if privileged else None,
        privileged_encoder_hidden_dims=[16, 8],
        privileged_embed_dim=4,
    )
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if name in ("std", "log_std") or name.startswith("privileged_"):
                continue
            parameter.add_(0.1 * torch.randn_like(parameter))
    return policy


def _priv_keys(policy: ActorCriticEpisodeContext) -> set[str]:
    return {k for k in policy.state_dict() if k.startswith("privileged_")}


def test_no_privileged_group_is_unchanged():
    """Without the group nothing is built: no ``privileged_*`` tensor, no change to any width."""
    policy = _make_policy(privileged=False)
    assert policy.privileged_dim == 0
    assert _priv_keys(policy) == set()
    assert policy.num_token_obs == policy.num_actor_obs == OBS_DIM
    assert policy.actor_obs_normalizer._mean.shape[-1] == OBS_DIM
    # The frame normalizer handed to the storage is the plain observation normalizer, as before.
    assert policy.frame_normalizer is policy.actor_obs_normalizer

    generator = torch.Generator(device=DEVICE).manual_seed(3)
    obs = _sample_obs(NUM_ENVS, generator)
    policy.reset()
    assert policy.act_inference(obs).shape == (NUM_ENVS, ACTION_DIM)


def test_zero_projection_reproduces_the_non_privileged_policy():
    """``privileged_proj`` is zero at init, so the privileged policy IS the BC policy on the same frames."""
    plain = _make_policy(seed=1, privileged=False)
    priv = _make_policy(seed=1, privileged=True)
    assert priv.privileged_dim == PRIV_DIM
    assert priv.num_token_obs == OBS_DIM
    assert priv.actor_obs_normalizer._mean.shape[-1] == OBS_DIM
    assert priv.privileged_normalizer._mean.shape[-1] == PRIV_DIM
    # Same shapes for everything a BC checkpoint carries -> the plain state dict loads as-is.
    priv.load_state_dict(plain.state_dict(), strict=False)

    generator = torch.Generator(device=DEVICE).manual_seed(4)
    plain.reset()
    priv.reset()
    for _ in range(T_EPISODE):
        obs = _sample_obs(NUM_ENVS, generator)
        expected = plain.act_inference(obs)
        # A different privileged draw must not move the actions either.
        obs["privileged"] = 10.0 * torch.randn(NUM_ENVS, PRIV_DIM, generator=generator, device=DEVICE)
        torch.testing.assert_close(priv.act_inference(obs), expected, rtol=0.0, atol=0.0)

    # Batched path too (what the PPO update runs).
    sequence = TensorDict(
        {
            "policy": torch.randn(NUM_ENVS, T_EPISODE, OBS_DIM, generator=generator),
            "privileged": torch.randn(NUM_ENVS, T_EPISODE, PRIV_DIM, generator=generator),
        },
        batch_size=[NUM_ENVS, T_EPISODE],
        device=DEVICE,
    )
    plain_hidden = plain.forward_window(sequence.select("policy"))
    priv_hidden = priv.forward_window(sequence)
    torch.testing.assert_close(priv_hidden, plain_hidden, rtol=0.0, atol=0.0)


def test_bc_checkpoint_loads_with_only_privileged_keys_missing():
    plain = _make_policy(seed=2, privileged=False)
    priv = _make_policy(seed=5, privileged=True)
    # ``ActorCriticEpisodeContext.load_state_dict`` returns a bool, so go through nn.Module for the report.
    incompatible = torch.nn.Module.load_state_dict(priv, plain.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == _priv_keys(priv)
    assert incompatible.unexpected_keys == []
    assert _priv_keys(priv) == {
        "privileged_normalizer._mean",
        "privileged_normalizer._var",
        "privileged_normalizer._std",
        "privileged_normalizer.count",
        "privileged_encoder.0.weight",
        "privileged_encoder.0.bias",
        "privileged_encoder.2.weight",
        "privileged_encoder.2.bias",
        "privileged_encoder.4.weight",
        "privileged_encoder.4.bias",
        "privileged_proj.weight",
        "privileged_proj.bias",
    }


def test_gradient_reaches_the_privileged_encoder():
    """One PPO-shaped rollout + update: the projection moves, and the encoder is trained on the next step."""
    policy = _make_policy(seed=6, privileged=True)
    ppo = EpisodeContextPPO(
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        learning_rate=1e-3,
        schedule="fixed",
        desired_kl=None,
        device=DEVICE,
    )
    ppo.init_storage("rl", NUM_ENVS, STEPS_PER_ENV, _sample_obs(NUM_ENVS), [ACTION_DIM])
    generator = torch.Generator(device=DEVICE).manual_seed(7)

    def collect() -> None:
        obs = _sample_obs(NUM_ENVS, generator)
        with torch.no_grad():
            for step in range(STEPS_PER_ENV):
                ppo.act(obs)
                obs = _sample_obs(NUM_ENVS, generator)
                rewards = torch.randn(NUM_ENVS, generator=generator, device=DEVICE)
                dones = torch.full((NUM_ENVS,), (step + 1) % T_EPISODE == 0, dtype=torch.bool, device=DEVICE)
                ppo.process_env_step(obs, rewards, dones, {})
            # ``defer_obs_normalization`` is on by default: the runner commits after the update, not per step.
            ppo.commit_obs_normalization(obs)
            ppo.compute_returns(obs)

    grads: dict[str, torch.Tensor] = {}
    handles = [
        policy.privileged_encoder[0].weight.register_hook(lambda g: grads.__setitem__("encoder", g.detach().clone())),
        policy.privileged_proj.weight.register_hook(lambda g: grads.__setitem__("proj", g.detach().clone())),
    ]
    before = policy.privileged_proj.weight.detach().clone()
    collect()
    ppo.update()
    assert "proj" in grads and grads["proj"].abs().sum() > 0.0, "no gradient reached privileged_proj"
    assert not torch.equal(policy.privileged_proj.weight.detach(), before), "privileged_proj did not move"
    assert policy.privileged_normalizer.count > 0, "the privileged normalizer never saw a frame"
    assert int(policy.actor_obs_normalizer.count) == int(policy.privileged_normalizer.count)

    # The encoder only sees gradient once the projection is non-zero, which the first step made it.
    grads.clear()
    collect()
    ppo.update()
    handles = [handle.remove() for handle in handles]
    assert "encoder" in grads and grads["encoder"].abs().sum() > 0.0, "no gradient reached privileged_encoder"
