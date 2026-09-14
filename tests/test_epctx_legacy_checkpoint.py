# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint compatibility of the ported episode-context model.

* ``EpisodeContextPPO.save()`` / ``load()`` round-trip, the legacy ``model_state_dict`` fallback, the
  ``strict=False`` zero-pad path and the ``load_cfg`` prefix filter -- all self-contained.
* Numerical parity with the rsl-rl 3.1 ``ActorCriticEpisodeContext``: a subprocess running the ``stable-ppo``
  worktree builds a policy, rolls it through both forward paths and dumps its ``state_dict`` + outputs; the
  5.2 model loads that dict unchanged and must reproduce every number. Skipped when that worktree is absent.
"""

from __future__ import annotations

import os
import pytest
import subprocess
import sys
import tempfile
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import EpisodeContextPPO
from rsl_rl.models import EpisodeContextModel

DEVICE = "cpu"
OBS_DIM = 5
CRITIC_OBS_DIM = 7
ACTION_DIM = 4
NUM_ENVS = 4
T_EPISODE = 6
STEPS_PER_ENV = 8
LEGACY_REPO = "/home/yandabao/rsl_rl/.claude/worktrees/stable-ppo-225077"

MODEL_KWARGS = dict(
    context_length=T_EPISODE,
    d_model=32,
    num_layers=2,
    num_heads=4,
    max_episode_length=T_EPISODE,
    ff_mult=2,
    actor_hidden_dims=[16],
    critic_hidden_dims=[16],
    init_noise_std=0.5,
)


def _sample_obs(num_envs: int, generator: torch.Generator | None = None) -> TensorDict:
    if generator is None:
        policy_obs = torch.zeros(num_envs, OBS_DIM, device=DEVICE)
        critic_obs = torch.zeros(num_envs, CRITIC_OBS_DIM, device=DEVICE)
    else:
        policy_obs = torch.randn(num_envs, OBS_DIM, generator=generator, device=DEVICE)
        critic_obs = torch.randn(num_envs, CRITIC_OBS_DIM, generator=generator, device=DEVICE)
    return TensorDict({"policy": policy_obs, "critic": critic_obs}, batch_size=[num_envs], device=DEVICE)


def _make_model(seed: int = 0, **overrides) -> EpisodeContextModel:
    torch.manual_seed(seed)
    kwargs = dict(MODEL_KWARGS, **overrides)
    model = EpisodeContextModel(
        obs=_sample_obs(NUM_ENVS),
        obs_groups={"actor": ["policy"], "critic": ["critic"]},
        num_actions=ACTION_DIM,
        **kwargs,
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in ("std", "log_std"):
                continue
            parameter.add_(0.1 * torch.randn_like(parameter))
    return model


def _make_ppo(model: EpisodeContextModel, **kwargs) -> EpisodeContextPPO:
    defaults = dict(num_learning_epochs=1, num_mini_batches=2, learning_rate=1e-3, schedule="fixed", desired_kl=None)
    defaults.update(kwargs)
    return EpisodeContextPPO.create(
        model, NUM_ENVS, STEPS_PER_ENV, _sample_obs(NUM_ENVS), [ACTION_DIM], device=DEVICE, **defaults
    )


def _same_state(a: EpisodeContextModel, b: EpisodeContextModel) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[key], sb[key]) for key in sa)


# --------------------------------------------------------------------------------------------------
# save / load round trips
# --------------------------------------------------------------------------------------------------


def test_save_load_round_trip_and_aliases() -> None:
    """``actor_state_dict`` is the whole joint model, ``critic_state_dict`` aliases its ``critic*`` keys."""
    source = _make_ppo(_make_model(seed=1, critic_design="separate_trunk", memory_tokens=2))
    saved = source.save()
    assert set(saved["actor_state_dict"]) == set(source.model.state_dict())
    assert saved["critic_state_dict"] and all(key.startswith("critic") for key in saved["critic_state_dict"])
    assert {key for key in saved["actor_state_dict"] if key.startswith("critic")} == set(saved["critic_state_dict"])

    target = _make_ppo(_make_model(seed=2, critic_design="separate_trunk", memory_tokens=2))
    assert not _same_state(source.model, target.model)
    assert target.load(saved, None, strict=True) is True
    assert _same_state(source.model, target.model)
    # The runner reads the policy through the view; it is the same joint module.
    assert target.get_policy().model is target.model
    print("[ok] save()/load() round trip; critic_state_dict aliases the critic* keys")


def test_legacy_model_state_dict_loads() -> None:
    """A 3.1 checkpoint (``model_state_dict``) maps onto the joint model; the remap is logged."""
    source = _make_model(seed=3, noise_std_type="gsde", critic_design="shared_trunk")
    legacy = {"model_state_dict": source.state_dict(), "optimizer_state_dict": {}, "iter": 7, "infos": None}
    target = _make_ppo(_make_model(seed=4, noise_std_type="gsde", critic_design="shared_trunk"))
    assert not _same_state(source, target.model)
    assert target.load(legacy, {"actor": True, "critic": True, "iteration": True}, strict=True) is True
    assert _same_state(source, target.model)
    print("[ok] legacy model_state_dict loads into the joint model")


def test_load_cfg_prefix_filter() -> None:
    """``load_cfg={"actor": True}`` leaves the critic pathway untouched, and vice versa."""
    source = _make_ppo(_make_model(seed=5, critic_design="separate_trunk")).save()
    for load_actor, load_critic in ((True, False), (False, True)):
        target = _make_ppo(_make_model(seed=6, critic_design="separate_trunk"))
        before = {key: value.clone() for key, value in target.model.state_dict().items()}
        target.load(source, {"actor": load_actor, "critic": load_critic}, strict=True)
        after = target.model.state_dict()
        for key in after:
            loaded = key.startswith("critic") if load_critic else not key.startswith("critic")
            expected = source["actor_state_dict"][key] if loaded else before[key]
            assert torch.equal(after[key], expected), f"{key}: {'not loaded' if loaded else 'overwritten'}"
    print("[ok] load_cfg selects the actor / critic key prefixes")


def test_non_strict_load_zero_pads_mismatched_shapes() -> None:
    """``strict=False`` zero-pads / truncates mismatched tensors and skips the optimizer state."""
    narrow = _make_ppo(_make_model(seed=7, actor_hidden_dims=[16]))
    wide = _make_ppo(_make_model(seed=8, actor_hidden_dims=[24]))
    saved = narrow.save()
    optimizer_before = wide.optimizer.state_dict()["param_groups"][0]["lr"]
    assert wide.load(saved, None, strict=False) is True
    # Overlap copied, the rest zero.
    weight = wide.model.actor[0].weight
    assert torch.equal(weight[:16], narrow.model.actor[0].weight)
    assert torch.equal(weight[16:], torch.zeros_like(weight[16:]))
    head = wide.model.actor[2].weight
    assert torch.equal(head[:, :16], narrow.model.actor[2].weight)
    assert torch.equal(head[:, 16:], torch.zeros_like(head[:, 16:]))
    # Every other tensor loaded exactly.
    assert torch.equal(wide.model.blocks[0].attn.q_proj.weight, narrow.model.blocks[0].attn.q_proj.weight)
    assert wide.optimizer.state_dict()["param_groups"][0]["lr"] == optimizer_before
    print("[ok] strict=False pads mismatched shapes with zeros")


def test_noise_type_mismatch_is_refused() -> None:
    saved = _make_ppo(_make_model(seed=9, noise_std_type="gsde")).save()
    target = _make_ppo(_make_model(seed=9, noise_std_type="scalar"))
    with pytest.raises(ValueError, match="noise_std_type"):
        target.load(saved, None, strict=True)
    print("[ok] a gsde checkpoint is refused by a scalar-noise policy")


# --------------------------------------------------------------------------------------------------
# Numerical parity with the 3.1 implementation
# --------------------------------------------------------------------------------------------------

_LEGACY_SCRIPT = r"""
import sys, torch
from tensordict import TensorDict
from rsl_rl.modules.actor_critic_episode_context import ActorCriticEpisodeContext, EpisodeContextPrefix

out_path, noise_std_type, critic_design, memory_tokens = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
OBS_DIM, CRITIC_OBS_DIM, ACTION_DIM, NUM_ENVS, T = 5, 7, 4, 4, 6
STEPS = 8
torch.manual_seed(0)
obs0 = TensorDict(
    {"policy": torch.zeros(NUM_ENVS, OBS_DIM), "critic": torch.zeros(NUM_ENVS, CRITIC_OBS_DIM)}, batch_size=[NUM_ENVS]
)
policy = ActorCriticEpisodeContext(
    obs=obs0, obs_groups={"policy": ["policy"], "critic": ["critic"]}, num_actions=ACTION_DIM,
    context_length=T, d_model=32, num_layers=2, num_heads=4, max_episode_length=T, ff_mult=2,
    actor_hidden_dims=[16], critic_hidden_dims=[16], init_noise_std=0.5,
    noise_std_type=noise_std_type, critic_design=critic_design, memory_tokens=memory_tokens, episodes_per_trial=2,
)
with torch.no_grad():
    for name, p in policy.named_parameters():
        if name in ("std", "log_std"):
            continue
        p.add_(0.1 * torch.randn_like(p))
    for name, p in policy.named_parameters():
        if name in ("std", "log_std"):
            p.add_(0.05 * torch.randn_like(p))
policy = policy.double().eval()
g = torch.Generator().manual_seed(11)
frames = torch.randn(STEPS, NUM_ENVS, OBS_DIM, generator=g, dtype=torch.float64)
critic_frames = torch.randn(STEPS, NUM_ENVS, CRITIC_OBS_DIM, generator=g, dtype=torch.float64)
actions = torch.randn(STEPS, NUM_ENVS, ACTION_DIM, generator=g, dtype=torch.float64)
dones = torch.zeros(STEPS, NUM_ENVS, dtype=torch.bool)
for env in range(NUM_ENVS):
    dones[T - 1 - env, env] = True
trial_dones = torch.zeros_like(dones)
means, stds, log_probs, values, entropies = [], [], [], [], []
with torch.no_grad():
    policy.initialize_state(NUM_ENVS, "cpu", dtype=torch.float64)
    for step in range(STEPS):
        obs = TensorDict({"policy": frames[step], "critic": critic_frames[step]}, batch_size=[NUM_ENVS])
        policy.act(obs)
        means.append(policy.action_mean.clone()); stds.append(policy.action_std.clone())
        log_probs.append(policy.get_actions_log_prob(actions[step]).clone())
        entropies.append(policy.entropy.clone())
        values.append(policy.evaluate(obs).clone())
        policy.reset(dones[step], trial_dones=trial_dones[step])
    # batched path over the whole run (positions from the dones)
    positions = torch.zeros(STEPS, NUM_ENVS, dtype=torch.long)
    step_counter = torch.zeros(NUM_ENVS, dtype=torch.long)
    for t in range(STEPS):
        positions[t] = step_counter
        step_counter = (step_counter + 1) * (~dones[t]).long()
    prefix = EpisodeContextPrefix(obs=frames[:0], positions=positions[:0], window_positions=positions)
    hidden = policy.forward_window(frames, prefix=prefix)
    policy.update_distribution_from_hidden(hidden)
    window_log_prob = policy.get_actions_log_prob(actions).clone()
torch.save({
    "model_state_dict": policy.state_dict(),
    "frames": frames, "critic_frames": critic_frames, "actions": actions, "dones": dones,
    "means": torch.stack(means), "stds": torch.stack(stds), "log_probs": torch.stack(log_probs),
    "entropies": torch.stack(entropies), "values": torch.stack(values), "window_log_prob": window_log_prob,
}, out_path)
"""


@pytest.mark.skipif(not os.path.isdir(LEGACY_REPO), reason="the rsl-rl 3.1 stable-ppo worktree is not available")
@pytest.mark.parametrize(
    "noise_std_type, critic_design, memory_tokens",
    [
        ("scalar", "privileged", 0),
        ("log", "shared_trunk", 0),
        ("gsde", "shared_trunk", 0),
        ("gsde", "separate_trunk", 2),
    ],
)
def test_numerics_match_the_31_implementation(noise_std_type: str, critic_design: str, memory_tokens: int) -> None:
    """The 3.1 ``ActorCriticEpisodeContext`` state dict loads unchanged and every number is reproduced."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "legacy.pt")
        env = dict(os.environ, PYTHONPATH=LEGACY_REPO)
        result = subprocess.run(
            [sys.executable, "-c", _LEGACY_SCRIPT, out_path, noise_std_type, critic_design, str(memory_tokens)],
            capture_output=True,
            text=True,
            env=env,
            cwd=tmp,
        )
        assert result.returncode == 0, f"legacy script failed:\n{result.stderr[-3000:]}"
        legacy = torch.load(out_path, map_location="cpu", weights_only=False)

    torch.manual_seed(123)
    model = (
        EpisodeContextModel(
            obs=_sample_obs(NUM_ENVS),
            obs_groups={"actor": ["policy"], "critic": ["critic"]},
            num_actions=ACTION_DIM,
            noise_std_type=noise_std_type,
            critic_design=critic_design,
            memory_tokens=memory_tokens,
            episodes_per_trial=2,
            **MODEL_KWARGS,
        )
        .double()
        .eval()
    )
    assert set(model.state_dict()) == set(legacy["model_state_dict"]), "the parameter names diverged from 3.1"
    assert model.load_state_dict(legacy["model_state_dict"], strict=True) is True

    frames, critic_frames, actions, dones = (
        legacy["frames"],
        legacy["critic_frames"],
        legacy["actions"],
        legacy["dones"],
    )
    means, stds, log_probs, values, entropies = [], [], [], [], []
    with torch.no_grad():
        model.initialize_state(NUM_ENVS, DEVICE, dtype=torch.float64)
        for step in range(STEPS_PER_ENV):
            obs = TensorDict({"policy": frames[step], "critic": critic_frames[step]}, batch_size=[NUM_ENVS])
            model.act(obs)
            means.append(model.action_mean.clone())
            stds.append(model.action_std.clone())
            log_probs.append(model.get_actions_log_prob(actions[step]).clone())
            entropies.append(model.entropy.clone())
            values.append(model.evaluate(obs).clone())
            model.reset(dones[step], trial_dones=torch.zeros_like(dones[step]))
        positions = torch.zeros(STEPS_PER_ENV, NUM_ENVS, dtype=torch.long)
        step_counter = torch.zeros(NUM_ENVS, dtype=torch.long)
        for t in range(STEPS_PER_ENV):
            positions[t] = step_counter
            step_counter = (step_counter + 1) * (~dones[t]).long()
        from rsl_rl.models import EpisodeContextPrefix

        prefix = EpisodeContextPrefix(obs=frames[:0], positions=positions[:0], window_positions=positions)
        model.update_distribution_from_hidden(model.forward_window(frames, prefix=prefix))
        window_log_prob = model.get_actions_log_prob(actions)

    for name, ours, theirs in (
        ("action mean", torch.stack(means), legacy["means"]),
        ("action std", torch.stack(stds), legacy["stds"]),
        ("log-prob", torch.stack(log_probs), legacy["log_probs"]),
        ("entropy", torch.stack(entropies), legacy["entropies"]),
        ("value", torch.stack(values), legacy["values"]),
        ("window log-prob", window_log_prob, legacy["window_log_prob"]),
    ):
        error = (ours - theirs).abs().max().item()
        assert error < 1e-10, f"{name} differs from the 3.1 implementation by {error:.3e}"
        assert theirs.abs().max().item() > 1e-6, f"{name}: the reference is ~0; the comparison would be vacuous"
    print(f"[ok] {noise_std_type}/{critic_design}/M={memory_tokens}: bit-level parity with the 3.1 implementation")


if __name__ == "__main__":
    test_save_load_round_trip_and_aliases()
    test_legacy_model_state_dict_loads()
    test_load_cfg_prefix_filter()
    test_non_strict_load_zero_pads_mismatched_shapes()
    test_noise_type_mismatch_is_refused()
    for case in (
        ("scalar", "privileged", 0),
        ("log", "shared_trunk", 0),
        ("gsde", "shared_trunk", 0),
        ("gsde", "separate_trunk", 2),
    ):
        test_numerics_match_the_31_implementation(*case)
    print("all legacy-checkpoint tests passed")
