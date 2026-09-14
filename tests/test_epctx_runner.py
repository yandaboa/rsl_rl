# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The episode-context algorithm through the STOCK :class:`OnPolicyRunner` (construct_algorithm, learn, save/load,
get_inference_policy). Pure torch, no Isaac."""

from __future__ import annotations

import os
import tempfile
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import EpisodeContextPPO
from rsl_rl.env import VecEnv
from rsl_rl.models import EpisodeContextActorView, EpisodeContextModel
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import EpisodeContextRolloutStorage

NUM_ENVS = 16
OBS_DIM = 5
CRITIC_OBS_DIM = 7
NUM_ACTIONS = 4
MAX_EP_LEN = 6


class DummyEnv(VecEnv):
    """Random observations / rewards; every env terminates after ``MAX_EP_LEN`` steps, staggered per env."""

    def __init__(self, device: str = "cpu") -> None:  # noqa: D107
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = MAX_EP_LEN
        self.episode_length_buf = torch.arange(NUM_ENVS, dtype=torch.long, device=device) % MAX_EP_LEN
        self.device = device
        self.cfg = {}

    def get_observations(self) -> TensorDict:  # noqa: D102
        return TensorDict(
            {
                "policy": torch.randn(self.num_envs, OBS_DIM, device=self.device),
                "critic": torch.randn(self.num_envs, CRITIC_OBS_DIM, device=self.device),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:  # noqa: D102
        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).long()
        self.episode_length_buf[dones.bool()] = 0
        rewards = torch.randn(self.num_envs, device=self.device)
        extras = {"time_outs": dones.float(), "episode": {"reward": torch.tensor(1.0)}}
        return self.get_observations(), rewards, dones, extras


def _train_cfg(**algorithm_overrides) -> dict:
    cfg: dict = {
        "num_steps_per_env": 8,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": {
            "class_name": "rsl_rl.models:EpisodeContextModel",
            "context_length": MAX_EP_LEN,
            "d_model": 32,
            "num_layers": 2,
            "num_heads": 4,
            "max_episode_length": MAX_EP_LEN,
            "ff_mult": 2,
            "hidden_dims": [16],
            "obs_normalization": True,
            "noise_std_type": "gsde",
            "init_noise_std": 0.5,
        },
        "critic": {
            "class_name": "rsl_rl.models:EpisodeContextModel",
            "hidden_dims": [16],
            "critic_design": "shared_trunk",
        },
        "algorithm": {
            "class_name": "rsl_rl.algorithms:EpisodeContextPPO",
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "learning_rate": 1e-3,
            "schedule": "adaptive",
            "desired_kl": 0.01,
            "rnd_cfg": None,
            "symmetry_cfg": None,
            **algorithm_overrides,
        },
    }
    return cfg


def test_runner_constructs_the_joint_model() -> None:
    runner = OnPolicyRunner(DummyEnv(), _train_cfg(), log_dir=None, device="cpu")
    alg = runner.alg
    assert isinstance(alg, EpisodeContextPPO)
    assert isinstance(alg.model, EpisodeContextModel)
    assert isinstance(alg.storage, EpisodeContextRolloutStorage)
    model = alg.model
    assert model.critic_design == "shared_trunk" and model.noise_std_type == "gsde"
    assert model.actor_obs_normalization and list(model.actor[0].weight.shape) == [16, 32]
    assert model.critic[0].in_features == model.d_model and model.critic[0].out_features == 16
    # The views partition the joint parameter set exactly.
    actor_names = {n for n, _ in alg.actor.named_parameters()}
    critic_names = {n for n, _ in alg.critic.named_parameters()}
    assert actor_names.isdisjoint(critic_names)
    assert actor_names | critic_names == {n for n, _ in model.named_parameters()}
    assert all(n.startswith("critic") for n in critic_names) and critic_names
    # One optimizer over the joint parameters, each exactly once.
    optimizer_params = [p for group in alg.optimizer.param_groups for p in group["params"]]
    assert len(optimizer_params) == len(list(model.parameters()))
    assert {id(p) for p in optimizer_params} == {id(p) for p in model.parameters()}
    assert isinstance(runner.get_inference_policy(), EpisodeContextActorView)
    print("[ok] construct_algorithm builds the joint model, storage and partitioned views")


def test_runner_learns_saves_and_loads() -> None:
    runner = OnPolicyRunner(
        DummyEnv(), _train_cfg(eval_env_fraction=0.25, grad_accumulation_steps=2), log_dir=None, device="cpu"
    )
    model = runner.alg.model
    assert runner.alg.eval_env_ids.tolist() == [12, 13, 14, 15]
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    count_before = int(model.actor_obs_normalizer.count)
    runner.learn(num_learning_iterations=3)
    assert runner.current_learning_iteration == 2
    moved = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
    assert moved, "nothing moved through the runner's learn loop"
    # Deferred normalization: committed once per update, over the whole rollout.
    assert int(model.actor_obs_normalizer.count) == count_before + 3 * 8 * NUM_ENVS

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model_2.pt")
        runner.save(path)
        saved = torch.load(path, weights_only=False)
        assert {"actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter"} <= set(saved)
        fresh = OnPolicyRunner(
            DummyEnv(), _train_cfg(eval_env_fraction=0.25, grad_accumulation_steps=2), log_dir=None, device="cpu"
        )
        assert not torch.equal(fresh.alg.model.token_embed.weight, model.token_embed.weight)
        fresh.load(path)
        assert fresh.current_learning_iteration == 2
        for name, parameter in fresh.alg.model.state_dict().items():
            assert torch.equal(parameter, model.state_dict()[name]), f"{name} did not survive save/load"
        # strict=False on a wider head: zero-padded, optimizer skipped, training continues.
        cfg = _train_cfg(eval_env_fraction=0.25, grad_accumulation_steps=2)
        cfg["actor"]["hidden_dims"] = [24]
        wider = OnPolicyRunner(DummyEnv(), cfg, log_dir=None, device="cpu")
        wider.load(path, strict=False)
        assert torch.equal(wider.alg.model.actor[0].weight[:16], model.actor[0].weight)
        wider.learn(num_learning_iterations=1)

    # With a writer the logger consumes ``loss_dict``, ``get_policy().output_std`` and the ep-extras key union.
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _train_cfg()
        cfg["logger"] = "tensorboard"
        logged = OnPolicyRunner(DummyEnv(), cfg, log_dir=tmp, device="cpu")
        logged.learn(num_learning_iterations=2)
        assert logged.logger.writer is not None
        assert os.path.exists(os.path.join(tmp, "model_1.pt"))

    # Inference through the actor view: deterministic, stateful, one step at a time.
    policy = runner.get_inference_policy()
    obs = DummyEnv().get_observations()
    policy.model.reset()
    with torch.inference_mode():
        action = policy(obs)
    assert action.shape == (NUM_ENVS, NUM_ACTIONS)
    assert policy.model.positions.tolist() == [1] * NUM_ENVS
    print("[ok] learn / save / load / strict=False load / inference policy all run through the stock runner")


if __name__ == "__main__":
    test_runner_constructs_the_joint_model()
    test_runner_learns_saves_and_loads()
    print("all episode-context runner tests passed")
