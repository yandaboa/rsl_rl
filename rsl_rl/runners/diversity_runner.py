# Copyright (c) 2024-2026, The UW Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import torch

from rsl_rl.algorithms import DiversityPPO
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


class DiversityRunner(OnPolicyRunner):
    """OnPolicyRunner specialised for :class:`DiversityPPO`.

    The only behavioural difference vs. the parent runner is that we publish the discriminator
    on the env after the algorithm is constructed, so the env-side reward term can call it.
    Saving / loading also persist the discriminator weights and optimiser state.
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env=env, train_cfg=train_cfg, log_dir=log_dir, device=device)
        if not isinstance(self.alg, DiversityPPO):
            raise TypeError(
                "DiversityRunner expects DiversityPPO; got "
                f"{type(self.alg).__name__}. Set algorithm.class_name='DiversityPPO'."
            )
        self.alg.attach_env(self.env)

    def save(self, path: str, infos: dict | None = None) -> None:
        super().save(path, infos)
        # Append discriminator state to the same checkpoint file.
        loaded = torch.load(path, weights_only=False, map_location="cpu")
        if self.alg.discriminator is not None:
            loaded["discriminator_state_dict"] = self.alg.discriminator.state_dict()
            loaded["discriminator_optimizer_state_dict"] = self.alg.discriminator_optimizer.state_dict()
            torch.save(loaded, path)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
        loaded = torch.load(path, weights_only=False, map_location=map_location)
        if "discriminator_state_dict" in loaded and self.alg.discriminator is not None:
            self.alg.discriminator.load_state_dict(loaded["discriminator_state_dict"])
            if load_optimizer and "discriminator_optimizer_state_dict" in loaded:
                self.alg.discriminator_optimizer.load_state_dict(loaded["discriminator_optimizer_state_dict"])
            # Re-publish on env in case the runner was just reconstructed.
            self.alg.attach_env(self.env)
        return infos

    def train_mode(self) -> None:
        super().train_mode()
        if self.alg.discriminator is not None:
            self.alg.discriminator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        if self.alg.discriminator is not None:
            self.alg.discriminator.eval()
