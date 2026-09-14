# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import pathlib
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter
from typing import Any

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("wandb package is required to log to Weights and Biases.") from None


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for W&B."""

    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        """Initialize a W&B run for logging."""
        super().__init__(log_dir, flush_secs=flush_secs)

        # Get the run name
        run_name = os.path.split(log_dir)[-1]

        # Get wandb project and entity
        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify wandb_project in the runner config, e.g. legged_gym.") from None
        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None

        # Initialize wandb. When the runner config provides a ``run_id`` (typically the
        # SLURM job id, propagated from ``train.py --run_id``), use it as a deterministic
        # wandb run id so a requeued cluster job resumes the same dashboard run instead
        # of starting a new one.
        wandb_kwargs: dict = dict(
            project=project,
            entity=entity,
            name=run_name,
            config={"log_dir": log_dir},
        )
        run_id = cfg.get("run_id")
        if run_id is not None:
            wandb_kwargs["id"] = str(run_id)
            wandb_kwargs["resume"] = "allow"
        wandb.init(**wandb_kwargs)

        # Initialize set to keep track of logged videos
        self.logged_videos: set[str] = set()

    def store_config(self, env_cfg: dict | object, train_cfg: dict) -> None:
        """Upload environment and training configuration to W&B.

        ``allow_val_change=True`` is required because ``store_config`` runs on every job
        attempt; on a resumed run the keys already exist and wandb's strict mode would
        otherwise raise a config conflict even when the values are unchanged.
        """
        wandb.config.update({"train_cfg": train_cfg}, allow_val_change=True)
        try:
            wandb.config.update({"env_cfg": env_cfg.to_dict()}, allow_val_change=True)  # type: ignore
        except Exception:
            wandb.config.update({"env_cfg": asdict(env_cfg)}, allow_val_change=True)  # type: ignore

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        """Log a scalar to both TensorBoard and W&B."""
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({tag: scalar_value}, step=global_step)

    def add_scalars_batch(
        self,
        tag_scalar_dict: dict[str, float],
        global_step: int | None = None,
        walltime: float | None = None,
        new_style: bool = False,
    ) -> None:
        """Log independent scalars to TensorBoard and one batched W&B row."""
        if not tag_scalar_dict:
            return
        for tag, scalar_value in tag_scalar_dict.items():
            # Bypass this class's single-scalar override so W&B receives one
            # payload below while TensorBoard retains the exact scalar schema.
            super().add_scalar(
                tag,
                scalar_value,
                global_step=global_step,
                walltime=walltime,
                new_style=new_style,
            )
        wandb.log(dict(tag_scalar_dict), step=global_step)

    def add_image(
        self,
        tag: str,
        img_tensor: Any,
        global_step: int | None = None,
        walltime: float | None = None,
        dataformats: str = "CHW",
    ) -> None:
        """Log an image to both TensorBoard and W&B.

        ``img_tensor`` is forwarded as-is to ``wandb.Image``, which handles numpy
        ``HWC``/``HW`` arrays and torch tensors directly.
        """
        super().add_image(
            tag,
            img_tensor,
            global_step=global_step,
            walltime=walltime,
            dataformats=dataformats,
        )
        wandb.log({tag: wandb.Image(img_tensor)}, step=global_step)

    def stop(self) -> None:
        """Finish the active W&B run."""
        wandb.finish()

    def save_model(self, model_path: str, it: int) -> None:
        """Upload a model checkpoint artifact to W&B."""
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path: str) -> None:
        """Upload an arbitrary file artifact to W&B."""
        wandb.save(path, base_path=os.path.dirname(path))

    def save_video(self, video: pathlib.Path, it: int) -> None:
        """Upload a video artifact once per filename to W&B."""
        if video.name not in self.logged_videos:
            wandb.log({"video": wandb.Video(str(video), format="mp4")}, step=it)
            self.logged_videos.add(video.name)
