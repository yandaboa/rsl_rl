# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2020 Preferred Networks, Inc.


from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(
        self,
        shape: int | tuple[int, ...] | list[int],
        eps: float = 1e-2,
        until: int | None = None,
        clip: float | None = 10.0,
        outlier_clip: float | None = 10.0,
        outlier_warmup: int = 1_000_000,
    ) -> None:
        """Initialize EmpiricalNormalization module.

        .. note:: The normalization parameters are computed over the whole batch, not for each environment separately.

        Args:
            shape: Shape of input values except batch axis.
            eps: Small value for stability.
            until: If this arg is specified, the module learns input values until the sum of batch sizes exceeds it.
            clip: Clamp normalized outputs to ``[-clip, clip]`` (``None`` disables).
            outlier_clip: Winsorize inputs to ``mean +- outlier_clip * std`` before they enter the running
                moments (``None`` disables). The running variance never forgets, so a single glitched sample
                (a 1e3 joint velocity from a PhysX blow-up) otherwise inflates a feature's std for the rest of
                the run and silently rescales that input for the policy.
            outlier_warmup: Winsorizing starts once ``count`` exceeds this many samples (the moments need to be
                trustworthy first).
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.clip = clip
        self.outlier_clip = outlier_clip
        self.outlier_warmup = outlier_warmup
        # Fraction of elements winsorized by the last ``update`` (diagnostic, not checkpointed).
        self.last_outlier_frac = 0.0
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
        # Runtime-only switch configured by PPO. Keeping it as a plain attribute
        # preserves the checkpoint state-dict schema exactly.
        self._distributed_sync_enabled = False

    @property
    def mean(self) -> torch.Tensor:
        """Return the current running mean."""
        return self._mean.squeeze(0).clone()  # type: ignore

    @property
    def std(self) -> torch.Tensor:
        """Return the current running standard deviation."""
        return self._std.squeeze(0).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize mean and variance of values based on empirical values."""
        y = (x - self._mean) / (self._std + self.eps)
        if self.clip is not None:
            y = y.clamp(-self.clip, self.clip)
        return y

    @torch.jit.unused
    def _winsorize(self, x: torch.Tensor) -> torch.Tensor:
        if self.outlier_clip is None or int(self.count) < self.outlier_warmup:
            return x
        width = self.outlier_clip * (self._std + self.eps)
        clamped = torch.maximum(torch.minimum(x, self._mean + width), self._mean - width)
        self.last_outlier_frac = (clamped != x).float().mean()  # tensor: no device sync per step
        return clamped

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        """Learn input values without computing the output values of them."""
        if not self.training:
            return
        if self.until is not None and self.count >= self.until:
            return

        x = self._winsorize(x)
        if self._distributed_sync_enabled:
            self._distributed_update(x)
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def set_distributed_sync(self, enabled: bool = True) -> None:
        """Enable per-update aggregation of input moments across distributed ranks.

        This is intentionally runtime-only: the running moments remain ordinary
        persistent buffers, while the synchronization policy is supplied by the
        active training configuration.
        """
        self._distributed_sync_enabled = enabled

    @torch.jit.unused
    def _distributed_update(self, x: torch.Tensor) -> None:
        """Merge one globally aggregated input batch into the shared running state."""
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError("Distributed EmpiricalNormalization requires an initialized process group.")

        # Accumulate sufficient statistics in float64. One packed collective per
        # normalizer keeps ordering explicit and avoids cancellation at large
        # distributed batch sizes.
        x_64 = x.detach().to(dtype=torch.float64)
        local_count = torch.tensor([x.shape[0]], device=x.device, dtype=torch.float64)
        local_sum = x_64.sum(dim=0, keepdim=True).reshape(-1)
        local_sq_sum = x_64.square().sum(dim=0, keepdim=True).reshape(-1)
        moments = torch.cat((local_count, local_sum, local_sq_sum))
        torch.distributed.all_reduce(moments, op=torch.distributed.ReduceOp.SUM)
        # Check only after the collective so every rank follows the same
        # ordering and fails together. Raising before all_reduce on the rank
        # that first sees Inf/NaN would strand its peers inside NCCL.
        if not torch.isfinite(moments).all():
            raise FloatingPointError("Non-finite observation moments detected during distributed normalization.")

        count_x = moments[0]
        if count_x <= 0:
            return
        num_features = self._mean.numel()
        mean_x = (moments[1 : 1 + num_features] / count_x).reshape_as(self._mean)
        second_moment_x = (moments[1 + num_features :] / count_x).reshape_as(self._mean)
        var_x = (second_moment_x - mean_x.square()).clamp_min_(0.0)
        self._merge_moments(count_x, mean_x, var_x)

    @torch.jit.unused
    def _merge_moments(self, count_x: torch.Tensor, mean_x: torch.Tensor, var_x: torch.Tensor) -> None:
        """Chan-merge population moments into the existing running moments."""
        count = self.count.to(dtype=torch.float64)
        new_count = count + count_x
        rate = count_x / new_count
        mean = self._mean.to(dtype=torch.float64)
        var = self._var.to(dtype=torch.float64)
        delta_mean = mean_x - mean
        new_mean = mean + rate * delta_mean
        new_var = var + rate * (var_x - var + delta_mean * (mean_x - new_mean))

        self.count.add_(count_x.to(dtype=self.count.dtype))
        self._mean.copy_(new_mean.to(dtype=self._mean.dtype))
        self._var.copy_(new_var.clamp_min_(0.0).to(dtype=self._var.dtype))
        self._std.copy_(torch.sqrt(self._var))

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """De-normalize values based on empirical values."""
        return y * (self._std + self.eps) + self._mean


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Reward normalization from Pathak's large scale study on PPO.

    Reward normalization. Since the reward function is non-stationary, it is useful to normalize the scale of the
    rewards so that the value function can learn quickly. We did this by dividing the rewards by a running estimate of
    the standard deviation of the sum of discounted rewards.
    """

    def __init__(
        self,
        shape: int | tuple[int, ...] | list[int],
        eps: float = 1e-2,
        gamma: float = 0.99,
        until: int | None = None,
    ) -> None:
        """Initialize discounted-reward normalization with running moments."""
        super().__init__()

        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = _DiscountedAverage(gamma)

    def forward(self, rew: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using the running std of discounted returns."""
        if self.training:
            # Update discounted rewards
            avg = self.disc_avg.update(rew)
            # Update moments from discounted rewards
            self.emp_norm.update(avg)

        # Normalize rewards with the empirical std
        if self.emp_norm._std > 0:  # type: ignore
            return rew / self.emp_norm._std  # type: ignore
        else:
            return rew


class _DiscountedAverage:
    r"""Discounted average of rewards.

    The discounted average is defined as:

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t
    """

    def __init__(self, gamma: float) -> None:
        """Initialize discounted accumulation with a fixed discount factor."""
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        """Update and return the discounted running average."""
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg
