# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Building blocks for neural models."""

from .cnn import CNN
from .distribution import (
    Distribution,
    GaussianDistribution,
    GsdeDistribution,
    HeteroscedasticGaussianDistribution,
    TanhGaussianDistribution,
)
from .mlp import MLP
from .normalization import EmpiricalDiscountedVariationNormalization, EmpiricalNormalization
from .rnn import RNN, HiddenState
from .student_teacher import StudentTeacher
from .student_teacher_vision import StudentTeacherVision
from .transformer import MultiHeadAttention, TrunkBlock

# NOTE: StudentTeacherRecurrent intentionally skipped — depends on rsl_rl.networks.Memory
# which feature/manipulation hasn't ported. Re-add once the recurrent student is needed.

__all__ = [
    "CNN",
    "MLP",
    "RNN",
    "Distribution",
    "EmpiricalDiscountedVariationNormalization",
    "EmpiricalNormalization",
    "GaussianDistribution",
    "GsdeDistribution",
    "HeteroscedasticGaussianDistribution",
    "TanhGaussianDistribution",
    "HiddenState",
    "MultiHeadAttention",
    "StudentTeacher",
    "StudentTeacherVision",
    "TrunkBlock",
]
