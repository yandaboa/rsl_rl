# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-block conditioning for SAPG: a learned phi_j at the MLP input and a per-block exploration std.

The block's phi_j is concatenated to the main-MLP input (after observation normalization / encoders /
head LayerNorm). The first Linear is split into ``W x + W_phi phi_j + b`` (the same map as a Linear over
``[x || phi_j]``) so a bare ``x`` from an exporter can be served with the leader's phi.

The distribution's std parameter (``log_std_param`` / ``std_param``) stays the leader's (block 0), so exporters
reading it get the leader; followers 1..M-1 live in ``sapg_follower_std`` (play2perfect ``fixed_sigma: coef_cond``).

With ``shared_layers`` the MLP's Linear layers past the first ``shared_layers`` become per-block heads
(``BlockLinear``); a bare forward (no block ids) runs block 0.
"""

from __future__ import annotations

import contextlib
import torch
import torch.nn as nn
from collections.abc import Iterator
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Optional

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.modules.distribution import Distribution, GaussianDistribution, GsdeDistribution


class PhiConditionedLinear(nn.Linear):
    """First MLP layer over ``[x || phi]``: ``weight`` acts on ``x``, ``phi_weight`` on ``phi``.

    ``phi`` is the ``[num_blocks, phi_dim]`` table, ``torch.randn`` like play2perfect's ``extra_params``. The layer
    is initialized as one fresh ``nn.Linear(in + phi_dim, out)`` split in two. An input WITHOUT the phi columns (an
    external exporter walking ``model.mlp`` on the bare latent) is evaluated with the leader's ``phi[0]``.
    """

    def __init__(self, linear: nn.Linear, num_blocks: int, phi_dim: int) -> None:
        """Replace ``linear`` (same shape on ``x``) and add the phi columns and the phi table."""
        device = linear.weight.device
        super().__init__(linear.in_features, linear.out_features, bias=linear.bias is not None, device=device)
        self.phi_dim = int(phi_dim)
        joint = nn.Linear(
            linear.in_features + phi_dim, linear.out_features, bias=linear.bias is not None, device=device
        )
        with torch.no_grad():
            self.weight.copy_(joint.weight[:, : linear.in_features])
            if self.bias is not None:
                self.bias.copy_(joint.bias)
        self.phi_weight = nn.Parameter(joint.weight[:, linear.in_features :].detach().clone())
        self.phi = nn.Parameter(torch.randn(num_blocks, phi_dim, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[x || phi]`` -> pre-activation; a bare ``x`` uses the leader's ``phi[0]``."""
        if x.shape[-1] == self.in_features + self.phi_dim:
            return nn.functional.linear(x[..., : self.in_features], self.weight, self.bias) + nn.functional.linear(
                x[..., self.in_features :], self.phi_weight
            )
        return nn.functional.linear(x, self.weight, self.bias) + nn.functional.linear(self.phi[0], self.phi_weight)


class BlockLinear(nn.Module):
    """``M`` independent Linear layers; row ``i`` uses block ``_sapg_block_ids[i]`` (``None`` = block 0 for every row).

    Block 0 is the replaced layer; blocks 1..M-1 are fresh orthogonal draws with its gain and a copy of its bias.
    """

    # Real type (not a postponed string) so TorchScript exporters can script the bare-latent (leader) path.
    __annotations__ = {"_sapg_block_ids": Optional[torch.Tensor]}

    def __init__(self, linear: nn.Linear, num_blocks: int) -> None:
        """Replace ``linear`` (needs an orthogonally-initialized weight: the gain is read off its singular values)."""
        super().__init__()
        assert linear.bias is not None, "BlockLinear needs a Linear with a bias."
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.num_blocks = int(num_blocks)
        self._sapg_block_ids = None
        with torch.no_grad():
            svals = torch.linalg.svdvals(linear.weight.detach().float())
            gain = float(svals[0])
            assert abs(gain - float(svals[-1])) <= 1e-3, (
                f"per-block heads re-draw an orthogonal init; this layer is not orthogonal (singular values"
                f" {float(svals[-1]):.4f}..{gain:.4f})."
            )
            weight = linear.weight.detach().unsqueeze(0).repeat(self.num_blocks, 1, 1)
            for j in range(1, self.num_blocks):
                nn.init.orthogonal_(weight[j], gain=gain)
            bias = linear.bias.detach().unsqueeze(0).repeat(self.num_blocks, 1)
        self.weight = nn.Parameter(weight, requires_grad=linear.weight.requires_grad)
        self.bias = nn.Parameter(bias, requires_grad=linear.bias.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Row ``i`` through block ``_sapg_block_ids[i]``; every row through block 0 when no ids are set."""
        ids = self._sapg_block_ids
        if ids is None:
            return nn.functional.linear(x, self.weight[0], self.bias[0])
        assert ids.shape[0] == x.shape[0], "SAPG block ids do not match the batch."
        out = x.new_empty(x.shape[0], self.out_features)
        # One matmul per block over its rows (a per-row weight gather would be B x out x in).
        for j in range(self.num_blocks):
            mask = ids == j
            out[mask] = nn.functional.linear(x[mask], self.weight[j], self.bias[j])
        return out

    def extra_repr(self) -> str:
        """Layer summary for ``print(model)``."""
        return f"num_blocks={self.num_blocks}, in_features={self.in_features}, out_features={self.out_features}"


def attach_block_heads(model: MLPModel, num_blocks: int, shared_layers: int) -> MLPModel:
    """Turn every Linear of ``model.mlp`` after the first ``shared_layers`` into a per-block ``BlockLinear``."""
    positions = [i for i, m in enumerate(model.mlp) if isinstance(m, nn.Linear)]
    assert 0 <= shared_layers < len(positions), (
        f"shared_layers must be in [0, {len(positions) - 1}] for a {len(positions)}-Linear MLP, got {shared_layers}."
    )
    for i in positions[shared_layers:]:
        layer = model.mlp[i]
        assert type(layer) is nn.Linear, f"expected nn.Linear at mlp[{i}], got {type(layer).__name__}."
        model.mlp[i] = BlockLinear(layer, num_blocks)
    model.sapg_shared_layers = int(shared_layers)  # type: ignore[attr-defined]
    return model


class _BlockStdMixin:
    """Per-block state-independent std: rows use ``[leader, *followers][block_ids]`` (``None`` = leader)."""

    _sapg_block_ids: torch.Tensor | None = None
    sapg_std_name: str
    sapg_follower_std: nn.Parameter
    _sapg_matrix_sampling: bool = False

    def sapg_std_table(self) -> torch.Tensor:
        """``[M, *std_shape]`` raw std parameters (leader first), in the parameter's own space (log or scalar)."""
        leader = getattr(self, self.sapg_std_name)
        return torch.cat([leader.unsqueeze(0), self.sapg_follower_std], dim=0)

    def _block_std(self) -> torch.Tensor:
        """``[M, *std_shape]`` clamped std (not log)."""
        table = self.sapg_std_table()
        if self.sapg_std_name == "std_param":
            return table.clamp(self.std_range[0], self.std_range[1])  # type: ignore[attr-defined]
        return torch.exp(table.clamp(self.log_std_range[0], self.log_std_range[1]))  # type: ignore[attr-defined]

    def update(self, mlp_output: torch.Tensor) -> None:
        ids = self._sapg_block_ids
        if ids is None:
            return super().update(mlp_output)  # type: ignore[misc]
        assert ids.shape[0] == mlp_output.shape[0], f"SAPG block ids ({ids.shape[0]}) != batch ({mlp_output.shape[0]})."
        std = self._block_std()
        if isinstance(self, GsdeDistribution):
            features = self._cached_features
            if features is None:
                raise RuntimeError("GsdeDistribution.update called before set_features().")
            num_blocks, latent_dim, num_out = std.shape
            # One matmul over every block's std columns, then pick each row's block: [B, M, A] -> [B, A].
            var_all = features.pow(2) @ std.pow(2).permute(1, 0, 2).reshape(latent_dim, num_blocks * num_out)
            variance = var_all.view(-1, num_blocks, num_out)[torch.arange(ids.shape[0], device=ids.device), ids]
            self._distribution = Normal(mlp_output, torch.sqrt(variance + self.epsilon))
        else:
            self._distribution = Normal(mlp_output, std[ids])

    def sample_weights(self) -> None:
        parent = getattr(super(), "sample_weights", None)
        if parent is not None:
            parent()
        if self._sapg_matrix_sampling:
            std = self._block_std().detach()[1:]
            self._sapg_follower_exploration = std * torch.randn_like(std)

    def sample(self) -> torch.Tensor:
        ids = self._sapg_block_ids
        if not self._sapg_matrix_sampling or ids is None:
            return super().sample()  # type: ignore[misc]
        # Stock gSDE: noise = features @ W, one W per block (the leader's is the parent's buffer).
        features = self._cached_features
        matrices = torch.cat([self._exploration_matrix.unsqueeze(0), self._sapg_follower_exploration], dim=0)
        num_blocks, latent_dim, num_out = matrices.shape
        noise_all = features @ matrices.permute(1, 0, 2).reshape(latent_dim, num_blocks * num_out)
        noise = noise_all.view(-1, num_blocks, num_out)[torch.arange(ids.shape[0], device=ids.device), ids]
        return self._distribution.mean + noise  # type: ignore[attr-defined]


class _BlockConditionedMixin:
    """Appends ``phi[block_ids]`` (leader ``phi[0]`` when no ids are set) to the model's MLP latent."""

    _sapg_block_ids: torch.Tensor | None = None

    @property
    def sapg_linear(self) -> PhiConditionedLinear | None:
        first = self.mlp[0]  # type: ignore[attr-defined]
        return first if isinstance(first, PhiConditionedLinear) else None

    @property
    def sapg_phi(self) -> nn.Parameter | None:
        linear = self.sapg_linear
        return None if linear is None else linear.phi

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        latent = super().get_latent(obs, masks, hidden_state)  # type: ignore[misc]
        phi = self.sapg_phi
        if phi is None:
            return latent
        ids = self._sapg_block_ids
        if ids is None:
            phi_rows = phi[0].expand(latent.shape[0], -1)
        else:
            assert ids.shape[0] == latent.shape[0], (
                f"SAPG block ids ({ids.shape[0]}) do not match the batch ({latent.shape[0]})."
            )
            phi_rows = phi[ids]
        return torch.cat([latent, phi_rows], dim=-1)

    @contextlib.contextmanager
    def sapg_blocks(self, block_ids: torch.Tensor | None) -> Iterator[None]:
        """Forward passes inside the context use block ``block_ids`` per sample (``None`` = leader)."""
        dist = getattr(self, "distribution", None)
        targets = [self] + ([dist] if isinstance(dist, _BlockStdMixin) else [])
        targets += [m for m in self.modules() if isinstance(m, BlockLinear)]  # type: ignore[attr-defined]
        previous = [t._sapg_block_ids for t in targets]
        for t in targets:
            t._sapg_block_ids = block_ids
        try:
            yield
        finally:
            for t, prev in zip(targets, previous):
                t._sapg_block_ids = prev


_CONDITIONED_CLASSES: dict[type, type] = {}


def _conditioned_class(base: type, mixin: type) -> type:
    key = (base, mixin)
    if key not in _CONDITIONED_CLASSES:
        _CONDITIONED_CLASSES[key] = type(f"SAPG{base.__name__}", (mixin, base), {})  # type: ignore[index]
    return _CONDITIONED_CLASSES[key]  # type: ignore[index]


def attach_block_std(dist: Distribution, num_blocks: int) -> Distribution:
    """Give ``dist`` one state-independent std per block (in place), all starting at the leader's init."""
    assert isinstance(dist, GaussianDistribution) and type(dist).update in (
        GaussianDistribution.update,
        GsdeDistribution.update,
    ), f"per-block std needs a Gaussian / gSDE distribution with a state-independent std, got {type(dist).__name__}."
    assert not isinstance(dist, _BlockStdMixin), "distribution already has per-block std."
    name = "std_param" if getattr(dist, "std_type", "log") == "scalar" else "log_std_param"
    leader = getattr(dist, name, None)
    assert isinstance(leader, nn.Parameter), f"{type(dist).__name__} has no '{name}' parameter."
    matrix_sampling = isinstance(dist, GsdeDistribution) and type(dist).sample is GsdeDistribution.sample
    dist.__class__ = _conditioned_class(type(dist), _BlockStdMixin)
    dist.sapg_std_name = name  # type: ignore[attr-defined]
    dist._sapg_matrix_sampling = matrix_sampling  # type: ignore[attr-defined]
    followers = leader.detach().unsqueeze(0).repeat(num_blocks - 1, *([1] * leader.dim()))
    dist.sapg_follower_std = nn.Parameter(followers, requires_grad=leader.requires_grad)  # type: ignore[attr-defined]
    if matrix_sampling:
        dist.sample_weights()
    return dist


def attach_block_conditioning(
    model: MLPModel,
    num_blocks: int,
    phi_dim: int,
    per_block_std: bool = False,
    shared_layers: int | None = None,
) -> MLPModel:
    """Condition ``model`` (in place) on its block; returns the model.

    Adds a ``[num_blocks, phi_dim]`` phi table (if ``phi_dim > 0``), with ``per_block_std`` one exploration
    std per block on its distribution, and with ``shared_layers`` per-block heads after that many shared Linears.
    """
    assert isinstance(model, MLPModel), f"SAPG conditioning needs an MLPModel-family model, got {type(model).__name__}."
    assert not model.is_recurrent, "SAPG conditioning supports feedforward models only."
    assert not is_block_conditioned(model), "model is already block-conditioned."
    if shared_layers is not None:
        # phi lives on the first Linear, which must then stay shared
        assert phi_dim == 0 or shared_layers >= 1, "phi_dim > 0 needs shared_layers >= 1."
        attach_block_heads(model, num_blocks, shared_layers)
    if phi_dim > 0:
        first = model.mlp[0]
        assert type(first) is nn.Linear, f"expected the MLP's first layer to be nn.Linear, got {type(first).__name__}."
        model.mlp[0] = PhiConditionedLinear(first, num_blocks, phi_dim)
    if per_block_std and num_blocks > 1 and model.distribution is not None:
        attach_block_std(model.distribution, num_blocks)
    model.__class__ = _conditioned_class(type(model), _BlockConditionedMixin)
    return model


def is_block_conditioned(model: nn.Module) -> bool:
    """Whether ``model`` carries SAPG block conditioning."""
    return isinstance(model, _BlockConditionedMixin)


@contextlib.contextmanager
def block_context(block_ids: torch.Tensor | None, *models: nn.Module) -> Iterator[None]:
    """Set per-sample block ids on every conditioned model among ``models`` (unconditioned ones are skipped)."""
    with contextlib.ExitStack() as stack:
        for model in models:
            if is_block_conditioned(model):
                stack.enter_context(model.sapg_blocks(block_ids))  # type: ignore[attr-defined]
        yield
