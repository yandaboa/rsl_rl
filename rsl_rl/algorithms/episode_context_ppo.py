# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO for the single-episode context transformer.

This is the **stock** :class:`~rsl_rl.algorithms.ppo.PPO` -- ``act``, ``process_env_step``, ``compute_returns``
and the whole ``update()`` are inherited byte-for-byte from the known-good implementation. The only override is
:meth:`init_storage`, which builds an :class:`EpisodeContextRolloutStorage` instead of a plain
:class:`RolloutStorage`; that storage's ``recurrent_mini_batch_generator`` then feeds PPO's existing recurrent
branch (``policy.is_recurrent`` is ``True``) with ``[W, B, ...]`` minibatches whose prefix rides in the
hidden-state slot.

The class exists only because ``PPO.init_storage`` names its storage class literally. Everything a reader might
expect to be different here -- the loss, the ratio, the KL schedule, the clipping, the diagnostics -- is not.
"""

from __future__ import annotations

import warnings
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules.actor_critic_episode_context import ActorCriticEpisodeContext
from rsl_rl.storage.episode_context_storage import EpisodeContextRolloutStorage


class EpisodeContextPPO(PPO):
    """PPO whose rollout storage keeps the per-environment frame ring the context transformer re-infers from."""

    def __init__(
        self,
        policy: ActorCriticEpisodeContext,
        *args,
        defer_obs_normalization: bool = True,
        **kwargs,
    ) -> None:
        """Same signature as :class:`PPO`, except that ``defer_obs_normalization`` defaults to ``True``.

        It has to: the update re-infers every window from the RAW stored frames, so if the normalizer moved
        during collection the reconstruction would run under different statistics than the acting path did and
        the PPO ratio would be off at epoch 0 by construction.
        """
        if not isinstance(policy, ActorCriticEpisodeContext):
            raise ValueError(
                "EpisodeContextPPO requires an ActorCriticEpisodeContext policy (it is what owns the KV cache and"
                f" the windowed forward); got {type(policy).__name__}."
            )
        if not defer_obs_normalization and getattr(policy, "actor_obs_normalization", False):
            warnings.warn(
                "EpisodeContextPPO with defer_obs_normalization=False: the rollout is acted and reconstructed"
                " under different observation-normalizer statistics, so the epoch-0 PPO ratio will not be 1.",
                stacklevel=2,
            )
        super().__init__(policy, *args, defer_obs_normalization=defer_obs_normalization, **kwargs)

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:
        self.storage = EpisodeContextRolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
            actor_obs_groups=self.policy.obs_groups["policy"],
            context_length=self.policy.context_length,
            max_episode_length=self.policy.max_episode_length,
            num_layers=self.policy.num_layers,
            actor_obs_normalizer=self.policy.actor_obs_normalizer,
            max_policy_lag=self.max_policy_lag,
        )
