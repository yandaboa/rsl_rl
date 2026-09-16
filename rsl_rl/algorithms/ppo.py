# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from collections.abc import Iterable
from itertools import chain
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import RandomNetworkDistillation, Symmetry, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.modules.distribution import TanhGaussianDistribution
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import (
    GradientNoiseScaleTracker,
    compile_model,
    resolve_callable,
    resolve_obs_groups,
    resolve_optimizer,
)


class PPO:
    """Proximal Policy Optimization algorithm.

    Reference:
        - Schulman et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
    """

    actor: MLPModel
    """The actor model."""

    critic: MLPModel
    """The critic model."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        distributed_obs_normalization: bool = False,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # Gradient noise scale parameters
        gradient_noise_scale_cfg: dict | None = None,
        # Eval env pool parameters
        eval_env_fraction: float = 0.0,
        eval_deterministic: bool = True,
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings.

        ``eval_env_fraction``: trailing fraction of the environments held out of the update and of the advantage
        statistics (the storage must have been built with the matching ``num_eval_envs``, see
        :meth:`_num_eval_envs`). ``eval_deterministic``: the pool executes the model's deterministic output
        instead of the sample.
        """
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND extension
        self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg) if rnd_cfg else None

        # Symmetry extension
        if symmetry_cfg is not None and (actor.is_recurrent or critic.is_recurrent):
            raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
        self.symmetry = Symmetry(**symmetry_cfg) if symmetry_cfg else None

        # PPO components
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Handles to the uncompiled modules for state_dict operations and export. If compilation is disabled, these
        # simply alias ``self.actor`` / ``self.critic``.
        self._raw_actor = self.actor
        self._raw_critic = self.critic

        # Observation-normalizer buffers are not gradients, so the ordinary
        # parameter reduction does not keep them aligned after the initial
        # broadcast. Opt in to exact per-step global moment aggregation for
        # every actor/critic EmpiricalNormalization descendant.
        self.distributed_obs_normalization = distributed_obs_normalization and self.is_multi_gpu
        if self.distributed_obs_normalization:
            self._enable_distributed_obs_normalization()

        # Create the optimizer
        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.actor.parameters(), self.critic.parameters()), lr=learning_rate
        )  # type: ignore

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # Gradient noise scale instrumentation (read-only metric; never writes p.grad)
        self.noise_scale_tracker = self._build_noise_scale_tracker(gradient_noise_scale_cfg)

        # Eval env pool: the TRAILING ``num_eval_envs`` environments (one contiguous training slice in front)
        assert 0.0 <= eval_env_fraction < 1.0, f"eval_env_fraction must be in [0, 1), got {eval_env_fraction}."
        self.eval_env_fraction = float(eval_env_fraction)
        self.eval_deterministic = bool(eval_deterministic)
        num_envs = storage.num_envs
        self.num_eval_envs = self._num_eval_envs(eval_env_fraction, num_envs)
        assert storage.num_eval_envs == self.num_eval_envs, (
            f"the storage was built with num_eval_envs={storage.num_eval_envs}, but eval_env_fraction="
            f"{eval_env_fraction} x {num_envs} envs = {self.num_eval_envs}. Build it with the matching num_eval_envs."
        )
        if self.num_eval_envs > 0:
            self.eval_env_ids = torch.arange(num_envs, device=self.device)[-self.num_eval_envs :]
            self.eval_env_mask: torch.Tensor | None = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
            self.eval_env_mask[self.eval_env_ids] = True
        else:
            self.eval_env_ids = torch.zeros(0, dtype=torch.long, device=self.device)
            # ``None`` (not an all-False mask): act() then takes the untouched stock path.
            self.eval_env_mask = None

    @property
    def _eval_mask(self) -> torch.Tensor | None:
        """Alias of :attr:`eval_env_mask` (older name)."""
        return self.eval_env_mask

    @staticmethod
    def _num_eval_envs(eval_env_fraction: float, num_envs: int) -> int:
        num_eval_envs = round(float(eval_env_fraction) * num_envs)
        assert num_eval_envs < num_envs, (
            f"eval_env_fraction={eval_env_fraction} leaves no training environment ({num_eval_envs} of {num_envs})."
        )
        return num_eval_envs

    def deterministic_actions(self) -> torch.Tensor:
        """The deterministic action of the distribution the actor's last forward pass left behind.

        ``output_mean`` is the mean of the base Gaussian, which for a squashed distribution
        (``TanhGaussianDistribution``) is pre-squash: apply the squash so the pool executes ``deterministic_output``.
        """
        mean = self.actor.output_mean
        if isinstance(getattr(self.actor, "distribution", None), TanhGaussianDistribution):
            mean = torch.tanh(mean)
        return mean

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data; a deterministic eval pool executes the mean instead."""
        # Record the hidden states for recurrent policies
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        # Compute the actions and values
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        if self.eval_env_mask is not None and self.eval_deterministic:
            # Same forward pass (recurrent state / KV caches advance in lockstep); only the executed action is
            # swapped, off the distribution the actor just left behind.
            self.transition.actions = torch.where(
                self.eval_env_mask.unsqueeze(-1), self.deterministic_actions().detach(), self.transition.actions
            )
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()  # type: ignore
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and update the normalizers."""
        # Update the normalizers
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions."""
        st = self.storage
        # Compute value for the last step
        last_values = self.critic(obs).detach()
        # Compute returns and advantages
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = advantage + st.values[step]
        # Compute the advantages
        st.advantages = st.returns - st.values
        # Normalize the advantages if per minibatch normalization is not used (statistics of the training pool)
        if not self.normalize_advantage_per_mini_batch:
            train_advantages = st.advantages[:, : st.num_train_envs]
            st.advantages = (st.advantages - train_advantages.mean()) / (train_advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Gradient noise scale: per-mode setup
        grad_noise_mode = self.noise_scale_tracker.mode if self.noise_scale_tracker is not None else None
        if grad_noise_mode == "across_minibatches":
            self.noise_scale_tracker.begin_iteration()
        tracked_minibatch_size: int | None = None

        # Per-minibatch pre-clip gradient norms (actor, critic, combined). Always on; the
        # mean tracks training-signal magnitude and the variance is one of M3's three
        # supporting metrics (gradient-shaping fingerprint of shaped reward).
        grad_norms: dict[str, list[torch.Tensor]] = {"actor": [], "critic": [], "combined": []}

        # Get mini-batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over mini-batches
        for i, batch in enumerate(generator):
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini-batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation if enabled
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with new parameters
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            # Note: We only keep the following tensors for the original samples in case of symmetry augmentation
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss. The log-ratio is clamped before exponentiation: with squashed/bounded
            # distributions (e.g. TanhGaussianDistribution) the Jacobian-corrected log-probs of
            # saturated actions can legitimately differ by more than ~88 between collection and
            # update, and exp() then overflows to inf -> inf surrogate -> NaN gradients -> NaN
            # network weights. exp(+-20) keeps the ratio finite while staying far outside the PPO
            # clip range, so the clamp is inactive for healthy updates (plain Gaussians live at
            # |log-ratio| < ~5) and only arrests numerical blow-ups.
            log_ratio = actions_log_prob - torch.squeeze(batch.old_actions_log_prob)  # type: ignore
            ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # RND loss
            rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None  # type: ignore

            # Symmetry loss
            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            # Gradient noise scale: pre-reduce hooks (read-only on p.grad)
            local_grad_norm_sq: torch.Tensor | None = None
            if grad_noise_mode == "across_minibatches" and i < self.num_mini_batches:
                self.noise_scale_tracker.accumulate_minibatch(self._actor_critic_parameters())
                tracked_minibatch_size = original_batch_size
            elif grad_noise_mode == "ddp_native":
                local_grad_norm_sq = self.noise_scale_tracker.grad_norm_sq(self._actor_critic_parameters())

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Gradient noise scale (ddp_native): step on the post-reduce gradient
            if grad_noise_mode == "ddp_native":
                global_grad_norm_sq = self.noise_scale_tracker.grad_norm_sq(self._actor_critic_parameters())
                self.noise_scale_tracker.step_ddp_native(
                    local_grad_norm_sq,
                    global_grad_norm_sq,
                    b_small=original_batch_size,
                    b_big=original_batch_size * self.gpu_world_size,
                )

            # Per-minibatch pre-clip gradient norms (read-only on p.grad). Stored as scalar
            # tensors to avoid host-syncs every minibatch; aggregated to floats once after the loop.
            with torch.no_grad():
                actor_sq = self._grad_norm_sq(self.actor.parameters())
                critic_sq = self._grad_norm_sq(self.critic.parameters())
                grad_norms["actor"].append(actor_sq.sqrt())
                grad_norms["critic"].append(critic_sq.sqrt())
                grad_norms["combined"].append((actor_sq + critic_sq).sqrt())

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd:
                self.rnd.optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Gradient noise scale (across_minibatches): finalize EMAs from epoch-0 accumulators
        if grad_noise_mode == "across_minibatches" and tracked_minibatch_size is not None:
            self.noise_scale_tracker.step_across_minibatches(
                b_small=tracked_minibatch_size, num_mini_batches=self.num_mini_batches
            )

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if self.noise_scale_tracker is not None:
            for key, value in self.noise_scale_tracker.state().items():
                loss_dict[f"noise_scale/{key}"] = value

        # Per-minibatch grad-norm mean/variance across the iteration's update steps
        for component, values in grad_norms.items():
            if not values:
                continue
            stacked = torch.stack(values)
            loss_dict[f"grad_norm/{component}_mean"] = stacked.mean().item()
            loss_dict[f"grad_norm/{component}_var"] = stacked.var(unbiased=False).item() if len(values) > 1 else 0.0

        # Clear the storage
        self.storage.clear()

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic.eval()
        if self.rnd:
            self.rnd.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd.optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, load all models and states
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd.optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self._raw_actor

    def _enable_distributed_obs_normalization(self) -> None:
        """Enable global moment aggregation on all actor/critic normalizers."""
        for model in (self._raw_actor, self._raw_critic):
            for module in model.modules():
                if isinstance(module, EmpiricalNormalization):
                    module.set_distributed_sync(True)

    def compile(self, mode: str | None = None) -> None:
        """Compile actor and critic with ``torch.compile``.

        See :func:`~rsl_rl.utils.compile_model` for the set of accepted modes.

        Args:
            mode: ``torch.compile`` mode. Defaults to ``None``, in which case compilation is disabled.
        """
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        """Construct the PPO algorithm."""
        # Resolve class callables
        alg_class: type[PPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Initialize the policy
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        # Initialize the storage (sized for the eval pool, if any)
        storage = RolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
            num_eval_envs=PPO._num_eval_envs(cfg["algorithm"].get("eval_env_fraction", 0.0), env.num_envs),
        )

        # Initialize the algorithm
        alg: PPO = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters and buffers to all GPUs."""
        modules = [self._raw_actor, self._raw_critic]
        if self.rnd:
            modules.append(self.rnd.predictor)
        for module in modules:
            for tensor in chain(module.parameters(), module.buffers()):
                torch.distributed.broadcast(tensor.data, src=0)

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        all_params = chain(self.actor.parameters(), self.critic.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel

    def _build_noise_scale_tracker(self, cfg: dict | None) -> GradientNoiseScaleTracker | None:
        """Resolve the gradient-noise-scale config and return a tracker (or ``None`` if disabled).

        ``cfg=None`` and ``cfg={"enabled": False}`` both disable the metric; any other
        cfg enables it. ``mode='auto'`` picks ``'ddp_native'`` under multi-GPU and
        ``'across_minibatches'`` otherwise. Explicit ``mode='ddp_native'`` is rejected
        when not multi-GPU because the unbiased estimator would divide by zero
        (``b_big == b_small`` when ``world_size == 1``).
        """
        if cfg is None:
            return None
        cfg = dict(cfg)  # shallow copy so we don't mutate the caller's dict
        if not cfg.pop("enabled", True):
            return None
        mode = cfg.pop("mode", "auto")
        if mode == "auto":
            mode = "ddp_native" if self.is_multi_gpu else "across_minibatches"
        if mode == "ddp_native" and not self.is_multi_gpu:
            raise ValueError(
                "gradient noise scale mode='ddp_native' requires multi-GPU; "
                "use mode='auto' or mode='across_minibatches' on a single GPU."
            )
        if mode == "within_minibatch":
            raise NotImplementedError(f"gradient noise scale mode={mode!r} is not implemented yet.")
        ema_decay = cfg.pop("ema_decay", 0.99)
        eps = cfg.pop("eps", 1e-12)
        cfg.pop("num_micro_shards", None)  # accepted for forward-compat with within_minibatch mode
        if cfg:
            raise ValueError(f"unrecognized keys in gradient_noise_scale_cfg: {sorted(cfg.keys())}")
        return GradientNoiseScaleTracker(
            mode=mode,
            ema_decay=ema_decay,
            is_multi_gpu=self.is_multi_gpu,
            gpu_world_size=self.gpu_world_size,
            device=self.device,
            eps=eps,
        )

    def _actor_critic_parameters(self) -> list[torch.nn.Parameter]:
        """Return actor + critic parameters (excluding RND) for the noise-scale tracker."""
        return list(chain(self.actor.parameters(), self.critic.parameters()))

    @staticmethod
    def _grad_norm_sq(params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
        """Return ``sum_p |p.grad|^2`` as a scalar tensor on the gradient's device.

        Skips parameters with ``p.grad is None``. Used by T-1's per-minibatch grad-norm
        mean/variance logging; the gradient-noise-scale tracker keeps its own copy of this
        helper to stay self-contained.
        """
        total: torch.Tensor | None = None
        for p in params:
            if p.grad is None:
                continue
            sq = p.grad.detach().pow(2).sum()
            total = sq if total is None else total + sq
        if total is None:
            raise RuntimeError("no gradients found on the supplied parameters.")
        return total
