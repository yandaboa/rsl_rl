# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from ..storage.rollout_storage import RolloutStorage
from rsl_rl.algorithms import PPO

class BCPPO(PPO):
    """Behavior Cloning + Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    policy: ActorCritic | ActorCriticRecurrent
    """The actor critic module."""

    def __init__(
        self,
        policy: ActorCritic | ActorCriticRecurrent,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        device: str = "cpu",
        normalize_advantage_per_mini_batch: bool = False,
        # BC parameters
        behavior_cloning_cfg: dict | None = None,
        # Offline configuration
        offline_algorithm_cfg: dict | None = None,
        **kwargs,
    ):
        print("Extra kwargs passed to BCPPO:", kwargs)
        super().__init__(
            policy=policy,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            device=device,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
        )

        assert behavior_cloning_cfg is not None, "BC config must be provided for BCPPO."
        assert len(behavior_cloning_cfg["experts_path"]) <= 1, "BC PPO only supports single expert currently."

        # Online configurations
        # BC components
        self.bc = behavior_cloning_cfg
        self.expert_obs_fn = self.bc["experts_observation_func"]
        loader = self.bc["experts_loader"]
        if not callable(loader):
            loader = eval(loader)
        self.expert = loader(self.bc["experts_path"][0]).to(self.device)
        self.expert.eval()

        self.bc_loss_coeff = self.bc["cloning_loss_coeff"]
        # self.bc_decay = 0.99999 #self.bc["loss_decay"]
        self.bc_decay = None
        self.learn_std = self.bc["learn_std"]
        self.min_bc_coeff = 0.2
        
        # self.advisor_alpha = 4.0
        self.advisor_loss = False #behavior_cloning_cfg["advisor_loss"]
    
    def set_bc_decay(self, num_learning_iterations: int) -> None:
        """Set the bc decay based on the number of learning iterations.
            mini batch size and num learning epochs.
        Args:
            num_learning_iterations (int): Number of learning iterations.
        """
        total_updates = num_learning_iterations * self.num_learning_epochs * self.num_mini_batches
        self.bc_decay = (self.min_bc_coeff / self.bc_loss_coeff) ** (1 / total_updates)
        

    def init_storage(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
    ) -> None:

        # create rollout storage
        self.storage = RolloutStorage(
            "bc_rl",
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            self.device,
        )

    def act(self, obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        
        # Compute the actions and values
        # Compute the actions and values
        self.transition.actions = self.policy.act(obs).detach()
        policy_obs = obs.pop("policy")
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        
        # Record observations before env.step()
        self.transition.observations = obs
        self.transition.policy_observations = policy_obs
        
        # expert_obs = self.expert_obs_fn(self.bc['_env'])
        expert_obs = obs['expert']
        mean, std = self.expert.compute_distribution(expert_obs)
        self.transition.privileged_actions_mean = mean.detach()
        self.transition.privileged_actions_std = std.detach()

        return self.transition.actions

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # -- BC loss
        if self.bc:
            mean_bc_loss = 0
        else:
            mean_bc_loss = None
        
        if self.advisor_loss:
            mean_advisor_weight = 0
            mean_auxillary_loss = 0
            mean_advisor_loss = 0
        else:
            mean_advisor_weight = None
            mean_auxillary_loss = None
            mean_il_loss = None
            mean_advisor_loss = None

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        # iterate over batches
        # Iterate over batches
        for (
            obs_batch,
            policy_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
            expert_action_mu_batch,
            expert_action_sigma_batch,
        ) in generator:

            original_batch_size = obs_batch.shape[0]

            # match format of policy_obs_batch with the one that actor_critic expects
            # TODO: fix the batch generation so that this is not necessary
            policy_obs_batch = {"policy": policy_obs_batch}

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.policy.act(policy_obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
            # Note: We only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped)

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()
            
            
            # imitation loss
            if self.advisor_loss:
                pg_loss = surrogate_loss - self.entropy_coef * entropy_batch
                
                # CE loss
                self.policy.auxillary_act(policy_obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
                aux_actions_log_prob_batch = self.policy.get_auxillary_actions_log_prob(expert_action_mu_batch)                
                # get kl-div between: (expert_action_mu_batch, expert_action_sigma_batch)
                aux_mu_batch = self.policy.auxillary_action_mean
                aux_sigma_batch = self.policy.auxillary_action_std
                auxillary_loss = -0.01 * aux_actions_log_prob_batch.mean()
                
                eps = 1e-8
                kl_div = (
                    torch.log((aux_sigma_batch + eps) / (expert_action_sigma_batch + eps))
                    + (expert_action_sigma_batch.pow(2) + (expert_action_mu_batch - aux_mu_batch).pow(2))
                    / (2.0 * (aux_sigma_batch + eps).pow(2))
                    - 0.5
                ).sum(dim=-1)   # (B,)
                advisor_weight = torch.exp(-0.001 * kl_div).detach()
                
                # mse_loss = torch.nn.MSELoss(reduction="none")
                mean_loss = ((mu_batch - expert_action_mu_batch)**2).mean(-1)
                bc_loss = mean_loss
                if self.learn_std:
                    std_loss = ((sigma_batch - expert_action_sigma_batch)**2).mean(-1)
                    bc_loss += std_loss
                
                advisor_loss = (1 - advisor_weight) * pg_loss + advisor_weight * 0.01 * bc_loss
                loss = advisor_loss.mean() + auxillary_loss
                
                mean_advisor_weight += advisor_weight.mean().item()
                mean_advisor_loss += advisor_loss.mean().item()
                mean_auxillary_loss += auxillary_loss.item()
                bc_loss_stat = bc_loss.mean().item()
            else:
                pg_loss = surrogate_loss.mean() - self.entropy_coef * entropy_batch.mean()
                
                mse_loss = torch.nn.MSELoss()
                mean_loss = mse_loss(mu_batch, expert_action_mu_batch)
                bc_loss = mean_loss
                if self.learn_std:
                    std_loss = mse_loss(sigma_batch, expert_action_sigma_batch)
                    bc_loss += std_loss
                self.bc_loss_coeff *= self.bc_decay
                self.bc_loss_coeff = max(self.bc_loss_coeff, self.min_bc_coeff)
                loss = (1 - self.bc_loss_coeff) * pg_loss + self.bc_loss_coeff * bc_loss
                bc_loss_stat = bc_loss.item()
            
            # value loss
            loss += self.value_loss_coef * value_loss

            # Gradient step
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.mean().item()
            mean_entropy += entropy_batch.mean().item()
            # -- BC loss
            if mean_bc_loss is not None:
                mean_bc_loss += bc_loss_stat

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        # -- For BC
        if mean_bc_loss is not None:
            mean_bc_loss /= num_updates        # -- Clear the storage
        else:
            mean_bc_loss = 0.0
        
        if mean_advisor_loss is not None:
            mean_advisor_weight /= num_updates
            mean_advisor_loss /= num_updates
            mean_auxillary_loss /= num_updates
        else:
            mean_advisor_weight = 0.0
            mean_advisor_loss = 0.0
            mean_auxillary_loss = 0.0
        
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "bc_loss": mean_bc_loss,
            "bc_loss_coeff": self.bc_loss_coeff,
            "advisor_loss": mean_advisor_loss,
            "auxillary_loss": mean_auxillary_loss,
            "advisor_weight": mean_advisor_weight
        }
        return loss_dict
