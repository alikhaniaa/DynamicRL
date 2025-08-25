import torch
import torch.optim as optim
import numpy as np
import gymnasium as gym
from typing import Any
import logging
from omegaconf import DictConfig


from dynamicrl.core.algorithm import RLAlgorithm
from dynamicrl.core.advantage import compute_gae_advantages
from dynamicrl.core.data_buffer import RolloutBuffer
from dynamicrl.core.policies import ActorCriticMLP
from dynamicrl.core.types import Batch, ObservationType

logger = logging.getLogger(__name__)

"""
    PPO implementation with logivs of policy, buffer, uptimizer and update
"""
class PPOAlgorithm(RLAlgorithm):
    def __init__(self, cfg: DictConfig, obs_space: gym.spaces.Space, act_space: gym.spaces.Space):
        super().__init__(cfg, obs_space, act_space)
        self.cfg = cfg
        
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cuda") #TODO CHANGE THIS LATER
        
        algo_cfg = cfg.algorithm
        training_cfg = cfg.training
        # PPO default hyperparams
        self.gamma = algo_cfg.gamma
        self.gae_lambda = algo_cfg.gae_lambda
        self.clip_epsilon = algo_cfg.clip_epsilon
        self.entropy_coef = algo_cfg.entropy_coef
        self.value_coef = algo_cfg.value_coef
        self.lr = training_cfg.learning_rate
        self.num_epochs = algo_cfg.num_epochs
        self.minibatch_size = algo_cfg.minibatch_size
        self.rollout_steps = training_cfg.rollout_steps
        self.last_observation: ObservationType | None = None
        
        # Component initialize
        self.policy: ActorCriticMLP = ActorCriticMLP(
            obs_dim=obs_space.shape[0],
            act_space=act_space,
            hidden_sizes=tuple(self.cfg.model.policy_hidden_dims),
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr = self.lr)
        
        is_continuous = isinstance(act_space, gym.spaces.Box)
        buffer_act_dim = act_space.shape[0] if is_continuous else 1
        
        self.buffer = RolloutBuffer(
            buffer_size=self.rollout_steps,
            num_envs=self.cfg.env.num_envs,
            obs_dim=obs_space.shape[0],
            act_dim=buffer_act_dim,
            device=self.device,
        )
        logger.info("PPO Algorithm initialized.")
        
    #Runs the data collection loop for a full rollout
    def collect_experiences(self, env: Any, current_obs: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        episode_rewards = []
        self.buffer.reset()
        
        for _ in range(self.rollout_steps):
            obs_tensor = torch.as_tensor(current_obs, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                action, log_prob = self.policy.forward(obs_tensor)
                value = self.policy.predict_value(obs_tensor)
                
            action_np = action.cpu().numpy()
            #Env interaction
            next_obs, reward, terminated, truncated, infos = env.step(action_np)
            done = np.logical_or(terminated, truncated)
            
            self.buffer.add(
                obs=current_obs,
                action=action_np,
                reward=reward,
                done=done,
                log_prob=log_prob.cpu().numpy(),
                value=value.cpu().numpy(),
            )
            current_obs = next_obs.astype(np.float32)
            
            # if done: TODO SHOULD DYNAMIC THIS LATER, FOR NOW OPTIMIZE IT FOR GYM
            #     if info.get("final_info"):
            #         for item in info.get("final_info", []):
            #             if item and "episode" in item:
            #                 episode_rewards.append(item["episode"]["r"].item())
                
            #     current_obs, _ = env.reset()
            #     current_obs = current_obs.astype(np.float32)
            if "final_info" in infos:
                for info in infos.get("final_info"):
                    if info and "episode" in info:
                        episode_rewards.append(info["episode"]["r"])
                        
        self.last_observation = current_obs
        collection_metrics = { "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0 }
        
        return current_obs, collection_metrics
        
    
    
    
    # Run the PPO update using the data in the buffer
    def update_policy(self) -> dict[str, float]:
        #Getting the raw data and compute advantage
        unflattened_data = self.buffer.get_all_unflattened()
        # raw_data = self.buffer.get_all()
        
        #get the last state of GAE value
        with torch.no_grad():
            last_obs_tensor = torch.as_tensor(self.last_observation, dtype=torch.float32, device=self.device)
            last_value = self.policy.predict_value(last_obs_tensor)
            
        advantages, returns = compute_gae_advantages(
            rewards=unflattened_data["rewards"],
            dones=unflattened_data["dones"],
            values=unflattened_data["values"],
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        
        flattened_data = self.buffer.get_all_flattened()
        #Gather all data into a Batch object(for easy access)
        full_batch = Batch(
            observations=flattened_data["observations"],
            actions=flattened_data["actions"],
            log_probs=flattened_data["log_probs"],
            values=flattened_data["values"],
            advantages=advantages.flatten(),
            returns=returns.flatten(),
            rewards=flattened_data["rewards"],
            dones=flattened_data["dones"],
        )
        
        #*PPO Update Loop
        policy_losses, value_losses, entropy_bonuses = [], [], []
                
        for _ in range(self.num_epochs):
            num_samples = self.rollout_steps * self.cfg.env.num_envs
            indices = torch.randperm(num_samples)
            
            for start in range(0, num_samples, self.minibatch_size):
                end = start + self.minibatch_size
                batch_indices = indices[start:end]
                
                minibatch = Batch(**{k: getattr(full_batch, k)[batch_indices] for k in full_batch.__annotations__ if hasattr(full_batch, k)})
                #Normalize advantage for the stability of minibatch
                mb_advantages = (minibatch.advantages - minibatch.advantages.mean()) / (minibatch.advantages.std() + 1e-8)
                
                new_log_probs, entropy, new_values = self.policy.evaluate_actions(
                    minibatch.observations, minibatch.actions
                )
                
                # #Create minibatch by slicing it
                # minibatch = Batch(
                #     observations=full_batch.observations[batch_indices],
                #     actions=full_batch.actions[batch_indices],
                #     rewards=full_batch.rewards[batch_indices],
                #     dones=full_batch.dones[batch_indices],
                #     log_probs=full_batch.log_probs[batch_indices],
                #     values=full_batch.values[batch_indices],
                #     advantages=full_batch.advantages[batch_indices],
                #     returns=full_batch.returns[batch_indices]
                #     # next_observations=full_batch.next_observations
                # )
                
                
                #*Policy Loss
                ratio = torch.exp(new_log_probs - minibatch.log_probs)
                surr1 = mb_advantages * ratio
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                print(f"DEBUG - Type of new_values: {type(new_values)}, Shape: {new_values.shape}")
                print(f"DEBUG - Type of minibatch.returns: {type(minibatch.returns)}, Shape: {minibatch.returns.shape}")
                #*Value Loss
                value_loss = ((new_values - minibatch.returns) ** 2).mean()
                entropy_bonus = entropy.mean()

                total_loss = (policy_loss+ self.value_coef * value_loss- self.entropy_coef * entropy_bonus)

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_bonuses.append(entropy_bonus.item())

        return {"policy_loss": sum(policy_losses) / len(policy_losses), "value_loss": sum(value_losses) / len(value_losses), "entropy": sum(entropy_bonuses) / len(entropy_bonuses)}

                
    def get_state(self) -> dict[str, Any]:
        """Returns the current state for checkpointing."""
        return {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load_state(self, state: dict[str, Any]):
        """Loads state from a checkpoint."""
        self.policy.load_state_dict(state["policy_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])            