import torch
import torch.optim as optim
import numpy as np
import gymnasium as gym

from dynamicrl.core.algorithm import RLAlgorithm
from dynamicrl.core.advantage import compute_gae_advantages
from dynamicrl.core.data_buffer import RolloutBuffer
from dynamicrl.core.policies import ActorCriticMLP
from dynamicrl.core.types import Batch, ObservationType

"""
    PPO implementation with logivs of policy, buffer, uptimizer and update
"""
class PPOAlgorithm(RLAlgorithm):
    def __init__(self, config: dict[str, any], obs_space: any, act_space: any):
        super().__init__(config, obs_space, act_space)
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cuda") #TODO CHANGE THIS LATER
        
        algo_config = self.config.get("algorithm", {})
        training_config = self.config.get("training", {})
        
        # PPO default hyperparams
        self.gamma = algo_config.get("gamma", 0.99)
        self.gae_lambda = algo_config.get("lambda", 0.95) 
        self.clip_range = algo_config.get("clip_range", 0.2)
        self.ent_coef = algo_config.get("entropy_coef", 0.01) 
        self.vf_coef = algo_config.get("value_loss_coef", 0.5)
        self.lr = algo_config.get("learning_rate", 3e-4) 
        self.num_epochs = training_config.get("epochs_per_update", 10)
        self.minibatch_size = training_config.get("batch_size", 64)
        self.rollout_steps = training_config.get("rollout_length", 2048) 
        self.last_observation: ObservationType = None
        
        # Component initialize
        self.policy: ActorCriticMLP = ActorCriticMLP(
            obs_dim = obs_space.shape[0],
            act_dim = act_space.shape[0],
            hidden_sizes = self.config.get("hidden_sizes", (64, 64))
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr = self.lr)
        
        self.buffer = RolloutBuffer(
            buffer_size = self.rollout_steps,
            obs_dim = obs_space.shape[0],
            act_dim = act_space.shape[0],
            device=self.device
        )
        
    #Runs the data collection loop for a full rollout
    def collect_experiences(self, env: gym.Env, current_obs: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        episode_rewards = []
        
        for _ in range(self.rollout_steps):
            obs_tensor = torch.as_tensor(current_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob = self.policy.forward(obs_tensor)
                value = self.policy.predict_value(obs_tensor)
                
            action_np = action.cpu().numpy().squeeze(0)
            #Env interaction
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            
            self.buffer.add(
                obs = current_obs,
                action = action_np,
                reward = reward,
                done = done,
                log_prob = log_prob.item(),
                value = value.item()
            )
            current_obs = next_obs.astype(np.float32)
            
            # if done: TODO SHOULD DYNAMIC THIS LATER, FOR NOW OPTIMIZE IT FOR GYM
            #     if info.get("final_info"):
            #         for item in info.get("final_info", []):
            #             if item and "episode" in item:
            #                 episode_rewards.append(item["episode"]["r"].item())
                
            #     current_obs, _ = env.reset()
            #     current_obs = current_obs.astype(np.float32)
            if done:
                if "episode" in info: 
                    episode_rewards.append(info["episode"]["lr"])
                current_obs, _ = env.reset()
                        
        self.last_observation = current_obs
        collection_metrics = {
            "mean_reward": sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
        }
        
        return current_obs.astype(np.float32), collection_metrics
        
    
    
    
    # Run the PPO update using the data in the buffer
    def update_policy(self) -> dict[str, float]:
        #Getting the raw data and compute advantage
        raw_data = self.buffer.get_all()
        
        #get the last state of GAE value
        with torch.no_grad():
            last_obs_tensor = torch.as_tensor(self.last_observation, device=self.device).unsqueeze(0)
            last_value = self.policy.predict_value(last_obs_tensor)
            
        advantage, returns = compute_gae_advantages(
            rewards=raw_data["rewards"],
            dones=raw_data["dones"],
            values=raw_data["values"],
            last_value=last_value,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        
        #Gather all data into a Batch object(for easy access)
        full_batch = Batch(
            observations=raw_data["observations"],
            actions=raw_data["actions"],
            rewards=raw_data["rewards"],
            dones=raw_data["dones"],
            log_probs=raw_data["log_probs"],
            values=raw_data["values"],
            advantages=advantage,
            returns=returns
            # next_observations=torch.zeros(0)
        )
        
        #*PPO Update Loop
        policy_losses, value_losses, entropy_bonuses = [], [], []
        
        indices = torch.randperm(self.rollout_steps)
        
        for _ in range(self.num_epochs):
            for start in range(0, self.rollout_steps, self.minibatch_size):
                end = start + self.minibatch_size
                batch_indices = indices[start:end]
                minibatch = Batch(**{k: getattr(full_batch, k)[batch_indices] for k in full_batch.__annotations__})

                #Normalize advantage for the stability of minibatch
                mb_advantage = (minibatch.advantages - minibatch.advantages.mean()) / (minibatch.advantages.std() + 1e-8)
                
                new_log_probs, entropy, new_values = self.policy.evaluate_actions(
                    minibatch.observations, minibatch.actions
                )
                
                #Create minibatch by slicing it
                minibatch = Batch(
                    observations=full_batch.observations[batch_indices],
                    actions=full_batch.actions[batch_indices],
                    rewards=full_batch.rewards[batch_indices],
                    dones=full_batch.dones[batch_indices],
                    log_probs=full_batch.log_probs[batch_indices],
                    values=full_batch.values[batch_indices],
                    advantages=full_batch.advantages[batch_indices],
                    returns=full_batch.returns[batch_indices]
                    # next_observations=full_batch.next_observations
                )
                
                
                #*Policy Loss
                ratio = torch.exp(new_log_probs - minibatch.log_probs)
                surr1 = mb_advantage * ratio
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantage
                policy_loss = -torch.min(surr1, surr2).mean()
                
                print(f"DEBUG - Type of new_values: {type(new_values)}, Shape: {new_values.shape}")
                print(f"DEBUG - Type of minibatch.returns: {type(minibatch.returns)}, Shape: {minibatch.returns.shape}")
                #*Value Loss
                value_loss = ((new_values - minibatch.returns) ** 2).mean()
                entropy_bonus = entropy.mean()

                total_loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy_bonus

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_bonuses.append(entropy_bonus.item())

        return {"policy_loss": sum(policy_losses) / len(policy_losses), "value_loss": sum(value_losses) / len(value_losses), "entropy": sum(entropy_bonuses) / len(entropy_bonuses)}

                
    def get_state(self) -> dict[str, any]:
        """Returns the current state for checkpointing."""
        return {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load_state(self, state: dict[str, any]):
        """Loads state from a checkpoint."""
        self.policy.load_state_dict(state["policy_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])            