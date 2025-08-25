import torch
from .types import DoneType, RewardType, ValueType

"""
    Computes Generalized Advantage Estimation (GAE) for a batch of trajectories.
"""
def compute_gae_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_steps, num_envs = rewards.shape
    # buffer_size = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae_lam = torch.zeros(num_envs, device=rewards.device)

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]

        next_non_terminal = 1.0 - dones[t]
        
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        advantages[t] = last_gae_lam

    returns = advantages + values
    return advantages, returns