from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, NamedTuple
import torch

"""
    this module defines immutable and standardized DS for the entire framework
"""
ObservationType = torch.Tensor
ActionType = torch.Tensor
RewardType = torch.Tensor
DoneType = torch.Tensor
LogProbType = torch.Tensor
ValueType = torch.Tensor

# Batch class repressent a batch of transitions that are structured for efficient processing.(basically an allias)
# Data stores as a collection of torch.Tensors
@dataclass(frozen=True)
class Batch:
    observations: ObservationType
    actions: ActionType
    rewards: RewardType
    dones: DoneType
    next_observations: ObservationType
    log_probs: LogProbType
    values: ValueType
    advantages: torch.Tensor
    returns: torch.Tensor
    
    #number of transitions in batch
    def __len__(self) -> int:
        return self.observations.shape[0]
    
    #move tensors of a batch to specified device
    def to(self, device: torch.device) -> Batch:
        return Batch(
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            next_observations=self.next_observations.to(device),
            log_probs=self.log_probs.to(device),
            values=self.values.to(device),
            advantages=self.advantages.to(device),
            returns=self.returns.to(device),
        )
    