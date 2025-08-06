import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    # -------------------------------------------------------------------
    # Experiment Setup
    # -------------------------------------------------------------------
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1  
    torch_deterministic: bool = True
    cuda: bool = True
    
    # -------------------------------------------------------------------
    # Logging & Visualization
    # -------------------------------------------------------------------
    track: bool = True
    wandb_project_name: str = "ppoManipulation"
    wandb_entity: str = None  
    capture_video: bool = True
    
    # -------------------------------------------------------------------
    # Environment Configuration
    # -------------------------------------------------------------------
    env_id: str = "NOT DEFINED FOR NOW, MAYBE MUJOCO"
    total_timesteps: int = 100000
    # Number of parallel game environmets
    num_envs: int = 4
    # Number of steps in each episode
    num_steps: int = 128
    
    # -------------------------------------------------------------------
    # Hyperparameters for TWEAKING
    # This is the section you will manipulate during training stages.
    # -------------------------------------------------------------------
    learning_rate: float = 2.5e-4
    # Toggle learning rate annealing for policy and value networks
    anneal_lr: bool = True
    gamma: float = 0.99
    # Lambda for the general advantage estimation
    gae_lambda: float = 0.95
    num_minibatch: int = 4
    update_epochs: int = 4
    # Toggle advantage normalization
    norm_adv: bool = True
    # Surrogate clipping coefficient
    clip_coef: float = 0.1
    # Toggle where or not to use a clipped loss for the value function
    clip_vloss: bool = True
    # Coefficient of the entropy
    ent_coef: float = 0.01
    # Coefficient of the value function
    vf_coef: float = 0.5
    # Maximum norm for the gradient clipping
    max_grad_norm: float = 0.5
    # Target KL diveregence threshold
    target_kl: float = None

    # -------------------------------------------------------------------
    # Runtime-Computed Values (do not change these)
    # -------------------------------------------------------------------
    # Filled in the runtime
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0
