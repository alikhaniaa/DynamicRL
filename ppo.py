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
    wandb_project_name: str = "DynamicRL"
    wandb_entity: str = None  
    capture_video: bool = True
    
    # -------------------------------------------------------------------
    # Environment Configuration
    # -------------------------------------------------------------------
    env_id: str = "CartPole-v1"
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


'''
    IMPLEMENTING BASIC PPO FOR NOW
'''
    
# Helper function to create a single env instance
def make_env(env_id, idx, run_home):
    def thunk():
        # TODO add video capturing later(dynamic since I need it a lot)
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        print(env)
        return env
    return thunk
    
# helper to initialize network layers
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    
# agent class to define NN architecture contain both actor and critic(policy and val function)
class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        
        # Critic network estimation of the value of alpha state
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0)
        )
    
        #Actor network decides which action to take
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.signel_action_space.n), std=0.01)
        )
        
    def get_value(self, x):
        return self.critic(x)
        
    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatch)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    # tensorboard for logging --> 'tensorboard --logdir runs'
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()])),
    )
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)
    
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    