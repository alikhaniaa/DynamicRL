import logging
import time
import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig

from dynamicrl.core.algorithm import RLAlgorithm
from dynamicrl.core.control import HyperparamServer
from dynamicrl.core.events import EventBus, Pause, Resume
from dynamicrl.core.policies import ActorCriticMLP
from dynamicrl.core.utils import set_seed
from dynamicrl.envs.vec.sync_vec import SyncVecEnv

ALGORITHM_REGISTRY = {"ppo": PPOAlgorithm}
logger = logging.getLogger(__name__)

"""
    RL training process. responsible for initialize components, running training loop and handle interactive events
"""
class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.is_paused = False
        logger.info("__Initializeing Trainer__")
        
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu") #TODO change this for the production
        logger.info(f"Used device: {self.device}")
        
        logger.info(f"Creating env: {self.cfg.env.name}")
        self.env = gym.make(self.cfg.env.name) #TODO should optimize this for parallel envs
        
        algo_name = self.cfg.algorithm.name
        logger.info(f"Instantiating algorithm: '{algo_name}'")
        if algo_name not in ALGORITHM_REGISTRY:
            raise ValueError(f"Unknown algorithm: '{algo_name}'")
        AlgorithmClass = ALGORITHM_REGISTRY[algo_name]
        self.algorithm: RLAlgorithm = AlgorithmClass(
            config=self.cfg,
            obs_space=self.env.observation_space,
            act_space=self.env.action_space
        )
        logger.info(f"Algorithm '{algo_name}' initialized successfully")
        
        self.event_bus = EventBus()
        self.hyperparam_server = HyperparamServer(initial_config=cfg)
        #TODO implement checkpoint manager here later if I wrote that too
        
        self.global_step = 0
        self.total_timesteps = self.cfg.training.total_time_steps
        
        logger.info("Trainer initialization completed")
        
    """
        *main training loop
    """
    def train(self) -> None:
        logger.info(f"Starting training for {self.total_timesteps} timesteps")
        obs, _ = self.env.reset(seed=self.cfg.training.seed)
        obs = obs.astype(np.float32)
        
        # **Pause point
        while self.global_step < self.total_timesteps:
            self._handle_control_events()
            
            if self.is_paused:
                time.sleep(0.1)
                continue
            
            collection_start_time = time.time()
            next_obs, collection_metrics = self.algorithm.collect_experiences(self.env, obs)
            steps_collected = self.algorithm.rollout_steps
            self.global_step += steps_collected
            obs = next_obs
            collection_end_time = time.time()
            
            fps = steps_collected / (collection_end_time - collection_start_time)
            logger.info(
                f"Step: {self.global_step} / {self.total_timesteps} | "
                f"Mean Reward: {collection_metrics.get('mean_reward', 0):.2f} |"
                f"FPS: {fps:.0f}"
            )
            
            self._handle_control_events()
            if self.is_paused: continue
            
            #Policy update
            update_metrics = self.algorithm.update_policy()
            logger.debug(f"Update Metrics: {update_metrics}")
            #TODO add checkpoints here too
            
        self.close()
        logger.info("Training complete")
        
    #Check the event-bus for new control events and process them, uses at every safe point
    def _handle_control_events(self) -> None:
        pass
    
    #cleans up resource
    def close(self) -> None:
        logger.info("Closing trainer and env")
        self.env.close()