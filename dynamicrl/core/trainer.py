import logging
import time
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from dynamicrl.algorithms.ppo.ppo_algorithm import PPOAlgorithm
from dynamicrl.core.algorithm import RLAlgorithm
from dynamicrl.core.checkpoint import CheckpointManager
from dynamicrl.core.control import HyperparamServer
from dynamicrl.core.events import (CheckpointReq, EventBus, Pause, Quit, Resume)
from dynamicrl.core.logger import DataLogger
from dynamicrl.core.utils import set_seed
from dynamicrl.envs.vec.sync_vec import SyncVecEnv

logger = logging.getLogger(__name__)
ALGORITHM_REGISTRY = {"ppo": PPOAlgorithm}

"""
    Orchestrator of RL training process, initialize all components, running training loop and ...
"""
class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.is_paused = False
        self.should_quit = False
        
        self._setup_services()
        self.global_step = 0
        self.total_timesteps = self.cfg.training.total_timesteps
        self.last_checkpoint_step = 0
        
        logger.info("Trainer initialization complete")
        
    def _setup_services(self) -> None:
        set_seed(self.cfg.training.seed)
        self.device = torch.device("cpu")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") TODO UNCOMMENT THIS IN PRODUCTION
        
        self.output_dir = Path.cwd()
        logger.info(f"Run output directory: {self.output_dir}")
        
        #services
        OmegaConf.update(self.cfg, "output_dir", str(self.output_dir))
        self.data_logger = DataLogger(self.cfg)
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.output_dir / "checkpoints",
            max_to_keep=self.cfg.training.get("checkpoints_to_keep", 5),
        )
        self.event_bus = EventBus()
        self.hyperparam_server = HyperparamServer(initial_config=self.cfg.copy())
        
        #Envs
        env_fns = [
            lambda: gym.make(self.cfg.env.name)
            for _ in range(self.cfg.env.num_envs)
        ]
        self.env = SyncVecEnv(env_fns)
        
        #Algorithm
        algo_name = self.cfg.algorithm.name
        AlgorithmClass = ALGORITHM_REGISTRY.get(algo_name)
        if not AlgorithmClass:
            raise ValueError(f"Unknown algorithm: '{algo_name}'")
        self.algorithm: RLAlgorithm = AlgorithmClass(
            cfg=self.cfg, # Use 'cfg' to match the PPOAlgorithm's definition
            obs_space=self.env.observation_space,
            act_space=self.env.action_space,
        )
        logger.info(f"Algorithm '{algo_name}' initialized successfully.")
        
    #**Taining loop
    def train(self) -> None:
        logger.info(f"Starting trainig for {self.total_timesteps} timesteps")
        obs, _ = self.env.reset()
        obs = obs.astype(np.float32)
        
        #pause points between iterations
        while self.global_step < self.total_timesteps and not self.should_quit:
            self._handle_control_events()
            if self.should_quit: break
            
            if self.is_paused:
                logger.debug("Training is paused. Waiting for Resume event...")
                time.sleep(0.5)
                continue
            
            #data collection
            rollout_start_time = time.time()
            next_obs, collection_metrics = self.algorithm.collect_experiences(
                self.env, obs
            )
            steps_this_rollout = self.algorithm.rollout_steps * self.cfg.env.num_envs
            self.global_step += steps_this_rollout
            obs = next_obs
            
            #logging
            fps = steps_this_rollout / (time.time() - rollout_start_time)
            metrics_to_log = {"fps": fps, **collection_metrics}
            
            self._handle_control_events()
            if self.should_quit: break
            if self.is_paused: continue
            
            #policy update
            update_metrics = self.algorithm.update_policy()
            metrics_to_log.update(update_metrics)
            self.data_logger.log(metrics_to_log, self.global_step)
            
            #checkpoints
            checkpoint_interval = self.cfg.training.get("checkpoint_interval", 0)
            if checkpoint_interval > 0 and self.global_step >= self.last_checkpoint_step + checkpoint_interval:
                self._save_checkpoint()
                self.last_checkpoint_step = self.global_step
                
        self._save_checkpoint(tag="final")
        self.close()
        logger.info("Training finished")
    def _handle_control_events(self) -> None:
        pending_events = self.event_bus.get_all_pending_sync()
        if not pending_events:
            return
        
        logger.debug(f"Handling {len(pending_events)} control events...")
        for envelope in pending_events:
            event = envelope.event
            if isinstance(event, Pause):
                self.is_paused = True
                logger.info(f"\x1b[1m\x1b[7m[PAUSE]\x1b[0m Training paused: {event.reason}")
            elif isinstance(event, Resume):
                if self.is_paused:
                    self._apply_staged_patches()
                    self.is_paused = False
                    logger.info("\x1b[1m\x1b[7m[RESUME]\x1b[0m Training resumed")
            elif isinstance(event, CheckpointReq):
                self._save_checkpoint(tag=event.tag)
            elif isinstance(event, Quit):
                self.should_quit = True
                logger.info("\x1b[1m\x1b[7m[QUIT]\x1b[0m Shutting down gracefully")
                
    def _apply_staged_patches(self):
        staged_batch = self.hyperparam_server.get_staged_batch()
        if not staged_batch:
            return
        
        version, patches = staged_batch
        logger.info(f"applying staged hyperparameter batch v{version}...")            
        for patch in patches:
            try: 
                OmegaConf.update(self.cfg, patch.path, patch.value)
            except Exception as e:
                logger.error(f"Failed to apply patch '{patch.path}': {e}")
                
        self.hyperparam_server.confirm_applied(version)
        logger.info("Hyperparameter patches applied successfully")
    
    def _save_checkpoint(self, tag:str | None = None):
        if tag is None:
            tag = f"step_{self.global_step}"
            
        trainer_state = {
            "global_step": self.global_step,
            "config": self.cfg,
            #TODO mybae add RNG later
            **self.algorithm.get_state(),
        }
        self.checkpoint_manager.save(trainer_state, tag)
        
    def close(self) -> None:
        logger.info("Closing all services.")
        self.data_logger.close()
        self.env.close()
        