import gymnasium as gym
import numpy as np
import torch 
from typing import Any, Dict
import asyncio

from .algorithm import RLAlgorithm
from .events import EventBus, Pause, Resume, Quit, CheckpointReq, ParamPatch, PatchBatch
from .control import HyperparamServer

"""
    Trainer will initialze all system components and run the main training loop.
    All the control-plane implemented here
"""
class Trainer:
    def __init__(self, config: Dict[str, Any], algorithm: RLAlgorithm, env: gym.Env, event_bus: EventBus, hyperparam_server: HyperparamServer):
        self.config = config
        self.algorithm = algorithm
        self.env = env
        self.event_bus = event_bus
        self.hyperparam_server = hyperparam_server
        
        #Training state
        self.total_timesteps = self.config["training"]["total_timesteps"]
        self.current_timesteps = 0
        self.is_paused = False
        self.should_quit = False
        
        #Env initialization
        seed = self.config["training"]["seed"]
        self.current_obs, _ = self.env.reset(seed=seed)
        self.current_obs = self.current_obs.astype(np.float32)
        
    #*Polls the event bus and process any pending control commands
    async def _handle_control_events(self):
        events = await self.event_bus.get_all_pending()
        for envelope in events:
            event = envelope.event
            if isinstance(event, Pause):
                self.is_paused = True
            elif isinstance(event, Resume):
                self.is_paused = False
            elif isinstance(event, Quit):
                self.should_quit = True
            elif isinstance(event, ParamPatch):
                self.hyperparam_server.stage_patches([event]) #TODO Add multibatch option later
            elif isinstance(event, PatchBatch):
                self.hyperparam_server.stage_patches(event.patches)
                
    #Apply validated and staged hyperparam patches                
    async def _apply_staged_patches(self):
        staged_batch = self.hyperparam_server.get_staged_batch()
        if staged_batch:
            version, patches = staged_batch
            for patch in patches:
                if patch.path == "algorithm.learning_rate":
                    self.algorithm.optimizer.param_groups[0]["lr"] = patch.value
            
            self.hyperparam_server.confirm_applied(version)
            print(f"[TRAINER] Applied hyperparameter batch v{version}")
            
    #**Main training loop with pause points
    async def train(self):
        print("---Starting Training---")
        while self.current_timesteps < self.total_timesteps and not self.should_quit:
            #Pause points
            await self._handle_control_events()
            while self.is_paused and not self.should_quit:
                await asyncio.sleep(0.1)
                await self._handle_control_events()
                
            await self._apply_staged_patches()
            
            #Data collection
            final_obs, collection_metrics = self.algorithm.collect_experiences(self.env, self.current_obs)
            self.current_obs = final_obs
            self.current_timesteps += self.algorithm.rollut_steps
            
            #Pause point after collection and before update
            await self._handle_control_events()
            if self.is_paused or self.should_quit: continue
            
            #policy update
            update_metrics = self.algorithm.update_policy()
            
            #Logging
            print(f"Timesteps: {self.current_timesteps}/{self.total_timesteps} | "
                  f"Mean Reward: {collection_metrics.get('mean_reward', 0):.2f} | "
                  f"Policy Loss: {update_metrics.get('policy_loss', 0):.3f}")
        print("--- Training Finished ---")
