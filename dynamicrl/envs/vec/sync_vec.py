import multiprocessing as mp
from typing import Any, Callable
import gymnasium as gym
import numpy as np

#target func for each env subproc
def worker(pipe, env_fn):
    env = env_fn()
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete) # Check action space type once

    while True:
        cmd, data = pipe.recv()
        if cmd == "step":
            action_to_step = data.item() if is_discrete else data
            obs, reward, terminated, truncated, info = env.step(action_to_step)
            if terminated or truncated:
                info["final_observation"] = obs
                info["final_info"] = info
                obs, _ = env.reset()
            pipe.send((obs, reward, terminated, truncated, info))
        elif cmd == "reset":
            obs, info = env.reset(seed=data)
            pipe.send((obs, info))
        elif cmd == "close":
            env.close()
            pipe.close()
            break
        else:
            raise NotImplementedError(f"Unknown command: {cmd}")
        
""" simple sync, subproc-based vectorized env """
class SyncVecEnv:
    def __init__(self, env_fns:list[Callable[[], gym.Env]]):
        self.num_envs = len(env_fns)
        ctx = mp.get_context("fork")
        self.pipes, worker_pipes = zip(*[ctx.Pipe() for _ in range(self.num_envs)])
        self.procs = [
            ctx.Process(target=worker, args=(p, fn), daemon=True)
            for p, fn in zip(worker_pipes, env_fns)
        ]
        for p in self.procs:
            p.start()
            
        self.observation_space = self._get_space(env_fns[0], "observation_space")
        self.action_space = self._get_space(env_fns[0], "action_space")
        
    #create a temp env to inspect it's spaces
    def _get_space(self, env_fn, space_name):
        temp_env = env_fn()
        space = getattr(temp_env, space_name)
        temp_env.close()
        return space
    
    def step(self, actions: np.ndarray):
        for pipe, action in zip(self.pipes, actions):
            pipe.send(("step", action))
            
        results = [pipe.recv() for pipe in self.pipes]
        obs, rewards, terminated, truncated, infos = zip(*results)
        
        return (np.stack(obs), np.array(rewards),np.array(terminated),np.array(truncated),list(infos),)
    
    def reset(self, seed:int | None = None):
        for i, pipe in enumerate(self.pipes):
            pipe.send(("reset", seed + i if seed is not None else None))
            
        results = [pipe.recv() for pipe in self.pipes]
        obs, infos = zip(*results)
        return np.stack(obs), list(infos)
    
    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except BrokenPipeError:
                pass 
        for p in self.procs:
            p.join()
