import multiprocessing as mp
from typing import Any, Callable
import gymnasium as gym
import numpy as np

#target func for each env subproc
def worker(pipe, env_fn):
    env = env_fn()
    while True:
        cmd, data = pipe.recv()
        if cmd == "step":
            obs, reward, terminated, truncated, info = env.step(data)
            done = terminated or truncated
            if done:
                obs, info = env.reset()
            pipe.send((obs, reward, done, info))
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
        self.pipes, worker_pipes = zip(*[mp.Pipe() for _ in range(self.num_envs)])
        self.procs = [
            mp.Process(target=worker, args=(p, fn), daemon=True)
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
        obs, rewards, dones, infos = zip(*results)
        
        return np.stack(obs), np.array(rewards), np.array(dones), list(infos)
    
    def reset(self, seed:int | None = None):
        for i, pipe in enumerate(self.pipes):
            pipe.send(("reset", seed + i if seed is not None else None))
            
        results = [pipe.recv() for pipe in self.pipes]
        obs, infos = zip(*results)
        return np.stack(obs), list(infos)
    
    def close(self):
        for pipe in self.pipes:
            pipe.send(("close", None))
        for p in self.procs:
            p.join()