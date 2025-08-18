"""
    HyperparamServer module. gatekeeper of all live config changes.
    recieves ParamPatch events and validate them for rules, version and also prepare it for trainer.
"""
from __future__ import annotations
from typing import Optional
from typing import Any
import threading
from .events import ParamPatch

# type alias for a staged batch containing verion and patch
StagedBatch = tuple[int, list[ParamPatch]]

#Validate, version and stage hyperparams
class HyperparamServer:
    def __init__(self, initial_config: dict[str, Any]):
        self._lock = threading.RLock()
        self._config = initial_config
        self._staged_batch: Optional[StagedBatch] = None
        self._next_version = 1
        
        #validation rules
        self._FORBIDDEN_PATHS = {
            "algorithm.name",
            "environment.id",
        }
        self._NUMERIC_BOUNDS = {
            "ppo.learning_rate" : (0.0, None),
            "ppo.clip_range": (0.0, 1.0),
            "ppo.gamma": (0.8, 1.0),
            "ppo.gae_lambda": (0.8, 1.0),
            "ppo.entropy_coef": (0.0, None)
        }
        
    #Policy layer for hyperparam changes
    def _validate_patch(self, patch: ParamPatch) -> tuple[bool, str]:
        if patch.path in self._FORBIDDEN_PATHS:
            return False, f"Path '{patch.path}' is immutable and can't be changed at runtime."
        
        if patch.path in self._NUMERIC_BOUNDS:
            if not isinstance(patch.value, (int, float)):
                return False, f"Path '{patch.path}' expects a numeric value, not {type(patch.value).__name__}"
             
            low, high = self._NUMERIC_BOUNDS[patch.path]
            if low is not None and patch.value < low:
                return False, f"Value {patch.value} is below the minimum bound of {low} for '{patch.path}'"
            if high is not None and patch.value > high:
                return False, f"Value {patch.value} is above the maximum bound of {low} for '{patch.path}'"
        
        return True, "OK"
    
    #Validate and stages a batch of patches. entry point for producers
    def stage_patches(self, patches: list[ParamPatch]) -> tuple[bool, str]:
        with self._lock:
            if self._staged_batch is not None:
                return False, f"Can't stage new batch: version {self._staged_batch[0]} is already pending"
            
            for patch in patches:
                is_valid, reason = self._validate_patch(patch)
                if not is_valid:
                    return False, f"Validation failed for '{patch.path}' : {reason}"
                
            self._staged_batch = (self._next_version, patches)
            version = self._next_version
            self._next_version += 1
            return True, f"Batch v{version} staged successfully"
    
    #called by trainer to peek at pending batch at a safe point
    def get_staged_batch(self) -> Optional[StagedBatch]:
        with self._lock:
            return self._staged_batch
    
    #call the trained after applying patches anf finalize transaction by mutating config and clear
    def confirm_applied(self, version: int):
        with self._lock:
            if self._staged_batch is None or self._staged_batch[0] != version:
                return
            
            batch_to_apply = self._staged_batch[1]
            for patch in batch_to_apply:
                #navigate to nested dict and apply changes
                keys = patch.path.split('.')
                node = self._config
                for key in keys[:-1]:
                    node = node.setdefault(key, {})
                node[keys[-1]] = patch.value
                
            #clear the stage for next trans
            self._staged_batch = None

    #apply a single validate patch to the internal config dict
    def _apply_patch_to_config(self, patch: ParamPatch):
        keys = patch.path.split('.')
        node = self._config
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            
        final_key = keys[-1]
        
        if patch.op == "set":
            node[final_key] = patch.value
        elif patch.op == "add":
            node[final_key] = node.get(final_key, 0) + patch.value
        elif patch.op == "mul":
            node[final_key] = node.get(final_key, 1) * patch.value
