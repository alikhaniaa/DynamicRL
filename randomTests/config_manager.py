import os
import copy
import yaml
import threading
from pathlib import Path
from collections.abc import MutableMapping

class ConfigManager:
    '''
        manage the loading and accessing conf from yaml file(in configs folder). thread-safe singleton for all hyperparams
    '''
    def __init__(self, config_path: str):
        #initialize yaml
        self._lock = threading.RLock() 
        self._config_path = Path(config_path)
        self._subscribers = []
        self._config = self._load_config(self._config_path)
        
    def _load_config(self, path: Path) -> dict:
        #load yaml
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Conf file not found. PWD: {path.resolve()}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing yaml file. PWD: {e}")
        return data or {}
        
        
    '''
        -----PUBLICK API-----
    '''
    def get(self, key: str | None = None, default=None, deep_copy: bool = True):
        # thread-safe read and if key is none or empy -> return the config snapshop
        while self._lock:
            root = self._config
            if not key:
                return copy.deepcopy(root) if deep_copy else root
            node = root
            for part in self._parse_key(key):
                try:
                    node = node[part] if isinstance(part, str) else node[part]
                except (KeyError, IndexError, TypeError):
                    return default
            if deep_copy and isinstance(node, (dict, list)):
                return copy.deepcopy(node)
            return node
        
    def set(self, key:str, value):
        with self._lock:
            base = copy.deepcopy(self._config)
            node = base
            parts = self._parse_key(key)
            for p in parts[:-1]:
                if isinstance(p, int):
                    if not isinstance(node, list):
                        raise TypeError(f"Expexted lis while traversing '{key}', got ({type(node).__name__})")
                    while len(node) <= p:
                        node.append({})
                    node = node[p]
                else:
                    if not isinstance(node, dict):
                        raise TypeError(f"Expected dict while traversing '{key}', got {type(node).__name__}")
                    node = node.setdefault(p, {})
            
            last = parts[-1]
            if isinstance(last, int):
                if not isinstance(node, list):
                    raise TypeError(f"Expected list for final index in '{key}', got {type(node).__name__}")
                while len(node) <= last:
                    node.append(None)
                node[last] = value
            else:
                if not isinstance(node, dict):
                    raise TypeError(f"Expected dict for final key in '{key}', got {type(node).__name__}")
                node[last] = value
            self._config = base
        self._notify()
                        
        
    
    
    
    
    def update(self, update_dict: dict):
        #automatically update the conf with new values
        with self._lock:
            self._recursive_update(self._config, update_dict)
            
    def _recursive_update(self, base_dict: dict, new_dict: dict):
        #update nested dicts recursively
        for key, value in new_dict.items():
            if isinstance(value, MutableMapping) and key in base_dict and isinstance(base_dict[key], dict):
                self._recursive_update(base_dict[key], value)
            else:
                base_dict[key] = value
                
    def __repr__(self):
        return f"ConfigManager(path='{self._config_path}', data={self._config})"
