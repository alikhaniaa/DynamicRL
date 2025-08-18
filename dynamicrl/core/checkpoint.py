import logging
import os
import shutil
from pathlib import Path
from typing import Any
import torch

logger = logging.getLogger(__name__)

"""
    Manage saving and loading of the complete training state.
    handle serialization, optimizer and other metadata
"""
class CheckpointManager:
    def __init__(self, save_dir: str | Path, max_to_keep: int = 5):
        self.save_dir = Path(save_dir)
        self.max_to_keep = max_to_keep
        self.checkpoints = []
        
        self.save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint directory set to: {self.save_dir.resolve()}")
        self._scan_existing_checkpoints()
        
    def _scan_existing_checkpoints(self) -> None:
        try:
            self.checkpoints = sorted(
                self.save_dir.glob("checkpoint_*.pth"),
                key=os.path.getmtime,
                reverse=True,
            )
            logger.info(f"Found {len(self.checkpoints)} existing checkpoints")
        except Exception as e:
            logger.warning(f"Could not scan existing checkpoints: {e}")
        
    #save trainer state    
    def save(self, trainer_state: dict[str, Any], tag: str) -> None:
        filename = f"checkpoint_{tag}.pth"
        filepath = self.save_dir / filename
        temp_filepath = filepath.with_suffix(".tmp")
        
        #write temporary to not upload corrupted checkpoint
        try:
            torch.save(trainer_state, temp_filepath)
            shutil.move(temp_filepath, filepath)
            
            self.checkpoints.append(filepath)
            logger.info(f"Successfully saved checkpoint to {filepath}")
            
            self._enforce_retention_policy()
        except Exception as e:
            logger.error(f"Failed to save checkpoint {tag}: {e}", exc_info=True)
            
    def load(self, tag: str, device: torch.device) -> dict[str, Any] | None:
        filename = f"checkpoint_{tag}.pth"
        filepath = self.save_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Checkpoint file not found: {filepath}")
            return None
        
        try:
            state_dict = torch.load(filepath, map_location=device)
            logger.info(f"Successfully loaded checkpoit from {filepath}")
            return state_dict
        except Exception as e:
            logger.error(f"Failed to load checkpoint {tag}: {e}", exc_info=True)
            return None
        
    #delete older checkpoint if exceeds max
    def _enforce_retention_policy(self) -> None:
        if self.max_to_keep is not None and self.max_to_keep > 0:
            self.checkpoints.sort(key=os.path.getmtime, reverse=True)
            
            while len(self.checkpoints) > self.max_to_keep:
                olderst_checkpoint = self.checkpoints.pop(-1)
                try:
                    olderst_checkpoint.unlink()
                    logger.info(f"Removed old checkpoint: {olderst_checkpoint}")
                except OSError as e:
                    logger.warning(f"Error removing old checkpoint {olderst_checkpoint}: {e}")