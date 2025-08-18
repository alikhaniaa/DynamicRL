import hydra
from omegaconf import DictConfig, OmegaConf

from dynamicrl.core.logging import configure_logging
from dynamicrl.core.trainer import Trainer

"""
    Main entry point for training
"""
@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    configure_logging()
    
    trainer = Trainer(cfg)
    trainer.train()
    
if __name__ == "__main__":
    main()