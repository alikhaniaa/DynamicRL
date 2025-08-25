import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from dynamicrl.core.logger import configure_logging
from dynamicrl.core.trainer import Trainer

logger = logging.getLogger(__name__)


# We go back to using the main config as the entry point
@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Runs the Trainer with a programmatically overridden configuration for testing.
    """
    configure_logging()
    
    logger.info("--- Overriding configuration for integration test ---")

    # --- THE KEY FIX ---
    # We temporarily unlock the configuration to allow for changes.
    OmegaConf.set_struct(cfg, False)

    # Now we can safely load and assign the cartpole config
    cartpole_config_path = Path(hydra.utils.get_original_cwd()).parent / "configs/env/cartpole.yaml"

    cfg.env = OmegaConf.load(cartpole_config_path)
    
    # And override the training parameters for a short run
    cfg.training.total_timesteps = cfg.env.num_envs * 1024
    cfg.training.rollout_steps = 512
    cfg.training.checkpoint_interval = 2048
    
    # Re-lock the configuration to keep it safe from accidental changes later
    OmegaConf.set_struct(cfg, True)

    output_dir = Path.cwd()
    logger.info(f"Test output will be saved to: {output_dir}")
    
    logger.info("--- Initializing and running the Trainer with test config ---")
    try:
        trainer = Trainer(cfg)
        trainer.train()
    except Exception as e:
        logger.error("Trainer failed during execution!", exc_info=True)
        raise e

    # --- Verification (remains the same) ---
    logger.info("--- Verifying test results ---")

    metrics_file = output_dir / "metrics.csv"
    assert metrics_file.exists(), "metrics.csv was not created!"
    with open(metrics_file, "r") as f:
        assert len(f.readlines()) > 1, "metrics.csv is empty!"
    logger.info("✅ Verification successful: metrics.csv created and contains data.")

    checkpoint_dir = output_dir / "checkpoints"
    checkpoints_found = list(checkpoint_dir.glob("checkpoint_*.pth"))
    assert len(checkpoints_found) > 0, "No checkpoint files were saved!"
    logger.info(f"✅ Verification successful: Found {len(checkpoints_found)} checkpoint(s).")

    logger.info("🎉 Integration test passed successfully! 🎉")


if __name__ == "__main__":
    main()