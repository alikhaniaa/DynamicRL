import logging
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf

# We import the services and the log configuration function we want to test
from dynamicrl.core.checkpoint import CheckpointManager
from dynamicrl.core.logger import configure_logging, DataLogger 


# Get a logger for this test script
logger = logging.getLogger(__name__)


def run_tests():
    """
    Runs a series of simple tests to verify the CheckpointManager and DataLogger.
    """
    # --- Setup ---
    # 1. Configure our beautiful rich-based logging for the console.
    # This should be the first thing called in any application entry point.
    configure_logging(level="DEBUG")

    # 2. Create a mock configuration object, similar to what Hydra would provide.
    # We use OmegaConf to create a DictConfig object.
    mock_cfg = OmegaConf.create(
        {
            "output_dir": "_test_outputs",
            "logging": {
                "enable_console": True,
                "enable_csv": True,
            },
        }
    )

    # Clean up any previous test runs
    test_output_dir = Path(mock_cfg.output_dir)
    if test_output_dir.exists():
        logger.warning(f"Removing previous test directory: {test_output_dir}")
        shutil.rmtree(test_output_dir)

    logger.info("--- Starting CheckpointManager Test ---")
    test_checkpoint_manager(mock_cfg)

    logger.info("\n--- Starting DataLogger Test ---")
    test_data_logger(mock_cfg)

    logger.info("\n--- Tests Finished ---")
    logger.info(f"Check the generated files in the '{test_output_dir.resolve()}' directory.")


def test_checkpoint_manager(cfg):
    """Tests saving, loading, and retention policy of the CheckpointManager."""
    # Define a simple dummy model and optimizer for the test
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Instantiate the manager
    checkpoint_dir = Path(cfg.output_dir) / "checkpoints"
    # Test retention: only keep the 2 most recent checkpoints
    manager = CheckpointManager(save_dir=checkpoint_dir, max_to_keep=2)

    # --- Test 1: Save and Load a checkpoint ---
    logger.info("Test 1: Saving and loading a checkpoint...")
    original_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": 100,
    }
    manager.save(original_state, tag="step_100")

    # Load the state back
    device = torch.device("cpu")
    loaded_state = manager.load(tag="step_100", device=device)

    # Verification
    assert loaded_state is not None, "Failed to load checkpoint."
    assert loaded_state["global_step"] == 100, "Global step mismatch."
    # Compare model weights tensor by tensor
    for key, val in original_state["model_state_dict"].items():
        assert torch.equal(val, loaded_state["model_state_dict"][key]), f"Model tensor mismatch for key: {key}"
    logger.info("Save and load successful. State is identical.")

    # --- Test 2: Test retention policy ---
    logger.info("Test 2: Verifying retention policy (max_to_keep=2)...")
    manager.save({"global_step": 200}, tag="step_200")
    manager.save({"global_step": 300}, tag="step_300") # This should trigger deletion of step_100

    assert not (checkpoint_dir / "checkpoint_step_100.pth").exists(), "Oldest checkpoint was not deleted."
    assert (checkpoint_dir / "checkpoint_step_200.pth").exists(), "Checkpoint 200 should exist."
    assert (checkpoint_dir / "checkpoint_step_300.pth").exists(), "Checkpoint 300 should exist."
    logger.info("Retention policy works as expected.")


def test_data_logger(cfg):
    """Tests logging to multiple backends (Console and CSV)."""
    data_logger = DataLogger(cfg)

    # Simulate a few steps of a training loop
    for i in range(5):
        step = (i + 1) * 100
        # Create a mock metrics dictionary
        metrics = {
            "mean_reward": 20.5 + i * 5,
            "fps": 1500 + i * 10,
            "policy_loss": -0.05 - i * 0.01,
            "value_loss": 0.1 - i * 0.005,
            "entropy": 1.2 - i * 0.02,
            "some_non_scalar": [1, 2, 3],  # This should be ignored by the CSV logger
        }
        data_logger.log(metrics, step=step)

    data_logger.close()


if __name__ == "__main__":
    run_tests()