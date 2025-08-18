import logging
import sys
import csv
from abc import ABC, abstractmethod
from rich.logging import RichHandler
from pathlib import Path
from typing import Any
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
DEFAULT_EXCLUSIONS = ["parso", "h5py"]

"""
    configure the root logger of the entire framework using RichHandler for good looks
"""
def configure_logging(
    level: str = "INFO",
    exclusions: list[str] | None = None
) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    #Base conf
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                tracebacks_suppress=[],
                show_path=False
            )
        ]
    )
    if exclusions is None:
        exclusions = DEFAULT_EXCLUSIONS
    
    for logger_name in exclusions:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
        
        
class LogBackend(ABC):
    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        pass
    
    def close(self) -> None:
        pass
    
#formatted summary of metrics
class ConsoleBackend(LogBackend):
    def __init__(self):
        self.key_metrics = [
            "mean_reward",
            "fps",
            "policy_loss",
            "value_loss",
            "entropy",
        ]
        
    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        log_str = f"Step: {step:<8}"
        for key in self.key_metrics:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, float):
                    log_str += f" | {key.replace('_', ' ').title()}: {value:<8.2f}"
                else:
                    log_str += f" | {key.replace('_', ' ').title()}: {value}"
                    
        logger.info(log_str)
        
#scalar metrics logger
class CSVBackend(LogBackend):
    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "metrics.csv"
        self.csv_file = None
        self.csv_writer = None
        self.headers_written = False
        
    def log_metrics(self, metrics: dict[str, Any], step: int) -> None:
        scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        if not scalar_metrics:
            return
        
        scalar_metrics["step"] = step
        
        try:
            if self.csv_writer is None:
                self.csv_file = open(self.log_file, "w", newline="")
                fieldnames = sorted(scalar_metrics.keys())
                self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
                
            if not self.headers_written:
                self.csv_writer.writeheader()
                self.headers_written = True
                
            row = {field: scalar_metrics.get(field) for field in self.csv_writer.fieldnames}
            self.csv_writer.writerow(row)
            self.csv_file.flush()

        except Exception as e:
            logger.error(f"Error writing to CSV file: {e}", exc_info=True)
            
    def close(self) -> None:
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_writer = None
            logger.info(f"CSV log saved to: {self.log_file.resolve()}")
            
class DataLogger:
    def __init__(self, cfg: DictConfig):
        self.backends: list[LogBackend] = []
        
        output_dir = cfg.get("output_dir", "outputs/latest_run")

        if cfg.logging.get("enable_console", True):
            self.backends.append(ConsoleBackend())
        if cfg.logging.get("enable_csv", True):
            self.backends.append(CSVBackend(log_dir=output_dir))
        # TODO: Add a TensorBoardBackend here if cfg.logging.enable_tensorboard is True
        
        logger.info(f"DataLogger initialized with backends: {[b.__class__.__name__ for b in self.backends]}")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        for backend in self.backends:
            try:
                backend.log_metrics(metrics, step)
            except Exception as e:
                logger.error(f"Logging backend {backend.__class__.__name__} failed: {e}", exc_info=True)

    def close(self) -> None:
        logger.info("Closing all data logger backends.")
        for backend in self.backends:
            backend.close()            