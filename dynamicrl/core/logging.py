import logging
import sys

from rich.logging import RichHandler

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