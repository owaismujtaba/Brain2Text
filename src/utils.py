import os
import sys
import logging

import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Load a YAML configuration file into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def create_logger(name: str):
    """Return a logger that writes to logs/<name>.log and stdout.

    Uses a per-name logger with its own handlers instead of logging.basicConfig,
    which only configures the root logger once — with basicConfig every module
    after the first silently shared the first module's log file.
    """
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # don't double-log through the root logger

    # Avoid attaching duplicate handlers if create_logger is called again.
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(f"logs/{name}.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger
