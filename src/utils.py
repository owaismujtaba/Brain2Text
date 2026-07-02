import os
import sys
import logging

import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Load a YAML configuration file into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def create_logger(name: str):
    """Return a logger that writes to a log file and stdout.

    Uses a per-name logger with its own handlers instead of logging.basicConfig,
    which only configures the root logger once — with basicConfig every module
    after the first silently shared the first module's log file.

    The log file location is controlled by environment variables so a single run
    (e.g. one ablation) can direct all its logs to a chosen path:
      * B2T_LOG_FILE -- exact log-file path (all modules of the run share it)
      * B2T_LOG_DIR  -- directory for per-module <name>.log files (default "logs")
    """
    log_file = os.environ.get("B2T_LOG_FILE")
    if log_file:
        log_dir = os.path.dirname(log_file) or "."
    else:
        log_dir = os.environ.get("B2T_LOG_DIR", "logs")
        log_file = os.path.join(log_dir, f"{name}.log")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # don't double-log through the root logger

    # Avoid attaching duplicate handlers if create_logger is called again.
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger
