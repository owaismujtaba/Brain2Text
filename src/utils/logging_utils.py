"""Simple per-module logging to both a file and stdout."""
import logging
import os
import sys

_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to ``logs/<name>.log`` and stdout.

    Each name gets its own log file, and handlers are attached only once so
    repeated calls are safe.
    """
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(_FORMAT)
        for handler in (logging.FileHandler(f"logs/{name}.log"),
                        logging.StreamHandler(sys.stdout)):
            handler.setFormatter(formatter)
            logger.addHandler(handler)

    return logger
