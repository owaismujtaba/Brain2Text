import os
import sys
import logging

def create_logger(name: str):
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    log_file = f"logs/{name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(name)
    return logger
