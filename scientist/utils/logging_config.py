"""
Logging configuration for the reproducibility agent.
"""

import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
import json


def setup_logging(log_level: str = None) -> logging.Logger:
    """
    Setup logging configuration for the entire application.
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    
    # Get log level from environment or parameter
    log_level = log_level or os.getenv("LOG_LEVEL", "WARNING")
    log_dir = os.getenv("LOG_DIR", "./data/outputs/logs")
    
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("reproducibility_agent")
    logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Console Handler - colorful output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File Handler - rotating file handler
    log_file = os.path.join(log_dir, "reproducibility_agent.log")
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("reproducibility_agent")
