"""
Logging Configuration Module

Centralized logging setup for the EvoNet package.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Set up logging for the EvoNet package.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional file path for logging
        format_string: Custom format string for log messages
    """
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s [%(name)s]: %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers
    )
    
    # Set level for evonet package
    logger = logging.getLogger('evonet')
    logger.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Name of the module (usually __name__)
        
    Returns:
        logging.Logger: Configured logger
    """
    return logging.getLogger(name)
