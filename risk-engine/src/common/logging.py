"""Logging utilities for the risk engine pipeline."""

import logging
import sys
from typing import Optional


def get_logger(name: str = "risk", level: str = "INFO") -> logging.Logger:
    """Get a configured logger instance.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if no handlers exist (avoid duplicate handlers)
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
        # Set level
        logger.setLevel(getattr(logging, level.upper()))
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logger


def setup_file_logging(
    logger: logging.Logger, 
    log_file: str, 
    level: str = "DEBUG"
) -> None:
    """Add file handler to existing logger.
    
    Args:
        logger: Logger instance to configure
        log_file: Path to log file
        level: Logging level for file handler
    """
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    
    # Set level
    file_handler.setLevel(getattr(logging, level.upper()))
    
    # Add handler
    logger.addHandler(file_handler)


def log_dataframe_info(logger: logging.Logger, df, name: str = "DataFrame") -> None:
    """Log basic information about a DataFrame.
    
    Args:
        logger: Logger instance
        df: DataFrame to log info about
        name: Name to use in log message
    """
    logger.info(f"{name} shape: {df.shape}")
    logger.info(f"{name} columns: {list(df.columns)}")
    logger.info(f"{name} memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Log missing values if any
    missing = df.isnull().sum()
    if missing.any():
        logger.info(f"{name} missing values: {missing[missing > 0].to_dict()}")


def log_execution_time(logger: logging.Logger, start_time, end_time, task_name: str) -> None:
    """Log execution time for a task.
    
    Args:
        logger: Logger instance
        start_time: Start time (from time.time())
        end_time: End time (from time.time())
        task_name: Name of the task
    """
    duration = end_time - start_time
    logger.info(f"{task_name} completed in {duration:.2f} seconds")
