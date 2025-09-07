"""I/O utilities for the risk engine pipeline."""

import os
import pandas as pd
from pathlib import Path
from typing import Optional, List, Union


def ensure_dir(path: Union[str, Path]) -> None:
    """Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path to create
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def read_csv(path: Union[str, Path], parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    """Read CSV file with optional date parsing.
    
    Args:
        path: Path to CSV file
        parse_dates: List of columns to parse as dates
        
    Returns:
        DataFrame with loaded data
    """
    return pd.read_csv(path, parse_dates=parse_dates)


def write_parquet(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Write DataFrame to Parquet format.
    
    Args:
        df: DataFrame to save
        path: Output path for Parquet file
    """
    ensure_dir(os.path.dirname(path))
    df.to_parquet(path, index=False)


def read_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """Read Parquet file.
    
    Args:
        path: Path to Parquet file
        
    Returns:
        DataFrame with loaded data
    """
    return pd.read_parquet(path)


def write_csv(df: pd.DataFrame, path: Union[str, Path], index: bool = False) -> None:
    """Write DataFrame to CSV format.
    
    Args:
        df: DataFrame to save
        path: Output path for CSV file
        index: Whether to include index in output
    """
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=index)


def file_exists(path: Union[str, Path]) -> bool:
    """Check if file exists.
    
    Args:
        path: Path to check
        
    Returns:
        True if file exists, False otherwise
    """
    return Path(path).exists()


def get_file_size(path: Union[str, Path]) -> int:
    """Get file size in bytes.
    
    Args:
        path: Path to file
        
    Returns:
        File size in bytes
    """
    return Path(path).stat().st_size
