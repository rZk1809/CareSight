"""Configuration utilities for the risk engine pipeline."""

import yaml
from pathlib import Path
from typing import Dict, Any, Union


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_all_configs(config_dir: Union[str, Path] = "configs") -> Dict[str, Dict[str, Any]]:
    """Load all configuration files from a directory.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        Dictionary with config filename (without extension) as key and config as value
    """
    config_dir = Path(config_dir)
    configs = {}
    
    for config_file in config_dir.glob("*.yaml"):
        config_name = config_file.stem
        configs[config_name] = load_config(config_file)
    
    return configs


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to value (e.g., "model.params.learning_rate")
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration dictionary
    """
    merged = {}
    
    for config in configs:
        merged.update(config)
    
    return merged


def validate_config_paths(config: Dict[str, Any], base_path: Union[str, Path] = ".") -> bool:
    """Validate that paths in configuration exist.
    
    Args:
        config: Configuration dictionary
        base_path: Base path for relative paths
        
    Returns:
        True if all paths exist, False otherwise
    """
    base_path = Path(base_path)
    
    # Common path keys to check
    path_keys = ['input_dir', 'interim_dir', 'processed_dir', 'models_dir', 'reports_dir']
    
    for key in path_keys:
        if key in config:
            path = base_path / config[key]
            if not path.exists():
                return False
    
    return True
