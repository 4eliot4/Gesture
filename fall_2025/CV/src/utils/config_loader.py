"""
Configuration loader utility for loading YAML configs.
"""
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """Configuration object with attribute-style access."""

    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary."""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting."""
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default fallback, like dict.get()."""
        return getattr(self, key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config back to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        """String representation."""
        return f"Config({self.to_dict()})"


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object with attribute-style access

    Example:
        config = load_config('config.yaml')
        print(config.data.data_root)
        print(config.training.learning_rate)
        print(config.wandb.project)
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def save_config(config: Config, output_path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Config object
        output_path: Path to output YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)
