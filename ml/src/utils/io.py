"""
I/O utilities for loading configs, saving JSONs, and managing paths.
"""

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict:
    """Load a YAML config file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    """
    Deep merge override config into base config.
    Override values take precedence.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(model_name: str, configs_dir: str | Path = None) -> dict:
    """
    Load merged config: base.yaml + {model_name}.yaml
    """
    if configs_dir is None:
        configs_dir = Path(__file__).resolve().parents[2] / 'configs'
    else:
        configs_dir = Path(configs_dir)

    base_config = load_yaml(configs_dir / 'base.yaml')
    model_config_path = configs_dir / f'{model_name}.yaml'

    if model_config_path.exists():
        model_config = load_yaml(model_config_path)
        return merge_configs(base_config, model_config)

    return base_config


def save_json(data: Any, path: str | Path) -> None:
    """Save data as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str | Path) -> Any:
    """Load a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def ensure_dirs(*paths: str | Path) -> None:
    """Create directories if they don't exist."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
