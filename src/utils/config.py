"""Configuration loading and merging."""
from pathlib import Path

import yaml


def load_config(
    model_config_path: str,
    base_config_path: str | None = None,
) -> dict:
    """
    Load a model config and deep-merge it with the base config.

    If `base_config_path` is None, it is resolved to `base_config.yaml`
    sitting next to the model config file.
    """
    model_path = Path(model_config_path).resolve()

    if base_config_path is None:
        base_path = model_path.parent / "base_config.yaml"
    else:
        base_path = Path(base_config_path).resolve()

    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found at {base_path}")

    base = _load_yaml(base_path)
    override = _load_yaml(model_path)
    return _deep_merge(base, override)


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into `base` (returns a new dict)."""
    result = {**base}
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result