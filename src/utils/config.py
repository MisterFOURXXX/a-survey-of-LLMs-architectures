"""Configuration loading and merging."""
import os
from pathlib import Path

import yaml


def load_config(
    model_config_path: str,
    base_config_path: str | None = None,
) -> dict:
    """
    Load a model config and deep-merge it with the base config.

    Resolution order for `base_config.yaml`:
        1. Explicit `base_config_path` if provided.
        2. `$STACKSAMPLE_CONFIG_DIR/base_config.yaml`, if that env var is set.
        3. Same directory as the model config.
        4. `./configs/base_config.yaml` relative to CWD.
        5. `<model_config_parent>/../configs/base_config.yaml` (repo layout).
        6. Walking up the tree looking for `configs/base_config.yaml`.
    """
    model_path = Path(model_config_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model config not found at {model_path}")

    base_path = _resolve_base_config(model_path, base_config_path)

    if not base_path.exists():
        raise FileNotFoundError(
            f"Base config not found at {base_path}\n"
            f"Searched relative to model config: {model_path}"
        )

    base = _load_yaml(base_path)
    override = _load_yaml(model_path)
    return _deep_merge(base, override)


def _resolve_base_config(model_path: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()

    env_dir = os.environ.get("STACKSAMPLE_CONFIG_DIR")
    if env_dir:
        candidate = Path(env_dir) / "base_config.yaml"
        if candidate.exists():
            return candidate

    # Same dir as model config
    candidate = model_path.parent / "base_config.yaml"
    if candidate.exists():
        return candidate

    # CWD/configs
    candidate = Path.cwd() / "configs" / "base_config.yaml"
    if candidate.exists():
        return candidate

    # Sibling layout: <parent>/../configs/base_config.yaml
    candidate = model_path.parent.parent / "configs" / "base_config.yaml"
    if candidate.exists():
        return candidate

    # Walk up the tree
    for parent in model_path.parents:
        candidate = parent / "configs" / "base_config.yaml"
        if candidate.exists():
            return candidate
        candidate = parent / "base_config.yaml"
        if candidate.exists():
            return candidate

    # None found — return the sibling expectation so the error message is clear
    return model_path.parent / "base_config.yaml"


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