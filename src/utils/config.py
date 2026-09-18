"""Configuration loading, merging, and numeric-type sanitization."""
import re
from pathlib import Path

import yaml


# Matches: 1e-4, 2E-5, 3.14, .5, -0.1, 100
_FLOAT_RE = re.compile(
    r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$"
)
_INT_RE = re.compile(r"^[+-]?\d+$")


def load_config(
    model_config_path: str,
    base_config_path: str | None = None,
) -> dict:
    """
    Load a model config and deep-merge it with the base config.

    After merging, numeric-looking strings (e.g. '1e-4', '0.05', '100') are
    coerced to their proper Python types. This is required because PyYAML's
    YAML 1.1 rules parse `1e-4` (no decimal point) as a *string*, not a float.
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
    merged = _deep_merge(base, override)
    return _sanitize_numerics(merged)


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


def _sanitize_numerics(obj):
    """
    Recursively convert numeric-looking strings to int or float.

    Handles the PyYAML quirk where values like `1e-4` are parsed as strings
    instead of floats. Ints are tried first, then floats.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_numerics(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_numerics(v) for v in obj]
    if isinstance(obj, str):
        stripped = obj.strip()
        if _INT_RE.match(stripped):
            try:
                return int(stripped)
            except ValueError:
                pass
        if _FLOAT_RE.match(stripped):
            try:
                return float(stripped)
            except ValueError:
                pass
        return obj
    return obj