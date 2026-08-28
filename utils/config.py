from pathlib import Path

import yaml


def _merge_dict(base, override):
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path, _stack=None):
    """Load YAML config with optional relative ``_base_`` inheritance."""
    config_path = Path(path).expanduser().resolve()
    stack = tuple(_stack or ())
    if config_path in stack:
        chain = " -> ".join(str(item) for item in (*stack, config_path))
        raise ValueError(f"Circular config inheritance detected: {chain}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")

    base_ref = config.pop("_base_", None)
    if base_ref is None:
        return config
    if not isinstance(base_ref, str) or not base_ref.strip():
        raise ValueError(f"_base_ must be a non-empty path string: {config_path}")

    base_path = Path(base_ref)
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    base = load_config(base_path, _stack=(*stack, config_path))
    return _merge_dict(base, config)
