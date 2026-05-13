"""
Configuration management — reads/writes .ht-repro.toml user config.
"""
import os
from pathlib import Path
from typing import Any, Optional

CONFIG_FILE = Path.home() / ".ht-repro.toml"

DEFAULT_CONFIG = """# ht-repro configuration
# https://github.com/NagusameCS/HyperTensor

[general]
# Default tier to run (T1, T2, T3 — auto-detected if unset)
default_tier = ""

# Number of retries on failure
retries = 1

# Timeout multiplier (1.0 = default, 2.0 = double all timeouts)
timeout_multiplier = 1.0

[display]
# Show progress spinner during tests
spinner = true

# Show verbose output on failure
verbose_on_fail = true

# Color mode: auto, always, never
color = "auto"

[paths]
# HyperTensor repository root (auto-detected if empty)
repo_root = ""

# Custom benchmark output directory
benchmark_dir = ""
"""

def load_config(filename=None) -> dict:
    """Load config, creating default if needed."""
    path = Path(filename) if filename else CONFIG_FILE

    if not path.exists():
        path.write_text(DEFAULT_CONFIG)
        return _parse_toml(DEFAULT_CONFIG)

    try:
        return _parse_toml(path.read_text())
    except Exception:
        return _parse_toml(DEFAULT_CONFIG)

def _parse_toml(text: str) -> dict:
    """Minimal TOML parser (avoids external dependency)."""
    config = {}
    current_section = ""

    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
            config[current_section] = {}
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if current_section:
                config.setdefault(current_section, {})[key] = _coerce(val)
            else:
                config[key] = _coerce(val)
    return config

def _coerce(val: str) -> Any:
    if val.lower() == "true": return True
    if val.lower() == "false": return False
    if val == "": return ""
    try: return int(val)
    except ValueError: pass
    try: return float(val)
    except ValueError: pass
    return val

def get(key: str, default=None) -> Any:
    """Get a config value by dotted key (e.g., 'display.spinner')."""
    config = load_config()
    parts = key.split(".")
    for part in parts:
        if isinstance(config, dict):
            config = config.get(part)
        else:
            return default
    return config if config is not None else default

def save_config(config: dict, filename=None):
    """Save config back to file."""
    path = Path(filename) if filename else CONFIG_FILE
    lines = []
    for section, values in config.items():
        if section.startswith("_"):
            continue
        lines.append(f"\n[{section}]")
        if isinstance(values, dict):
            for k, v in values.items():
                lines.append(f"{k} = {_format_val(v)}")
    path.write_text("\n".join(lines).lstrip() + "\n")

def _format_val(v: Any) -> str:
    if isinstance(v, bool): return "true" if v else "false"
    if isinstance(v, str): return f'"{v}"' if v else '""'
    return str(v)
