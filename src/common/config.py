"""Config loading.

One YAML file is the single source of truth for every stage of the pipeline, so a
training run, a batch scoring job and a drift check can never disagree about the
horizon, the lag set or the feature list.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("conf/config.yaml")

# ${VAR} and ${VAR:-default}. os.path.expandvars understands neither the default
# form nor Windows semantics consistently, so the substitution is explicit.
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class Config(dict):
    """dict with attribute access, so `cfg.spark.shuffle_partitions` reads cleanly."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:  # pragma: no cover - programmer error
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _expand_str(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, default = match.group(1), match.group(2)
        # `:-` semantics: an empty value falls back to the default, like the shell.
        # A reference with no default and no value is left alone, so a missing
        # variable is visible in the config rather than silently becoming "".
        current = os.environ.get(name)
        if current:
            return current
        return default if default is not None else match.group(0)

    return _ENV_REF.sub(replace, value)


def _expand_env(value: Any) -> Any:
    """Allow ${VAR} and ${VAR:-default} inside the YAML."""
    if isinstance(value, str):
        return _expand_str(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config(_expand_env(raw))


def resolve(cfg: Config, dotted: str, default: Any = None) -> Any:
    """Look up `a.b.c` without a chain of .get() calls."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
