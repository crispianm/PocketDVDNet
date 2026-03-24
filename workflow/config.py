from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml

from .paths import REPO_ROOT


PATH_KEYS = {
    "trainset_dir",
    "valset_dir",
    "output_dir",
    "recipe_path",
    "checkpoint_path",
    "teacher_checkpoint",
    "pretrained_checkpoint",
    "noise_csv",
    "external_shift_net_dir",
    "metrics_path",
}
PATH_SUFFIXES = ("_dir", "_path", "_file")
UNRESOLVED_ENV_PATTERN = re.compile(r"\$\{[^}]+\}")


def _resolve_pathlike(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve())


def _resolve_value(key: str | None, value):
    if isinstance(value, dict):
        return {
            nested_key: _resolve_value(nested_key, nested_value)
            for nested_key, nested_value in value.items()
        }

    if isinstance(value, list):
        return [_resolve_value(key, item) for item in value]

    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if UNRESOLVED_ENV_PATTERN.search(expanded):
            raise ValueError(f"Unresolved environment variable in config value: {value}")

        if key and (key in PATH_KEYS or key.endswith(PATH_SUFFIXES)):
            return _resolve_pathlike(expanded)
        return expanded

    return value


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    resolved = _resolve_value(None, config)
    resolved["config_path"] = str(config_path.resolve())
    return resolved


def dump_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
