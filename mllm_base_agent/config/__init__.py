"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class Config(dict):
    def get_all(self) -> Dict[str, Any]:
        return dict(self)


def load_config(path: str | Path) -> Config:
    with open(path, 'r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    return Config(data)


def print_config(config: Dict[str, Any]) -> None:
    data = config.get_all() if hasattr(config, 'get_all') else dict(config)
    print(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))

__all__ = ['Config', 'load_config', 'print_config']
