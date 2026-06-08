"""Compatibility configuration loader for historical scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from actions.max_steps import resolve_max_steps_from_task
from mllm_base_agent.config import Config, load_config as _load_generic_config
from mllm_base_agent.config import print_config


def load_config(config_path: str | Path | None = None, **overrides: Any) -> Config:
    if config_path is None:
        config_path = Path("experiments/configs/ai2thor/config_close_gpt-5.yaml")
    config_path = Path(config_path)

    loader = None
    parts = set(config_path.parts)
    if "ai2thor" in parts:
        from configs.ai2thor.load_config import load_config as loader
    elif "procthor" in parts:
        from configs.procthor.load_config import load_config as loader
    elif "carla" in parts:
        from configs.carla.load_config import load_config as loader
    elif "virtualhome" in parts:
        from configs.virtualhome.load_config import load_config as loader

    if loader is not None:
        return loader(str(config_path), **overrides)

    config = _load_generic_config(config_path)
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


__all__ = ["Config", "load_config", "print_config", "resolve_max_steps_from_task"]
