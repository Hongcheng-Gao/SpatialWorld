"""Environment registry for SpatialWorld."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY: Dict[str, Callable[[Optional[Dict[str, Any]]], Any]] = {}


def register_env(name: str, factory: Callable[[Optional[Dict[str, Any]]], Any]) -> None:
    _REGISTRY[name.lower()] = factory


class LegacyImportContext:
    def __init__(self, project_dir: str):
        self.project_path = str(_REPO_ROOT / project_dir)

    def __enter__(self):
        if self.project_path not in sys.path:
            sys.path.insert(0, self.project_path)
        # Legacy projects all use top-level names core/envs/evaluators/config.
        # Clearing them before import prevents one environment's modules from
        # accidentally satisfying another environment's absolute imports.
        for name in list(sys.modules):
            if name == 'core' or name.startswith('core.') or name == 'envs' or name.startswith('envs.'):
                sys.modules.pop(name, None)
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _legacy_factory(project_dir: str, module_name: str, class_name: str):
    def factory(config: Optional[Dict[str, Any]] = None):
        with LegacyImportContext(project_dir):
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            return cls(config=config)
    return factory


@dataclass
class EmbodiedCityAdapter:
    config: Optional[Dict[str, Any]] = None

    def create_client(self):
        cfg = self.config or {}
        sys.path.insert(0, str(_REPO_ROOT / 'embodiedcity'))
        from embodiedcity.client import DroneClient
        return DroneClient(
            base_url=cfg.get('base_url', ''),
            drone_id=cfg.get('drone_id', ''),
            token=cfg.get('token', ''),
        )


@dataclass
class GameAdapter:
    config: Optional[Dict[str, Any]] = None

    def get_evaluation_engine_class(self):
        sys.path.insert(0, str(_REPO_ROOT / 'game'))
        from core.evaluation_engine import EvaluationEngine
        return EvaluationEngine


def _embodiedcity_factory(config: Optional[Dict[str, Any]] = None):
    return EmbodiedCityAdapter(config=config)


def _game_factory(config: Optional[Dict[str, Any]] = None):
    return GameAdapter(config=config)


register_env('ai2thor', _legacy_factory('ai2thor', 'envs.ai2thor.wrapper', 'AI2ThorEnvWrapper'))
register_env('carla', _legacy_factory('carla', 'envs.carla.wrapper', 'CarlaEnvWrapper'))
register_env('carla_walker', _legacy_factory('carla', 'envs.carla.wrapper', 'WalkerEnvWrapper'))
register_env('procthor', _legacy_factory('procthor', 'envs.procthor_wrapper', 'ProcTHOREnvWrapper'))
register_env('virtualhome', _legacy_factory('virtualhome', 'envs.virtualhome.wrapper', 'VirtualHomeEnvWrapper'))
register_env('embodiedcity', _embodiedcity_factory)
register_env('game', _game_factory)


def create_env(env_type: str, config: Optional[Dict[str, Any]] = None):
    key = (env_type or '').lower()
    if key == 'carla' and (config or {}).get('executor_type') == 'walker':
        key = 'carla_walker'
    if key not in _REGISTRY:
        raise ValueError(f'Unknown environment type: {env_type}. Registered: {sorted(_REGISTRY)}')
    return _REGISTRY[key](config)


__all__ = ['create_env', 'register_env', 'EmbodiedCityAdapter', 'GameAdapter']
