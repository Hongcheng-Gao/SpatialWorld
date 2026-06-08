"""
Dual Agent Configuration Module
Loads and manages configuration for the dual-agent system
"""

from importlib import util
from pathlib import Path


_CONFIG_PATH = Path(__file__).resolve().parents[4] / "configs" / "ai2thor" / "load_config.py"
_SPEC = util.spec_from_file_location("_ai2thor_config_load_config", _CONFIG_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load AI2-THOR config module from {_CONFIG_PATH}")

_MODULE = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

ConfigLoader = _MODULE.ConfigLoader
load_config = _MODULE.load_config
print_config = _MODULE.print_config

__all__ = ["ConfigLoader", "load_config", "print_config"]
