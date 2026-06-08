"""Dual-agent package for AI2-THOR compatibility scripts.

Keep package initialization lightweight: most modules are imported by direct
submodule path, and eager re-exports can break script startup when legacy names
move into the unified ``mllm_base_agent`` package.
"""

__version__ = "2.0.0"
__author__ = "Spatial Planning Team"

__all__ = ["__version__", "__author__"]

