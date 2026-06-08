"""Compatibility wrapper for the unified mllm_base_agent package."""

from __future__ import annotations

from typing import Any, Mapping

from mllm_base_agent.llm.provider import create_vlm, get_vlm as _get_vlm


def _agent_config_kwargs(agent_config: Mapping[str, Any] | None) -> dict[str, Any]:
    if not agent_config:
        return {}

    return {
        "provider": agent_config.get("provider", "openai"),
        "model_name": agent_config.get("model_name") or agent_config.get("model"),
        "temperature": agent_config.get("temperature", 0.7),
        "max_tokens": agent_config.get("max_tokens", 2000),
        "top_p": agent_config.get("top_p"),
        "base_url": agent_config.get("base_url") or agent_config.get("api_base"),
        "api_key": agent_config.get("api_key"),
    }


def get_vlm(*, agent_config: Mapping[str, Any] | None = None, **kwargs: Any):
    """Build a VLM from either legacy agent_config or provider kwargs."""
    if agent_config is not None:
        merged = _agent_config_kwargs(agent_config)
        merged.update({key: value for key, value in kwargs.items() if value is not None})
        return _get_vlm(**merged)
    return _get_vlm(**kwargs)


def use_separate_agent_models(config: Mapping[str, Any]) -> bool:
    return bool(config.get("agent_1") or config.get("agent_2"))


def get_dual_agent_vlms(config: Mapping[str, Any]):
    base_model = config.get("model", {}) if isinstance(config.get("model"), Mapping) else {}
    agent_1 = dict(base_model)
    agent_1.update(config.get("agent_1") or {})
    agent_2 = dict(base_model)
    agent_2.update(config.get("agent_2") or {})
    return get_vlm(agent_config=agent_1), get_vlm(agent_config=agent_2)


__all__ = [
    "create_vlm",
    "get_vlm",
    "get_dual_agent_vlms",
    "use_separate_agent_models",
]
