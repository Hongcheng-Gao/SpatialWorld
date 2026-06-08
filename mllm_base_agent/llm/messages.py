"""Small message schema used by SpatialWorld providers and runners.

The classes intentionally mirror the tiny subset of legacy message objects that
this project used: a role, a content payload, and optional metadata on responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Union

MessageContent = Union[str, List[Dict[str, Any]]]


@dataclass
class BaseMessage:
    content: MessageContent
    role: str


@dataclass
class SystemMessage(BaseMessage):
    def __init__(self, content: MessageContent):
        super().__init__(content=content, role="system")


@dataclass
class UserMessage(BaseMessage):
    def __init__(self, content: MessageContent):
        super().__init__(content=content, role="user")


class HumanMessage(UserMessage):
    """Backward-compatible alias for the old message class name."""


@dataclass
class AssistantMessage(BaseMessage):
    def __init__(self, content: MessageContent):
        super().__init__(content=content, role="assistant")


class AIMessage(AssistantMessage):
    """Backward-compatible alias for the old message class name."""


@dataclass
class TextPart:
    text: str

    def to_payload(self) -> Dict[str, str]:
        return {"type": "text", "text": self.text}


@dataclass
class ImagePart:
    url: str

    def to_payload(self) -> Dict[str, Dict[str, str]]:
        return {"type": "image_url", "image_url": {"url": self.url}}


@dataclass
class ModelResponse:
    content: str
    response_metadata: Dict[str, Any] = field(default_factory=dict)
    usage_metadata: Dict[str, Any] = field(default_factory=dict)
    additional_kwargs: Dict[str, Any] = field(default_factory=dict)


def _normalize_content(content: Any) -> MessageContent:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        normalized: List[Dict[str, Any]] = []
        for item in content:
            if isinstance(item, TextPart):
                normalized.append(item.to_payload())
            elif isinstance(item, ImagePart):
                normalized.append(item.to_payload())
            elif isinstance(item, dict):
                normalized.append(item)
            else:
                normalized.append({"type": "text", "text": str(item)})
        return normalized
    return str(content)


def coerce_message(message: Any) -> BaseMessage:
    if isinstance(message, BaseMessage):
        return message
    role = getattr(message, "role", None)
    content = getattr(message, "content", None)
    if role in {"system", "user", "assistant"}:
        return BaseMessage(content=_normalize_content(content), role=role)
    name = type(message).__name__.lower()
    if "system" in name:
        return SystemMessage(_normalize_content(content))
    if "ai" in name or "assistant" in name:
        return AssistantMessage(_normalize_content(content))
    return UserMessage(_normalize_content(content))


def to_openai_messages(messages: Sequence[Any]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for raw in messages:
        msg = coerce_message(raw)
        payload.append({"role": msg.role, "content": _normalize_content(msg.content)})
    return payload


__all__ = [
    "AIMessage",
    "AssistantMessage",
    "BaseMessage",
    "HumanMessage",
    "ImagePart",
    "MessageContent",
    "ModelResponse",
    "SystemMessage",
    "TextPart",
    "UserMessage",
    "coerce_message",
    "to_openai_messages",
]
