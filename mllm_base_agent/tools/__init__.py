"""Tool helpers without external chain frameworks."""

from functools import wraps
from typing import Any, Callable, Optional


def tool(func: Optional[Callable[..., Any]] = None, *decorator_args: Any, **decorator_kwargs: Any):
    """No-op replacement for the previous external tool decorator.

    It preserves the wrapped function and attaches small metadata fields used by
    simple tool registries.
    """
    def decorate(inner: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(inner)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return inner(*args, **kwargs)
        wrapper.name = decorator_kwargs.get('name') or getattr(inner, '__name__', 'tool')
        wrapper.description = decorator_kwargs.get('description') or getattr(inner, '__doc__', '')
        wrapper.is_spatialworld_tool = True
        return wrapper
    if callable(func):
        return decorate(func)
    return decorate

__all__ = ['tool']
