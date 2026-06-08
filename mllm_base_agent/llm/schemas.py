"""Core action and observation schemas shared across environments."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ThorAction(BaseModel):
    name: str
    args: Dict[str, Any] = Field(default_factory=dict)


class EnvAction(BaseModel):
    move: Optional[Literal["forward", "back", "left", "right"]] = None
    turn: Optional[float] = None
    interact: bool = False
    comment: Optional[str] = None
    thor_actions: Optional[List[ThorAction]] = None
    low_level_code: Optional[str] = None


class EnvObservation(BaseModel):
    image_path: str
    text_state: str
    reward: float = 0.0
    done: bool = False
    metadata: Optional[Dict[str, Any]] = None
