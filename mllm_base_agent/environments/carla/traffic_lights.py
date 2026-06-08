"""
CARLA traffic light helpers.

This controller keeps traffic lights deterministic for discrete-action tasks:
- All traffic lights are frozen.
- The whole map shares a fixed cycle: Red -> Green -> Red -> ...
- Initial state is always Red for reproducibility.
- Vehicle actions advance the cycle one step at a time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:
    import carla
except ImportError:
    carla = None


class TrafficLightController:
    """Deterministic whole-map traffic light controller."""

    DEFAULT_CYCLE = ["Red", "Green"]

    def __init__(self, world: Any, *, carla_module: Any = None):
        self.world = world
        self.carla = carla_module if carla_module is not None else carla
        self.traffic_lights: List[Any] = []
        self.cycle: List[str] = list(self.DEFAULT_CYCLE)
        self.current_index = 0
        self.state_history: List[str] = []
        self.enabled = False

    @property
    def initial_state(self) -> str:
        return self.cycle[0]

    @property
    def current_state(self) -> Optional[str]:
        if not self.enabled or not self.state_history:
            return None
        return self.state_history[-1]

    @property
    def is_red(self) -> bool:
        return self.current_state == "Red"

    @property
    def is_green(self) -> bool:
        return self.current_state == "Green"

    def initialize(self) -> bool:
        """Collect all world traffic lights and force the initial state (Red)."""
        if self.world is None:
            self.enabled = False
            return False

        actors = self.world.get_actors()
        if hasattr(actors, "filter"):
            self.traffic_lights = list(actors.filter("traffic.traffic_light*"))
        else:
            self.traffic_lights = [
                actor
                for actor in actors
                if getattr(actor, "type_id", "").startswith("traffic.traffic_light")
            ]

        if not self.traffic_lights:
            self.enabled = False
            self.state_history = []
            return False

        self.enabled = True
        self.current_index = 0
        self.state_history = [self.initial_state]
        self._apply_state(self.initial_state)
        return True

    def advance(self) -> Optional[str]:
        """Advance all traffic lights to the next state in the fixed cycle."""
        if not self.enabled:
            return None

        self.current_index = (self.current_index + 1) % len(self.cycle)
        next_state = self.cycle[self.current_index]
        self.state_history.append(next_state)
        self._apply_state(next_state)
        return next_state

    def release(self):
        """Unfreeze lights when the environment closes."""
        for light in self.traffic_lights:
            freeze = getattr(light, "freeze", None)
            if callable(freeze):
                try:
                    freeze(False)
                except Exception:
                    pass

    def build_report(self) -> Dict[str, Any]:
        return {
            "traffic_lights_enabled": bool(self.enabled),
            "traffic_light_cycle": list(self.cycle),
            "traffic_light_initial_state": self.initial_state if self.enabled else None,
            "traffic_light_current_state": self.current_state,
            "traffic_light_state_history": list(self.state_history),
            "traffic_light_count": len(self.traffic_lights),
        }

    def _apply_state(self, state_name: str):
        state_value = self._resolve_state_value(state_name)
        for light in self.traffic_lights:
            set_state = getattr(light, "set_state", None)
            if callable(set_state):
                set_state(state_value)
            set_green_time = getattr(light, "set_green_time", None)
            if callable(set_green_time):
                set_green_time(9999.0)
            set_red_time = getattr(light, "set_red_time", None)
            if callable(set_red_time):
                set_red_time(9999.0)
            set_yellow_time = getattr(light, "set_yellow_time", None)
            if callable(set_yellow_time):
                set_yellow_time(0.0)
            freeze = getattr(light, "freeze", None)
            if callable(freeze):
                freeze(True)

    def _resolve_state_value(self, state_name: str) -> Any:
        if self.carla is None:
            return state_name
        return getattr(self.carla.TrafficLightState, state_name)
