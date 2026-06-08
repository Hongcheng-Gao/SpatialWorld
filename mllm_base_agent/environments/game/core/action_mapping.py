#!/usr/bin/env python3
"""
      
          ，         /    
"""
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum


class ActionType(Enum):
    """      """
    KEY_PRESS = "key_press"
    KEY_RELEASE = "key_release"
    MOUSE_CLICK = "mouse_click"
    MOUSE_MOVE = "mouse_move"
    GAME_SPECIFIC = "game_specific"


@dataclass
class ActionMapping:
    """      """
    key: str  #     ，  "w", "a", "s", "d", "space", "up", "down"
    action_type: ActionType  #     
    action_name: str  #     ，  "move_forward", "turn_left", "jump"
    description: str  #     
    method_name: Optional[str] = None  #       ，  "move_forward"
    parameters: Dict[str, Any] = field(default_factory=dict)  #     


@dataclass
class GameActionProfile:
    """      """
    game_name: str  #     
    game_version: str = "1.0"  #     
    default_mapping: List[ActionMapping] = field(default_factory=list)  #     
    alternative_mappings: Dict[str, List[ActionMapping]] = field(default_factory=dict)  #       

    def add_mapping(self, mapping: ActionMapping):
        """      """
        self.default_mapping.append(mapping)

    def add_alternative(self, scheme_name: str, mappings: List[ActionMapping]):
        """        """
        self.alternative_mappings[scheme_name] = mappings

    def get_mapping_by_key(self, key: str) -> Optional[ActionMapping]:
        """        """
        for mapping in self.default_mapping:
            if mapping.key == key:
                return mapping
        return None

    def get_mapping_by_action(self, action_name: str) -> Optional[ActionMapping]:
        """         """
        for mapping in self.default_mapping:
            if mapping.action_name == action_name:
                return mapping
        return None

    def get_all_keys(self) -> List[str]:
        """      """
        return [mapping.key for mapping in self.default_mapping]

    def get_all_actions(self) -> List[str]:
        """      """
        return [mapping.action_name for mapping in self.default_mapping]


class ActionMapper:
    """       """

    def __init__(self):
        self.profiles: Dict[str, GameActionProfile] = {}

    def register_profile(self, profile: GameActionProfile):
        """        """
        self.profiles[profile.game_name] = profile

    def get_profile(self, game_name: str) -> Optional[GameActionProfile]:
        """        """
        return self.profiles.get(game_name)

    def map_key_to_action(self, game_name: str, key: str) -> Optional[ActionMapping]:
        """        """
        profile = self.get_profile(game_name)
        if profile:
            return profile.get_mapping_by_key(key)
        return None

    def map_action_to_key(self, game_name: str, action_name: str) -> Optional[ActionMapping]:
        """        """
        profile = self.get_profile(game_name)
        if profile:
            return profile.get_mapping_by_action(action_name)
        return None

    def create_action(self, game_name: str, key: str, metadata: Optional[Dict] = None) -> Optional[Dict]:
        """
              

        Returns:
            Dict with keys: type, key, metadata
        """
        mapping = self.map_key_to_action(game_name, key)
        if not mapping:
            return None

        action_dict = {
            "type": mapping.action_type.value,
            "key": mapping.key,
            "metadata": metadata or {}
        }

        #           
        action_dict["metadata"]["action_name"] = mapping.action_name
        action_dict["metadata"]["description"] = mapping.description

        if mapping.method_name:
            action_dict["metadata"]["method_name"] = mapping.method_name

        if mapping.parameters:
            action_dict["metadata"]["parameters"] = mapping.parameters

        return action_dict


#           
def create_snake_action_profile() -> GameActionProfile:
    """           """
    profile = GameActionProfile(
        game_name="3D Snake",
        game_version="1.0"
    )

    #     6    
    mappings = [
        ActionMapping(
            key="d",
            action_type=ActionType.KEY_PRESS,
            action_name="move_right",
            description="     (+X  )",
            method_name="move_right"
        ),
        ActionMapping(
            key="a",
            action_type=ActionType.KEY_PRESS,
            action_name="move_left",
            description="     (-X  )",
            method_name="move_left"
        ),
        ActionMapping(
            key="w",
            action_type=ActionType.KEY_PRESS,
            action_name="move_up",
            description="     (-Y  )",
            method_name="move_up"
        ),
        ActionMapping(
            key="s",
            action_type=ActionType.KEY_PRESS,
            action_name="move_down",
            description="     (+Y  )",
            method_name="move_down"
        ),
        ActionMapping(
            key="q",
            action_type=ActionType.KEY_PRESS,
            action_name="move_forward",
            description="     (-Z  )",
            method_name="move_forward"
        ),
        ActionMapping(
            key="e",
            action_type=ActionType.KEY_PRESS,
            action_name="move_backward",
            description="     (+Z  )",
            method_name="move_backward"
        ),
    ]

    for mapping in mappings:
        profile.add_mapping(mapping)

    #       ：       X/Y 
    arrow_mappings = [
        ActionMapping(
            key="right",
            action_type=ActionType.KEY_PRESS,
            action_name="move_right",
            description="     (+X  )",
            method_name="move_right"
        ),
        ActionMapping(
            key="left",
            action_type=ActionType.KEY_PRESS,
            action_name="move_left",
            description="     (-X  )",
            method_name="move_left"
        ),
        ActionMapping(
            key="up",
            action_type=ActionType.KEY_PRESS,
            action_name="move_up",
            description="     (-Y  )",
            method_name="move_up"
        ),
        ActionMapping(
            key="down",
            action_type=ActionType.KEY_PRESS,
            action_name="move_down",
            description="     (+Y  )",
            method_name="move_down"
        ),
    ]

    profile.add_alternative("arrow_keys", arrow_mappings)

    return profile


def create_maze_action_profile() -> GameActionProfile:
    """          """
    profile = GameActionProfile(
        game_name="3D Maze",
        game_version="1.0"
    )

    #         
    mappings = [
        ActionMapping(
            key="w",
            action_type=ActionType.KEY_PRESS,
            action_name="move_forward",
            description="    ",
            method_name="move_forward"
        ),
        ActionMapping(
            key="s",
            action_type=ActionType.KEY_PRESS,
            action_name="move_backward",
            description="    ",
            method_name="move_backward"
        ),
        ActionMapping(
            key="a",
            action_type=ActionType.KEY_PRESS,
            action_name="turn_left",
            description="   ",
            method_name="turn_left"
        ),
        ActionMapping(
            key="d",
            action_type=ActionType.KEY_PRESS,
            action_name="turn_right",
            description="   ",
            method_name="turn_right"
        ),
        ActionMapping(
            key="space",
            action_type=ActionType.KEY_PRESS,
            action_name="jump",
            description="  ",
            method_name="jump"
        ),
    ]

    for mapping in mappings:
        profile.add_mapping(mapping)

    return profile


def create_rubik_action_profile() -> GameActionProfile:
    """          """
    profile = GameActionProfile(
        game_name="Rubik's Cube",
        game_version="1.0"
    )

    #        
    mappings = [
        ActionMapping(
            key="u",
            action_type=ActionType.KEY_PRESS,
            action_name="rotate_up",
            description="    ",
            method_name="rotate_up"
        ),
        ActionMapping(
            key="d",
            action_type=ActionType.KEY_PRESS,
            action_name="rotate_down",
            description="    ",
            method_name="rotate_down"
        ),
        ActionMapping(
            key="l",
            action_type=ActionType.KEY_PRESS,
            action_name="rotate_left",
            description="    ",
            method_name="rotate_left"
        ),
        ActionMapping(
            key="r",
            action_type=ActionType.KEY_PRESS,
            action_name="rotate_right",
            description="    ",
            method_name="rotate_right"
        ),
        ActionMapping(
            key="f",
            action_type=ActionType.KEY_PRESS,
            action_name="rotate_front",
            description="    ",
            method_name="rotate_front"
        ),
        ActionMapping(
            key="b",
            action_type=ActionType.KEY_PRESS,
            action_name="rotate_back",
            description="    ",
            method_name="rotate_back"
        ),
    ]

    for mapping in mappings:
        profile.add_mapping(mapping)

    return profile


#          
action_mapper = ActionMapper()

#           
action_mapper.register_profile(create_snake_action_profile())
action_mapper.register_profile(create_maze_action_profile())
action_mapper.register_profile(create_rubik_action_profile())


if __name__ == "__main__":
    #         
    print("===          ===")

    #        
    snake_profile = action_mapper.get_profile("3D Snake")
    if snake_profile:
        print(f"\n         :")
        for mapping in snake_profile.default_mapping:
            print(f"  {mapping.key} -> {mapping.action_name} ({mapping.description})")

    #           
    test_keys = ["d", "a", "w", "s", "q", "e"]
    print(f"\n      :")
    for key in test_keys:
        mapping = action_mapper.map_key_to_action("3D Snake", key)
        if mapping:
            print(f"  '{key}' -> {mapping.action_name}")
        else:
            print(f"  '{key}' ->    ")

    #       
    print(f"\n      :")
    for key in test_keys:
        action_dict = action_mapper.create_action("3D Snake", key)
        if action_dict:
            print(f"  '{key}' -> {action_dict}")

    print("\n===      ===")