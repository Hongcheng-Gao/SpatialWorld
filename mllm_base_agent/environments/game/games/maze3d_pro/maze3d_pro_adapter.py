import pygame
import sys
import os
import math
import random
import numpy as np
from typing import Dict, Any, Optional, Tuple

# ---      ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir)) 
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ---        (        ) ---
try:
    from main import (
        WALL, EMPTY, START, EXIT, STAIR,
        NORTH, SOUTH, EAST, WEST,
        COLORS,
        SCREEN_WIDTH, SCREEN_HEIGHT,
        load_tower, 
        draw_3d_view
    )
except ImportError:
    try:
        from games.maze3d_pro.main import (
            WALL, EMPTY, START, EXIT, STAIR,
            NORTH, SOUTH, EAST, WEST,
            COLORS,
            SCREEN_WIDTH, SCREEN_HEIGHT,
            load_tower,
            draw_3d_view
        )
    except ImportError:
        print("Error: Cannot import game module. Make sure 'main.py' is accessible.")
        raise

class Maze3DProGame:
    """3D           -    V3       """

    def __init__(self, maze_file: str = None):
        self.maze_file = maze_file
        
        #     
        self.tower_data = {} 
        self.all_exits = {} 
        
        #     
        self.px = 1
        self.py = 1
        self.pz = 0 
        self.pDir = NORTH
        
        #     
        self.game_won = False
        self.steps_taken = 0
        self.max_steps = 2000
        self.total_floors = 0
        
        # Pygame   
        self.screen = None
        self.font = None
        self.large_font = None

    def init(self):
        """     """
        if not pygame.get_init():
            pygame.init()

        #        ，           
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.HIDDEN) 
        self.font = pygame.font.Font(None, 28)
        self.large_font = pygame.font.Font(None, 48)

        # 1.         
        target_file = self.maze_file
        if not target_file:
            base_path = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(base_path, "Level_01.txt"),
                os.path.join(base_path, "../data/maze3d_pro/Level_01.txt"),
            ]
            for path in candidates:
                if os.path.exists(path):
                    target_file = path
                    break
        
        if not target_file:
            target_file = self._create_temp_sample()

        # 2.     
        try:
            self.tower_data, start_pos, self.all_exits = load_tower(target_file)
            self.px, self.py, self.pz = start_pos
        except Exception as e:
            print(f"Error loading map: {e}")
            raise RuntimeError("Failed to load tower map")

        self.total_floors = len(self.tower_data)
        self.pDir = NORTH
        self.game_won = False
        self.steps_taken = 0

        return True

    def _create_temp_sample(self):
        """        ，       """
        content = """#######\n#S T  #\n#######\n===\n#######\n#  T E#\n#######"""
        path = 'temp_level.txt'
        with open(path, 'w') as f:
            f.write(content)
        return path

    def update(self):
        """      """
        self.steps_taken += 1
        
        #         
        current_exit = self.all_exits.get(self.pz)
        if current_exit and (self.px, self.py) == current_exit:
            self.game_won = True

        if self.steps_taken >= self.max_steps:
            pass #              

    def render(self, screen=None):
        """    """
        #    Pygame     ，          
        if pygame.get_init():
            pygame.event.pump()

        render_screen = screen if screen is not None else self.screen
        if not render_screen: return

        render_screen.fill((0, 0, 0))

        #        
        current_map = self.tower_data.get(self.pz, {})
        current_exit = self.all_exits.get(self.pz, (-1, -1))

        draw_3d_view(
            render_screen, 
            current_map, 
            self.px, self.py, 
            self.pDir, 
            current_exit
        )

        self._draw_ui(render_screen)

    def _draw_ui(self, surface):
        """   Agent     UI"""
        #     
        overlay = pygame.Surface((SCREEN_WIDTH, 60))
        overlay.set_alpha(180)
        overlay.fill((0,0,0))
        surface.blit(overlay, (0,0))

        #     
        info = f"POS: {self.px},{self.py} | FLOOR: {self.pz+1} | DIR: {self.pDir}"
        text_surf = self.font.render(info, True, COLORS['text'])
        surface.blit(text_surf, (20, 20))

        #         
        current_map = self.tower_data.get(self.pz, {})
        tile_under_foot = current_map.get((self.px, self.py), EMPTY)
        
        #    UI    (    ：   VLM        )
        if tile_under_foot == STAIR and not self.game_won:
            tips = []
            if (self.pz + 1) in self.tower_data: 
                tips.append("[Q] UP")
            if (self.pz - 1) in self.tower_data: 
                tips.append("[E] DOWN")
            
            if tips:
                t_surf = self.large_font.render(" | ".join(tips), True, (0, 255, 255))
                #            
                surface.blit(t_surf, (SCREEN_WIDTH//2 - t_surf.get_width()//2, SCREEN_HEIGHT - 80))

        #     
        if self.game_won:
            msg = self.large_font.render("MISSION COMPLETE", True, (255, 215, 0))
            surface.blit(msg, (SCREEN_WIDTH//2 - msg.get_width()//2, SCREEN_HEIGHT//2))

    def get_state(self) -> Dict[str, Any]:
        """    Agent      """
        #        numpy     VLM    (H, W, C    RGB)
        frame_array = None
        if self.screen:
            # transpose (W, H, C) -> (H, W, C)
            frame_array = np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

        current_floor = self.pz + 1
        total_floors = len(self.tower_data)

        #                （            ）
        #          ：XY     +    
        if self.all_exits:
            final_floor = max(self.all_exits.keys())
            final_exit = self.all_exits[final_floor]
            distance_to_exit = (abs(self.px - final_exit[0])
                                + abs(self.py - final_exit[1])
                                + abs(self.pz - final_floor))
        else:
            distance_to_exit = -1

        return {
            "player_x": self.px,
            "player_y": self.py,
            "player_z": self.pz,
            "floor_display": current_floor,
            "current_floor": current_floor,
            "total_floors": total_floors,
            "player_direction": self.pDir,
            "game_won": self.game_won,
            "steps_taken": self.steps_taken,
            "cell_under_foot": self.tower_data.get(self.pz, {}).get((self.px, self.py), EMPTY),
            "distance_to_exit": distance_to_exit,
            "frame": frame_array
        }

    # ---        ---

    def _granularity_to_steps(self, granularity=None) -> int:
        """Convert prompt granularity to grid cells."""
        if isinstance(granularity, int):
            return max(1, min(granularity, 3))

        value = str(granularity or "small").strip().lower()
        return {"small": 1, "medium": 2, "large": 3}.get(value, 1)

    def _move_along_direction(self, direction_map, granularity=None) -> bool:
        """Move up to the requested number of cells, stopping before walls."""
        dx, dy = direction_map[self.pDir]
        current_map = self.tower_data.get(self.pz, {})
        current_exit = self.all_exits.get(self.pz)
        moved = False

        for _ in range(self._granularity_to_steps(granularity)):
            nx, ny = self.px + dx, self.py + dy
            target = current_map.get((nx, ny), WALL)

            if current_exit and (nx, ny) == current_exit:
                self.px, self.py = nx, ny
                self.game_won = True
                return True

            if target == EMPTY or target == STAIR:
                self.px, self.py = nx, ny
                moved = True
                continue

            break

        return moved

    def move_forward(self, granularity=None) -> bool:
        """W: move forward 1/2/3 cells for small/medium/large."""
        return self._move_along_direction(
            {NORTH:(0,-1), SOUTH:(0,1), EAST:(1,0), WEST:(-1,0)},
            granularity,
        )

    def move_backward(self, granularity=None) -> bool:
        """S: move backward 1/2/3 cells for small/medium/large."""
        return self._move_along_direction(
            {NORTH:(0,1), SOUTH:(0,-1), EAST:(-1,0), WEST:(1,0)},
            granularity,
        )

    def turn_left(self):
        """A:   """
        self.pDir = {NORTH:WEST, WEST:SOUTH, SOUTH:EAST, EAST:NORTH}[self.pDir]
        return True

    def turn_right(self):
        """D:   """
        self.pDir = {NORTH:EAST, EAST:SOUTH, SOUTH:WEST, WEST:NORTH}[self.pDir]
        return True

    def climb_up(self) -> bool:
        """Q:   """
        current_map = self.tower_data.get(self.pz, {})
        tile = current_map.get((self.px, self.py), EMPTY)
        
        if tile == STAIR:
            next_floor = self.pz + 1
            if next_floor in self.tower_data:
                #       WALL，            
                target_cell = self.tower_data[next_floor].get((self.px, self.py), WALL)
                if target_cell != WALL:
                    self.pz = next_floor
                    return True
        return False

    def climb_down(self) -> bool:
        """E:   """
        current_map = self.tower_data.get(self.pz, {})
        tile = current_map.get((self.px, self.py), EMPTY)
        
        if tile == STAIR:
            prev_floor = self.pz - 1
            if prev_floor in self.tower_data:
                #       WALL
                target_cell = self.tower_data[prev_floor].get((self.px, self.py), WALL)
                if target_cell != WALL:
                    self.pz = prev_floor
                    return True
        return False

    def execute_mapped_action(self, key: str, granularity=None) -> bool:
        """          """
        if not key: return False
        key = key.lower()
        if key == "w": return self.move_forward(granularity)
        elif key == "s": return self.move_backward(granularity)
        elif key == "a": return self.turn_left()
        elif key == "d": return self.turn_right()
        elif key == "q": return self.climb_up()
        elif key == "e": return self.climb_down()
        return False

    def get_action_mapping(self, key: str):
        """      """
        if not key: return None
        mapping = {
            "w": {"key": "w", "action_name": "move_forward", "description": "    "},
            "s": {"key": "s", "action_name": "move_backward", "description": "    "},
            "a": {"key": "a", "action_name": "turn_left", "description": "   "},
            "d": {"key": "d", "action_name": "turn_right", "description": "   "},
            "q": {"key": "q", "action_name": "climb_up", "description": "   (      )"},
            "e": {"key": "e", "action_name": "climb_down", "description": "   (      )"},
        }
        return mapping.get(key.lower())

if __name__ == "__main__":
    #     
    game = Maze3DProGame()
    try:
        game.init()
        #         ，       frame_array   
        game.render() 
        state = game.get_state()
        print("Initial State POS:", state["player_x"], state["player_y"])
        if state["frame"] is not None:
            print(f"Frame shape ready: {state['frame'].shape}")
        
        #     
        game.execute_mapped_action("w")
        game.render()
        print("After W POS:", game.get_state()["player_x"], game.get_state()["player_y"])
        
    except Exception as e:
        print(f"Test Failed: {e}")
