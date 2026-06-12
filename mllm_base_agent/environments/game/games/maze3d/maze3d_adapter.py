"""
3D        -  3D      MLLM    
"""
import pygame
import sys
import os
import math
import random
from typing import Dict, Any, Optional

# ---      ---
#        games.maze3d     
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir)) #       project/games/maze3d/adapter.py
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# ---        ---
#   ：                 maze3d_pygame.py
#                 
try:
    from maze3d_pygame import (
        WALL, EMPTY, START, EXIT, NORTH, SOUTH, EAST, WEST,
        WALL_COLORS, FLOOR_COLORS, CEILING_COLORS, EXIT_COLORS, UI_COLORS,
        SCREEN_WIDTH, SCREEN_HEIGHT, VIEW_WIDTH, VIEW_HEIGHT,
        load_maze, darken_color, blend_colors, 
        draw_3d_view, draw_modern_ui #                 
    )
except ImportError:
    #              
    from games.maze3d.maze3d_pygame import (
        WALL, EMPTY, START, EXIT, NORTH, SOUTH, EAST, WEST,
        WALL_COLORS, FLOOR_COLORS, CEILING_COLORS, EXIT_COLORS, UI_COLORS,
        SCREEN_WIDTH, SCREEN_HEIGHT, VIEW_WIDTH, VIEW_HEIGHT,
        load_maze, darken_color, blend_colors, 
        draw_3d_view, draw_modern_ui
    )

class Maze3DGame:
    """3D        """

    def __init__(self, maze_file: str = None):
        """
           3D    
        Args:
            maze_file:       ，   None       
        """
        self.maze_file = maze_file
        self.maze = None
        self.px = None
        self.py = None
        self.exitx = None
        self.exity = None
        self.WIDTH = None
        self.HEIGHT = None
        self.pDir = NORTH
        self.game_won = False
        self.steps_taken = 0
        self.max_steps = 1000  #       
        self.animation_time = 0

        #    pygame    
        self.screen = None
        self.font = None
        self.small_font = None
        self.large_font = None

    def init(self):
        """     """
        #    pygame（       ）
        if not pygame.get_init():
            pygame.init()

        #       （             ）
        self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        #      
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)
        self.large_font = pygame.font.Font(None, 56)

        #     
        if self.maze_file:
            result = load_maze(self.maze_file)
        else:
            #                    
            search_paths = ['.', 'games/maze3d']
            found_file = None
            
            for path in search_paths:
                if os.path.exists(path):
                    files = [os.path.join(path, f) for f in os.listdir(path) if f.startswith('maze') and f.endswith('.txt')]
                    if files:
                        found_file = files[0]
                        break
            
            if found_file:
                result = load_maze(found_file)
            else:
                #       
                sample_maze = """##########
#S       #
# #### # #
# #    # #
# # #### #
# #      #
# ###### #
#       E#
##########"""
                #       
                target_dir = 'games/maze3d'
                os.makedirs(target_dir, exist_ok=True)
                target_file = os.path.join(target_dir, 'maze_sample.txt')
                
                with open(target_file, 'w') as f:
                    f.write(sample_maze)
                result = load_maze(target_file)

        #    load_maze    None    
        if result is None or result[0] is None:
            raise RuntimeError("Failed to load maze")

        self.maze, self.px, self.py, self.exitx, self.exity, self.WIDTH, self.HEIGHT = result
        self.pDir = NORTH
        self.game_won = False
        self.steps_taken = 0
        self.animation_time = 0

        return True

    def update(self):
        """      """
        self.steps_taken += 1
        #   ：animation_time     render()      

        #       
        if (self.px, self.py) == (self.exitx, self.exity):
            self.game_won = True

        #       
        if self.steps_taken >= self.max_steps:
            self.game_won = False  #     

    def render(self, screen=None):
        """      """
        #          ，          
        render_screen = screen if screen is not None else self.screen
        if not render_screen:
            return

        #     
        render_screen.fill((5, 10, 15))

        #                
        self.animation_time += 1

        #                 
        #     Hack，             animation_time
        try:
            import sys
            #               
            if 'games.maze3d.maze3d_pygame' in sys.modules:
                mod = sys.modules['games.maze3d.maze3d_pygame']
            elif 'maze3d_pygame' in sys.modules:
                mod = sys.modules['maze3d_pygame']
            else:
                mod = None
            
            original_time = 0
            if mod:
                original_time = getattr(mod, 'animation_time', 0)
                mod.animation_time = self.animation_time
            
            #       
            #   ：         draw_modern_ui   
            draw_3d_view(render_screen, self.maze, self.px, self.py, self.pDir, self.exitx, self.exity)
            draw_modern_ui(render_screen, self.px, self.py, self.pDir)
            
        except Exception as e:
            print(f"Render error: {e}")
        finally:
            #     （             ，     ）
            if mod:
                mod.animation_time = original_time

        #       ，      
        if self.game_won:
            self._render_victory(render_screen)

    def _render_victory(self, screen=None):
        """      """
        render_screen = screen if screen is not None else self.screen
        if not render_screen:
            return

        victory_overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        victory_overlay.fill((0, 0, 0, 200))
        render_screen.blit(victory_overlay, (0, 0))

        static_animation_time = 10 

        #       
        for i in range(3):
            scale = 1.0 + math.sin(static_animation_time * 0.1 + i) * 0.1
            victory_font = pygame.font.Font(None, int(72 * scale))
            
            # ===       ===
            #        EXIT_COLORS['pulse']，            
            #        'core'   'glow'     ，     
            color_mix = blend_colors(
                EXIT_COLORS['glow'],
                EXIT_COLORS['core'], #   ：   'core'        'pulse'
                (math.sin(static_animation_time * 0.1) + 1) / 2
            )
            
            victory_text = victory_font.render("VICTORY!", True, color_mix)

            x = SCREEN_WIDTH // 2 - victory_text.get_width() // 2
            y = SCREEN_HEIGHT // 2 - 80 + i * 2
            render_screen.blit(victory_text, (x, y))

        subtitle = self.font.render("You escaped the maze!", True, UI_COLORS['text'])
        render_screen.blit(subtitle, (SCREEN_WIDTH // 2 - subtitle.get_width() // 2,
                                      SCREEN_HEIGHT // 2 + 20))

        stats = self.small_font.render(f"Steps taken: {self.steps_taken}", True, UI_COLORS['accent'])
        render_screen.blit(stats, (SCREEN_WIDTH // 2 - stats.get_width() // 2,
                                   SCREEN_HEIGHT // 2 + 60))

    def get_state(self) -> Dict[str, Any]:
        """      """
        return {
            "player_x": self.px,
            "player_y": self.py,
            "player_direction": self.pDir,
            "exit_x": self.exitx,
            "exit_y": self.exity,
            "game_won": self.game_won,
            "steps_taken": self.steps_taken,
            "max_steps": self.max_steps,
            "maze_width": self.WIDTH,
            "maze_height": self.HEIGHT,
            "distance_to_exit": abs(self.px - self.exitx) + abs(self.py - self.exity) if self.px is not None else 0,
            "game_over": self.game_won or self.steps_taken >= self.max_steps,
            "success": self.game_won,
        }

    def reset(self):
        """    """
        self.init()

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
        moved = False

        for _ in range(self._granularity_to_steps(granularity)):
            new_x, new_y = self.px + dx, self.py + dy
            if self.maze.get((new_x, new_y), WALL) == EMPTY or (new_x, new_y) == (self.exitx, self.exity):
                self.px, self.py = new_x, new_y
                moved = True
                continue

            break

        if (self.px, self.py) == (self.exitx, self.exity):
            self.game_won = True

        return moved

    def move_forward(self, granularity=None) -> bool:
        return self._move_along_direction(
            {NORTH:(0,-1), SOUTH:(0,1), EAST:(1,0), WEST:(-1,0)},
            granularity,
        )

    def turn_left(self):
        self.pDir = {NORTH: WEST, WEST: SOUTH, SOUTH: EAST, EAST: NORTH}[self.pDir]

    def turn_right(self):
        self.pDir = {NORTH: EAST, EAST: SOUTH, SOUTH: WEST, WEST: NORTH}[self.pDir]

    def move_backward(self, granularity=None) -> bool:
        return self._move_along_direction(
            {NORTH:(0,1), SOUTH:(0,-1), EAST:(-1,0), WEST:(1,0)},
            granularity,
        )

    def execute_mapped_action(self, key: str, granularity=None) -> bool:
        if key == "w": return self.move_forward(granularity)
        elif key == "a": self.turn_left(); return True
        elif key == "d": self.turn_right(); return True
        elif key == "s": return self.move_backward(granularity)
        else: return False

    def get_action_mapping(self, key: str):
        #          ，       core  ，        
        #                 import core.action_mapping
        return {
            "w": {"key": "w", "action_name": "move_forward", "description": "    "},
            "a": {"key": "a", "action_name": "turn_left", "description": "   "},
            "d": {"key": "d", "action_name": "turn_right", "description": "   "},
            "s": {"key": "s", "action_name": "move_backward", "description": "    "},
        }.get(key)

if __name__ == "__main__":
    game = Maze3DGame()
    game.init()
    print("Initial state:", game.get_state())
    game.move_forward()
    print("Moved forward:", game.get_state())
