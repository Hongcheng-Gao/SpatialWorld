"""
3D          -  Block3D    MLLM    
    ：New Visual Style (Asymmetric View)
"""

import pygame
import sys
import os
import math
import random
from typing import Dict, Any, Optional, List, Tuple
from data.Block3D.level_data import LEVEL_DESIGNS
#          (    )
try:
    from core.action_mapping import ActionMapping, ActionType, GameActionProfile
except ImportError:
    from enum import Enum

    class ActionType(Enum):
        KEY_PRESS = "key_press"

    class ActionMapping:
        def __init__(self, key, action_type, action_name, description, method_name=None):
            self.key = key
            self.action_type = action_type
            self.action_name = action_name
            self.description = description
            self.method_name = method_name

    class GameActionProfile:
        def __init__(self, game_name, game_version="1.0"):
            self.game_name = game_name
            self.game_version = game_version
            self.default_mapping = []


class Block3DGameAdapter:
    """3D          """

    def __init__(self, level: int = 0, max_steps: int = 200):
        """
           3D      
        """
        self.level = level
        self.max_steps = max_steps

        # --- [   ]          ---
        self.SCREEN_WIDTH = 1100
        self.SCREEN_HEIGHT = 600
        self.BLOCK_SIZE = 40
        self.GRID_SIZE = 6
        
        #        
        self.OFFSET_X = 640 
        self.OFFSET_Y = 280

        # --- [   ]        ---
        self.WHITE = (255, 255, 255)
        self.BLACK = (20, 20, 20)
        self.GRAY = (80, 80, 80)
        self.GREEN = (50, 200, 50)
        self.BLUE = (60, 120, 220)
        self.RED = (220, 60, 60)
        self.HIGHLIGHT = (100, 200, 250)
        self.GRID_COLOR = (70, 70, 75) #   
        self.EYE_COLOR = (255, 255, 0)

        #         
        self.SHADOW_COLOR = (30, 30, 35)       #       
        self.RIM_LIGHT_COLOR = (255, 255, 220) #       
        self.GROUND_PLANE_COLOR = (45, 45, 50) #       

        #        
        self.COLOR_X = self.RED
        self.COLOR_Y = self.GREEN
        self.COLOR_Z = self.BLUE

        #      (    )
        self.LEVEL_DESIGNS = LEVEL_DESIGNS

        #     
        self.blocks = {} 
        self.cursor = [2, 2, 0] 
        self.target_blocks = {} 
        self.won = False 
        self.game_over = False 
        self.steps_taken = 0 
        self.score = 0 

        # Pygame  
        self.screen = None
        self.font = None
        self.bold_font = None
        self.large_font = None
        self.debug_font = None

    def init(self):
        """     """
        if not pygame.get_init():
            pygame.init()

        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        self.font = pygame.font.Font(None, 14)
        self.bold_font = pygame.font.Font(None, 18)
        self.large_font = pygame.font.Font(None, 40)
        self.debug_font = pygame.font.Font(None, 14)

        self.reset_game()
        return True

    def reset_game(self):
        """      """
        level_data = self.LEVEL_DESIGNS[str(self.level)]
        self.blocks = {}
        self.cursor = [2, 2, 0]
        self.target_blocks = {coord: True for coord in level_data["blocks"]}
        self.won = False
        self.game_over = False
        self.steps_taken = 0
        self.score = 0

    def update(self):
        """      """
        self.steps_taken += 1

        if self.steps_taken >= self.max_steps:
            self.game_over = True
            return False

        if self.blocks == self.target_blocks:
            self.won = True
            self.game_over = True
            self.score = len(self.blocks)
            return False

        return not self.game_over

    def render(self, screen=None):
        """      """
        render_screen = screen if screen is not None else self.screen
        if not render_screen:
            return

        render_screen.fill(self.BLACK)

        #         (    )
        self._draw_orthographic_views(render_screen)

        #      
        pygame.draw.line(render_screen, self.WHITE, (220, 20), (220, self.SCREEN_HEIGHT - 20), 2)

        #     3D   (    )
        self._draw_3d_scene(render_screen)

        #       
        render_screen.blit(self.debug_font.render(f"Cursor: {tuple(self.cursor)}", True, self.WHITE),
                          (self.SCREEN_WIDTH - 200, self.SCREEN_HEIGHT - 40))

        if self.won:
            render_screen.blit(self.large_font.render("VICTORY!", True, self.GREEN), (500, 50))

    def should_continue(self) -> bool:
        return not self.game_over

    def get_state(self) -> Dict[str, Any]:
        """       (    )"""
        matched_blocks = 0
        for block_pos in self.blocks:
            if block_pos in self.target_blocks:
                matched_blocks += 1

        total_target_blocks = len(self.target_blocks)
        progress = matched_blocks / total_target_blocks if total_target_blocks > 0 else 0

        return {
            "cursor_x": self.cursor[0],
            "cursor_y": self.cursor[1],
            "cursor_z": self.cursor[2],
            "blocks_placed": len(self.blocks),
            "target_blocks": total_target_blocks,
            "matched_blocks": matched_blocks,
            "progress": progress,
            "score": self.score,
            "won": self.won,
            "game_over": self.game_over,
            "steps_taken": self.steps_taken,
            "max_steps": self.max_steps,
            "level": self.level,
            "level_name": self.LEVEL_DESIGNS[str(self.level)]["name"],
            "success": self.won,
            "steps_remaining": self.max_steps - self.steps_taken,
            "completion_percentage": progress * 100,
            "blocks_remaining": total_target_blocks - matched_blocks
        }

    def reset(self):
        self.reset_game()

    # ---        (    ) ---
    def move_left(self) -> bool:
        if self.game_over: return False
        self.cursor[0] = max(0, self.cursor[0] - 1)
        return self.update()

    def move_right(self) -> bool:
        if self.game_over: return False
        self.cursor[0] = min(self.GRID_SIZE - 1, self.cursor[0] + 1)
        return self.update()

    def move_up(self) -> bool:
        if self.game_over: return False
        self.cursor[1] = max(0, self.cursor[1] - 1)
        return self.update()

    def move_down(self) -> bool:
        if self.game_over: return False
        self.cursor[1] = min(self.GRID_SIZE - 1, self.cursor[1] + 1)
        return self.update()

    def move_forward(self) -> bool:
        if self.game_over: return False
        self.cursor[2] = min(self.GRID_SIZE - 1, self.cursor[2] + 1)
        return self.update()

    def move_backward(self) -> bool:
        if self.game_over: return False
        self.cursor[2] = max(0, self.cursor[2] - 1)
        return self.update()

    def place_block(self) -> bool:
        if self.game_over: return False
        pos = tuple(self.cursor)
        if pos in self.blocks:
            del self.blocks[pos]
        else:
            self.blocks[pos] = True
        if self.blocks == self.target_blocks:
            self.won = True
            self.game_over = True
            self.score = len(self.blocks)
        return self.update()

    # --- [     ]          ---

    def _iso_project(self, x, y, z):
        """
        [  ]         ，        
        """
        #     
        scale_x = 1.0
        scale_y = 0.85 

        #     
        slope = 0.5

        screen_x = int(self.OFFSET_X + (x * scale_x - y * scale_y) * self.BLOCK_SIZE)
        screen_y = int(self.OFFSET_Y + (x * scale_x + y * scale_y) * (self.BLOCK_SIZE * slope) - (z * self.BLOCK_SIZE))
        
        return screen_x, screen_y

    def _draw_drop_shadow(self, screen, x, y):
        """[  ]       """
        p1 = self._iso_project(x, y, 0.05) 
        p2 = self._iso_project(x+1, y, 0.05)
        p3 = self._iso_project(x+1, y+1, 0.05)
        p4 = self._iso_project(x, y+1, 0.05)
        pygame.draw.polygon(screen, self.SHADOW_COLOR, [p1, p2, p3, p4])

    def _draw_cube(self, screen, x, y, z, color, outline=False):
        """[  ]   3D   ，      """
        def get_pt(dx, dy, dz):
            return self._iso_project(x + dx, y + dy, z + dz)

        p_top_0 = get_pt(0, 0, 1) 
        p_top_1 = get_pt(1, 0, 1) 
        p_top_2 = get_pt(1, 1, 1) 
        p_top_3 = get_pt(0, 1, 1) 
        p_bot_2 = get_pt(1, 1, 0) 

        pts_top = [p_top_0, p_top_1, p_top_2, p_top_3]
        pts_right = [get_pt(1,0,1), get_pt(1,1,1), get_pt(1,1,0), get_pt(1,0,0)]
        pts_left = [get_pt(0,1,1), get_pt(1,1,1), get_pt(1,1,0), get_pt(0,1,0)]

        if outline:
            thickness = 3 
            for pts in [pts_top, pts_right, pts_left]: 
                pygame.draw.polygon(screen, self.HIGHLIGHT, pts, thickness)
        else:
            c_top = color
            c_right = (max(0,color[0]-40), max(0,color[1]-40), max(0,color[2]-40))
            c_left = (max(0,color[0]-80), max(0,color[1]-80), max(0,color[2]-80))
            
            pygame.draw.polygon(screen, c_top, pts_top)
            pygame.draw.polygon(screen, c_right, pts_right)
            pygame.draw.polygon(screen, c_left, pts_left)
            
            for pts in [pts_top, pts_right, pts_left]: 
                pygame.draw.polygon(screen, self.BLACK, pts, 1)

            # [  ]       
            pygame.draw.line(screen, self.RIM_LIGHT_COLOR, p_top_3, p_top_0, 2)
            pygame.draw.line(screen, self.RIM_LIGHT_COLOR, p_top_0, p_top_1, 2)
            pygame.draw.line(screen, self.RIM_LIGHT_COLOR, p_top_2, p_bot_2, 2)
            pygame.draw.line(screen, self.RIM_LIGHT_COLOR, p_top_1, p_top_2, 1)

    def _draw_orthographic_views(self, screen):
        """        (      ，       )"""
        view_cfg = [
            ("TOP (XY)", lambda x, y, z: ((self.GRID_SIZE - 1) - y, x), "<- Green(Y)", "Red(X) v", "5", "0"),
            ("FRONT (YZ)", lambda x, y, z: ((self.GRID_SIZE - 1) - y, (self.GRID_SIZE - 1) - z), "<- Green(Y)", "Blue(Z) ^", "5", "0"),
            ("SIDE (XZ)", lambda x, y, z: (x, (self.GRID_SIZE - 1) - z), "Red(X) ->", "Blue(Z) ^", "0", "0")
        ]

        pad, vs, y_off = 40, 15, 30
        for title, mapping, xlabel, ylabel, origin_x_label, origin_y_label in view_cfg:
            color = self.WHITE
            if "SIDE" in title: color = self.EYE_COLOR
            if "FRONT" in title: color = self.HIGHLIGHT

            screen.blit(self.bold_font.render(title, True, color), (pad, y_off - 20))
            pygame.draw.rect(screen, self.GRID_COLOR, (pad, y_off, self.GRID_SIZE * vs, self.GRID_SIZE * vs), 1)

            screen.blit(self.font.render(origin_x_label, True, self.GRAY), (pad - 10, y_off))
            end_label = "0" if origin_x_label == "5" else "5"
            screen.blit(self.font.render(end_label, True, self.GRAY), (pad + self.GRID_SIZE * vs + 2, y_off))

            screen.blit(self.font.render(xlabel, True, self.GRAY),
                       (pad + self.GRID_SIZE * vs // 2 - 20, y_off + self.GRID_SIZE * vs + 5))
            screen.blit(self.font.render(ylabel, True, self.GRAY),
                       (pad + 5, y_off + self.GRID_SIZE * vs // 2))

            for (x, y, z) in self.target_blocks:
                u, v = mapping(x, y, z)
                pygame.draw.rect(screen, self.GRAY, (pad + u * vs, y_off + v * vs, vs - 1, vs - 1))

            y_off += self.GRID_SIZE * vs + 60

    def _draw_3d_scene(self, screen):
        """[  ]   3D   (                  )"""
        
        # A.       
        ground_pts = [
            self._iso_project(0, 0, 0),
            self._iso_project(self.GRID_SIZE, 0, 0),
            self._iso_project(self.GRID_SIZE, self.GRID_SIZE, 0),
            self._iso_project(0, self.GRID_SIZE, 0)
        ]
        pygame.draw.polygon(screen, self.GROUND_PLANE_COLOR, ground_pts)

        # B.      
        for x in range(self.GRID_SIZE + 1):
            p1 = self._iso_project(x, 0, 0)
            p2 = self._iso_project(x, self.GRID_SIZE, 0)
            pygame.draw.line(screen, self.GRID_COLOR, p1, p2, 1)
        for y in range(self.GRID_SIZE + 1):
            p1 = self._iso_project(0, y, 0)
            p2 = self._iso_project(self.GRID_SIZE, y, 0)
            pygame.draw.line(screen, self.GRID_COLOR, p1, p2, 1)

        # C.       
        active_columns = set()
        for (bx, by, bz) in self.blocks:
            if bz >= 0: active_columns.add((bx, by))
            
        cx, cy, cz = self.cursor
        if cz > 0: active_columns.add((cx, cy))

        for (sx, sy) in active_columns:
            self._draw_drop_shadow(screen, sx, sy)

        # D.          
        origin = self._iso_project(0, 0, 0)
        pygame.draw.line(screen, self.COLOR_X, origin, self._iso_project(7,0,0), 3)
        pygame.draw.line(screen, self.COLOR_Y, origin, self._iso_project(0,7,0), 3)
        pygame.draw.line(screen, self.COLOR_Z, origin, self._iso_project(0,0,7), 3)
        screen.blit(self.bold_font.render("X", True, self.COLOR_X), self._iso_project(7.2,0,0))
        screen.blit(self.bold_font.render("Y", True, self.COLOR_Y), self._iso_project(0,7.2,0))
        
        #        
        self._draw_projection_lines(screen)

        # E.     （     ）
        draw_list = [{'c': k, 't': 'b'} for k in self.blocks] + [{'c': tuple(self.cursor), 't': 'c'}]
        #           
        draw_list.sort(key=lambda k: sum(k['c']))

        for item in draw_list:
            self._draw_cube(screen, *item['c'],
                           self.BLUE if item['t'] == 'b' else self.HIGHLIGHT,
                           outline=(item['t'] == 'c'))

    def _draw_projection_lines(self, screen):
        """[  ]      ，        """
        #       (       ，   )
        focus_x, focus_y, focus_z = -0.8, -0.5, 0.1
        dist = 8.0

        # --- TOP VIEW ---
        eye_top_start = self._iso_project(focus_x, focus_y, focus_z + dist-2)
        eye_top_end   = self._iso_project(focus_x, focus_y, focus_z) 
        self._draw_dashed_line(screen, eye_top_start, eye_top_end)
        self._draw_eye_icon(screen, eye_top_start, "TOP")

        # --- FRONT VIEW ---
        eye_front_start = self._iso_project(focus_x + dist, focus_y, focus_z)
        eye_front_end   = self._iso_project(focus_x, focus_y, focus_z)
        self._draw_dashed_line(screen, eye_front_start, eye_front_end)
        self._draw_eye_icon(screen, eye_front_start, "FRONT (along X)")

        # --- SIDE VIEW ---
        eye_side_start = self._iso_project(focus_x, focus_y + dist, focus_z)
        eye_side_end   = self._iso_project(focus_x, focus_y, focus_z)
        self._draw_dashed_line(screen, eye_side_start, eye_side_end)
        self._draw_eye_icon(screen, eye_side_start, "SIDE (along Y)")
        
        #     
        ix, iy = eye_front_end
        pygame.draw.circle(screen, self.EYE_COLOR, (ix, iy), 4)
        pygame.draw.circle(screen, self.RED, (ix, iy), 2)

    def _draw_dashed_line(self, screen, start_pos, end_pos, color=None):
        """    """
        if color is None:
            color = self.EYE_COLOR

        x1, y1 = start_pos
        x2, y2 = end_pos
        dl = 10
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            return

        angle = math.atan2(y2 - y1, x2 - x1)
        segments = int(length / dl)
        for i in range(0, segments, 2):
            start = (x1 + math.cos(angle) * i * dl, y1 + math.sin(angle) * i * dl)
            end = (x1 + math.cos(angle) * (i + 1) * dl, y1 + math.sin(angle) * (i + 1) * dl)
            pygame.draw.line(screen, color, start, end, 2)

    def _draw_eye_icon(self, screen, pos, text):
        """      """
        x, y = pos
        pygame.draw.circle(screen, self.EYE_COLOR, (int(x), int(y)), 8, 2)
        pygame.draw.circle(screen, self.EYE_COLOR, (int(x), int(y)), 3)
        screen.blit(self.debug_font.render(text, True, self.EYE_COLOR), (x + 15, y - 10))

    #        (    )
    @classmethod
    def get_action_profile(cls) -> GameActionProfile:
        """        """
        profile = GameActionProfile(
            game_name="3D Block Builder",
            game_version="1.1" # Version update
        )

        mappings = [
            ActionMapping(key="left", action_type=ActionType.KEY_PRESS, action_name="move_left", description="       (-X  )", method_name="move_left"),
            ActionMapping(key="right", action_type=ActionType.KEY_PRESS, action_name="move_right", description="       (+X  )", method_name="move_right"),
            ActionMapping(key="up", action_type=ActionType.KEY_PRESS, action_name="move_up", description="       (-Y  )", method_name="move_up"),
            ActionMapping(key="down", action_type=ActionType.KEY_PRESS, action_name="move_down", description="       (+Y  )", method_name="move_down"),
            ActionMapping(key="w", action_type=ActionType.KEY_PRESS, action_name="move_forward", description="       (+Z  )", method_name="move_forward"),
            ActionMapping(key="s", action_type=ActionType.KEY_PRESS, action_name="move_backward", description="       (-Z  )", method_name="move_backward"),
            ActionMapping(key="space", action_type=ActionType.KEY_PRESS, action_name="place_block", description="       ", method_name="place_block"),
        ]

        for mapping in mappings:
            profile.add_mapping(mapping)

        return profile

    def get_action_mapping(self, key: str) -> Optional[ActionMapping]:
        profile = self.get_action_profile()
        return profile.get_mapping_by_key(key)

    def execute_mapped_action(self, key: str) -> bool:
        mapping = self.get_action_mapping(key)
        if not mapping or not mapping.method_name:
            return False
        method = getattr(self, mapping.method_name, None)
        if method and callable(method):
            return method()
        return False


#          
if __name__ == "__main__":
    game = Block3DGameAdapter(level=0, max_steps=50)

    #      
    game.init()
    print("Initial state:", game.get_state())

    #     
    game.move_right()
    #     render  （   headless         ）
    # pygame.display.set_mode((1100, 600))
    # game.render()
    # pygame.display.flip()
    # pygame.time.wait(2000)