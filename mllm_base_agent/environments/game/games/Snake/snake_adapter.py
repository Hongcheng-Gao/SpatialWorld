"""
3D         -  3D       MLLM    
    ：Body-Start, Grid-Frame Visuals, W/S Depth Control
"""
import pygame
import sys
import os
import math
import random
from typing import Dict, Any, Optional, List

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


class Point3D:
    """3D  """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def copy(self):
        return Point3D(self.x, self.y, self.z)


class SnakeGameAdapter:
    """3D         """

    def __init__(self, grid_size: int = 5, max_steps: int = 200):
        """
           3D     
        """
        self.grid_size = grid_size
        self.max_steps = max_steps

        # --- [   ]          ---
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 600
        self.CUBE_SPACING = 80  # Updated
        self.FOV = 600

        # --- [   ]        ---
        self.BG_COLOR = (20, 20, 25)
        self.GRID_FRAME_COLOR = (80, 80, 100) # New
        self.GRID_DOT_COLOR = (50, 50, 60)    # New
        self.PROJECTION_LINE_COLOR = (60, 60, 70) # New
        
        self.WHITE = (220, 220, 220)
        self.RED = (255, 80, 80)
        self.GREEN = (80, 255, 80)
        self.HEAD_COLOR = (120, 255, 120) # Updated
        self.BLUE = (80, 80, 255)
        self.YELLOW = (255, 220, 0)

        #     
        self.snake = []
        self.last_move_dir = None
        self.food = None
        self.score = 0
        self.game_over = False
        self.steps_taken = 0

        # --- [   ]        ---
        self.angle_x = 0.5 
        self.angle_y = 0.6

        # Pygame  
        self.screen = None
        self.font = None
        self.large_font = None
        self.title_font = None

    def init(self):
        """     """
        if not pygame.get_init():
            pygame.init()

        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))

        self.font = pygame.font.Font(None, 14)
        self.large_font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 40)

        self.reset_game()
        return True

    def reset_game(self):
        """[   ]        -             """
        mid = self.grid_size // 2
        start_y = self.grid_size - 1
        start_z = mid
        
        #         2 (1  + 1 )
        target_length = 2
        self.snake = []
        
        #        (+X)，      X         
        for i in range(target_length):
            pos = Point3D(mid - i, start_y, start_z)
            self.snake.append(pos)

        self.last_move_dir = Point3D(1, 0, 0)  #       ，      
        self.food = self._generate_food()
        self.score = 0
        self.game_over = False
        self.steps_taken = 0

    def _generate_food(self):
        """      """
        while True:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            z = random.randint(0, self.grid_size - 1)
            p = Point3D(x, y, z)

            collision = False
            for segment in self.snake:
                if p == segment:
                    collision = True
                    break
            if not collision:
                return p

    def update(self):
        """       -        _step()     """
        #       
        if self.steps_taken >= self.max_steps:
            self.game_over = True
        return not self.game_over

    def render(self, screen=None):
        """      """
        render_screen = screen if screen is not None else self.screen
        if not render_screen:
            return

        render_screen.fill(self.BG_COLOR)

        # 1.     （  +  ）
        self._draw_grid_lattice(render_screen)

        # 2.             (Painter's Algorithm)
        draw_list = []
        draw_list.append({'type': 'food', 'obj': self.food, 'draw_proj': True})
        for i, segment in enumerate(self.snake):
            is_head = (i == 0)
            #           
            draw_list.append({'type': 'snake', 'obj': segment, 'is_head': is_head, 'draw_proj': is_head})

        # --- [   ]            ---
        draw_list.sort(key=lambda item: self._get_z_depth(item['obj']), reverse=True)

        # 3.        (       )
        self._draw_snake_body_lines(render_screen)

        # 4.        
        for item in draw_list:
            if item['type'] == 'food':
                self._draw_element(render_screen, item['obj'], self.YELLOW, 14, draw_projections=True)
            elif item['type'] == 'snake':
                color = self.HEAD_COLOR if item['is_head'] else self.GREEN
                self._draw_element(render_screen, item['obj'], color, 12, item['is_head'], draw_projections=item['draw_proj'])

        # 5. UI   
        self._draw_direction_guides(render_screen)
        self._draw_axis_gizmo(render_screen)

        score_text = self.large_font.render(f"Score: {self.score}", True, self.WHITE)
        render_screen.blit(score_text, (20, 20))

        help_text = self.font.render(f"Steps: {self.steps_taken}/{self.max_steps} | Q/E: Depth", True, (150, 150, 150))
        render_screen.blit(help_text, (20, self.SCREEN_HEIGHT - 30))

        if self.game_over:
            self._render_game_over(render_screen)

    def should_continue(self) -> bool:
        return not self.game_over

    def get_state(self) -> Dict[str, Any]:
        """       (      )"""
        head = self.snake[0] if self.snake else Point3D(0, 0, 0)
        return {
            "snake_length": len(self.snake),
            "snake_head_x": head.x,
            "snake_head_y": head.y,
            "snake_head_z": head.z,
            "last_move_dir_x": self.last_move_dir.x if self.last_move_dir else 0,
            "last_move_dir_y": self.last_move_dir.y if self.last_move_dir else 0,
            "last_move_dir_z": self.last_move_dir.z if self.last_move_dir else 0,
            "food_x": self.food.x if self.food else 0,
            "food_y": self.food.y if self.food else 0,
            "food_z": self.food.z if self.food else 0,
            "score": self.score,
            "game_over": self.game_over,
            "steps_taken": self.steps_taken,
            "max_steps": self.max_steps,
            "grid_size": self.grid_size,
            "distance_to_food": abs(head.x - self.food.x) + abs(head.y - self.food.y) + abs(head.z - self.food.z) if self.food else 0,
            "success": self.score > 0,
            "steps_remaining": self.max_steps - self.steps_taken
        }

    def reset(self):
        self.reset_game()

    def _try_move(self, dx, dy, dz):
        """    """
        if self.game_over:
            # raise RuntimeError("Game is over") # Optional: silent fail preferred in adapters
            return False

        if (self.last_move_dir and
            self.last_move_dir.x + dx == 0 and
            self.last_move_dir.y + dy == 0 and
            self.last_move_dir.z + dz == 0):
            return False

        self.last_move_dir = Point3D(dx, dy, dz)
        return self._step(dx, dy, dz)

    def _step(self, dx, dy, dz):
        head = self.snake[0]
        new_head = Point3D(head.x + dx, head.y + dy, head.z + dz)

        if (new_head.x < 0 or new_head.x >= self.grid_size or
            new_head.y < 0 or new_head.y >= self.grid_size or
            new_head.z < 0 or new_head.z >= self.grid_size):
            self.game_over = True
            #           
            self.steps_taken += 1
            if self.steps_taken >= self.max_steps:
                self.game_over = True
            return False

        #   ：     new_game    ，   snake[:-1]    
        for segment in self.snake[:-1]:
            if new_head == segment:
                self.game_over = True
                #           
                self.steps_taken += 1
                if self.steps_taken >= self.max_steps:
                    self.game_over = True
                return False

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            if len(self.snake) == self.grid_size ** 3:
                self.game_over = True
            else:
                self.food = self._generate_food()
        else:
            self.snake.pop()

        #          
        self.steps_taken += 1
        if self.steps_taken >= self.max_steps:
            self.game_over = True

        return not self.game_over

    # ---        ---
    def move_right(self) -> bool: return self._try_move(1, 0, 0)
    def move_left(self) -> bool: return self._try_move(-1, 0, 0)
    def move_up(self) -> bool: return self._try_move(0, -1, 0)
    def move_down(self) -> bool: return self._try_move(0, 1, 0)
    def move_forward(self) -> bool: return self._try_move(0, 0, -1) # q (In)
    def move_backward(self) -> bool: return self._try_move(0, 0, 1)  # e (Out)

    # --- [   ]        ---
    
    def _apply_depth_shading(self, color, scale):
        """[  ]       """
        factor = min(1.2, max(0.4, scale * 0.8))
        r = int(min(255, color[0] * factor))
        g = int(min(255, color[1] * factor))
        b = int(min(255, color[2] * factor))
        return (r, g, b)

    def _project_point(self, x, y, z, offset_x=None, offset_y=None, scale_mult=1.0):
        if offset_x is None: offset_x = self.SCREEN_WIDTH / 2
        if offset_y is None: offset_y = self.SCREEN_HEIGHT / 2

        # 3D     (     )
        cos_x = math.cos(self.angle_x)
        sin_x = math.sin(self.angle_x)
        y_rot = y * cos_x - z * sin_x
        z_rot = y * sin_x + z * cos_x
        y = y_rot
        z = z_rot

        cos_y = math.cos(self.angle_y)
        sin_y = math.sin(self.angle_y)
        x_rot = x * cos_y + z * sin_y
        z_rot = -x * sin_y + z * cos_y
        x = x_rot
        z = z_rot

        camera_distance = 600 # Updated
        if camera_distance + z <= 1: 
            scale_factor = 0
        else:
            scale_factor = self.FOV / (camera_distance + z) * scale_mult

        px = int(x * scale_factor + offset_x)
        py = int(y * scale_factor + offset_y)

        return (px, py), scale_factor

    def _project_grid(self, point):
        x = (point.x - (self.grid_size - 1) / 2) * self.CUBE_SPACING
        y = (point.y - (self.grid_size - 1) / 2) * self.CUBE_SPACING
        z = (point.z - (self.grid_size - 1) / 2) * self.CUBE_SPACING
        return self._project_point(x, y, z)

    def _draw_grid_lattice(self, screen):
        """[  ]      (      +    )"""
        G_MAX = self.grid_size - 1
        
        # 1.       
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                for z in range(self.grid_size):
                    #         
                    if x == 0 or x == G_MAX or y == 0 or y == G_MAX or z == 0 or z == G_MAX:
                        p = Point3D(x, y, z)
                        pos, scale = self._project_grid(p)
                        radius = max(1, int(2 * scale))
                        color = self._apply_depth_shading(self.GRID_DOT_COLOR, scale)
                        pygame.draw.circle(screen, color, pos, radius)

        # 2.        
        lines = []
        lines.append((Point3D(0,0,0), Point3D(G_MAX,0,0)))
        lines.append((Point3D(0,G_MAX,0), Point3D(G_MAX,G_MAX,0)))
        lines.append((Point3D(0,0,G_MAX), Point3D(G_MAX,0,G_MAX)))
        lines.append((Point3D(0,G_MAX,G_MAX), Point3D(G_MAX,G_MAX,G_MAX)))
        lines.append((Point3D(0,0,0), Point3D(0,G_MAX,0)))
        lines.append((Point3D(G_MAX,0,0), Point3D(G_MAX,G_MAX,0)))
        lines.append((Point3D(0,0,G_MAX), Point3D(0,G_MAX,G_MAX)))
        lines.append((Point3D(G_MAX,0,G_MAX), Point3D(G_MAX,G_MAX,G_MAX)))
        lines.append((Point3D(0,0,0), Point3D(0,0,G_MAX)))
        lines.append((Point3D(G_MAX,0,0), Point3D(G_MAX,0,G_MAX)))
        lines.append((Point3D(0,G_MAX,0), Point3D(0,G_MAX,G_MAX)))
        lines.append((Point3D(G_MAX,G_MAX,0), Point3D(G_MAX,G_MAX,G_MAX)))

        for p_start, p_end in lines:
            pos1, _ = self._project_grid(p_start)
            pos2, _ = self._project_grid(p_end)
            pygame.draw.line(screen, self.GRID_FRAME_COLOR, pos1, pos2, 2)

    def _draw_snake_body_lines(self, screen):
        """[  ]        (     )"""
        if len(self.snake) < 2: return
        for i in range(len(self.snake) - 1):
            p1 = self.snake[i]
            p2 = self.snake[i+1]
            pos1, scale1 = self._project_grid(p1)
            pos2, scale2 = self._project_grid(p2)
            
            base_color = (60, 200, 60)
            final_color = self._apply_depth_shading(base_color, (scale1+scale2)/2)
            width = max(1, int(6 * ((scale1+scale2)/2)))
            pygame.draw.line(screen, final_color, pos1, pos2, width)

    def _draw_element(self, screen, point, color, base_radius=11, is_head=False, draw_projections=False):
        """[  ]      (       )"""
        pos, scale = self._project_grid(point)
        G_MAX = self.grid_size - 1
        
        #        
        if draw_projections:
            planes = [
                Point3D(0, point.y, point.z),       
                Point3D(G_MAX, point.y, point.z),   
                Point3D(point.x, 0, point.z),       
                Point3D(point.x, G_MAX, point.z),   
                Point3D(point.x, point.y, 0),       
                Point3D(point.x, point.y, G_MAX),   
            ]
            proj_color = self.PROJECTION_LINE_COLOR
            for target_p in planes:
                if target_p == point: continue
                target_pos, _ = self._project_grid(target_p)
                pygame.draw.line(screen, proj_color, pos, target_pos, 1)
                pygame.draw.circle(screen, (min(255, proj_color[0]+30), min(255, proj_color[1]+30), min(255, proj_color[2]+30)), target_pos, 3)

        radius = int(base_radius * scale)
        if radius < 3: radius = 3

        shadow_pos = (pos[0] + 3, pos[1] + 3)
        pygame.draw.circle(screen, (10, 10, 10), shadow_pos, radius)

        final_color = self._apply_depth_shading(color, scale)
        pygame.draw.circle(screen, final_color, pos, radius)

        if is_head:
            pygame.draw.circle(screen, self.WHITE, pos, radius, 2)
            pygame.draw.circle(screen, (255, 255, 255), (pos[0]-radius//3, pos[1]-radius//3), max(1, radius//4))

    def _draw_direction_guides(self, screen):
        """[  ]       """
        if self.game_over or not self.snake: return
        head = self.snake[0]
        head_pos_screen, _ = self._project_grid(head)

        directions = [
            (1, 0, 0, "R", self.RED), (-1, 0, 0, "L", self.RED),
            (0, 1, 0, "D", self.GREEN), (0, -1, 0, "U", self.GREEN),
            (0, 0, 1, "S", self.BLUE), (0, 0, -1, "W", self.BLUE)
        ]

        for dx, dy, dz, label, color in directions:
            nx, ny, nz = head.x + dx, head.y + dy, head.z + dz
            #     
            if (nx < 0 or nx >= self.grid_size or ny < 0 or ny >= self.grid_size or nz < 0 or nz >= self.grid_size):
                continue
            #     
            is_body = False
            for seg in self.snake: 
                if seg.x == nx and seg.y == ny and seg.z == nz:
                    is_body = True
                    break
            if is_body: continue
            
            hx = (head.x - (self.grid_size - 1) / 2) * self.CUBE_SPACING
            hy = (head.y - (self.grid_size - 1) / 2) * self.CUBE_SPACING
            hz = (head.z - (self.grid_size - 1) / 2) * self.CUBE_SPACING
            
            tx = hx + dx * self.CUBE_SPACING * 0.8
            ty = hy + dy * self.CUBE_SPACING * 0.8
            tz = hz + dz * self.CUBE_SPACING * 0.8
            text_pos_screen, _ = self._project_point(tx, ty, tz)
            
            pygame.draw.line(screen, color, head_pos_screen, text_pos_screen, 2)
            
            text_surf = self.font.render(label, True, self.WHITE)
            text_rect = text_surf.get_rect(center=text_pos_screen)
            pygame.draw.rect(screen, (0,0,0), text_rect.inflate(4,4))
            screen.blit(text_surf, text_rect)

    def _draw_axis_gizmo(self, screen):
        """     """
        origin = (60, self.SCREEN_HEIGHT - 60)
        axis_length = 30
        axes = [(axis_length, 0, 0, "X", self.RED), (0, axis_length, 0, "Y", self.GREEN), (0, 0, axis_length, "Z", self.BLUE)]
        for x, y, z, label, color in axes:
            end_pos, _ = self._project_point(x, y, z, offset_x=origin[0], offset_y=origin[1], scale_mult=1.0)
            pygame.draw.line(screen, color, origin, end_pos, 3)
            screen.blit(self.font.render(label, True, color), end_pos)

    def _get_z_depth(self, point):
        """[  ]           (     )"""
        p = point
        #            ，     Z-depth  
        #       ，          
        # x_w = (p.x - ...) * SPACING
        # z_w = (p.z - ...) * SPACING
        # z_rot = z_w * cos_y (   ，        z     ，  angle_x/y  )
        
        #         :
        # z = (p.z - (GRID_SIZE - 1) / 2) * CUBE_SPACING
        # cos_y = math.cos(self.angle_y)
        # z_rot = z * cos_y
        
        z = (p.z - (self.grid_size - 1) / 2) * self.CUBE_SPACING
        cos_y = math.cos(self.angle_y)
        return z * cos_y

    def _render_game_over(self, screen):
        cx, cy = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2
        bg_rect = pygame.Rect(0, 0, 320, 180)
        bg_rect.center = (cx, cy)
        
        pygame.draw.rect(screen, (20, 20, 25), bg_rect)
        pygame.draw.rect(screen, self.WHITE, bg_rect, 2)
        
        title = self.title_font.render("GAME OVER", True, self.RED)
        score = self.large_font.render(f"Final Score: {self.score}", True, self.YELLOW)
        
        screen.blit(title, title.get_rect(center=(cx, cy - 40)))
        screen.blit(score, score.get_rect(center=(cx, cy + 10)))

    #       
    @classmethod
    def get_action_profile(cls) -> GameActionProfile:
        """        """
        profile = GameActionProfile(
            game_name="3D Snake",
            game_version="2.0" # Updated version
        )

        mappings = [
            ActionMapping(key="d", action_type=ActionType.KEY_PRESS, action_name="move_right", description="     (+X  )", method_name="move_right"),
            ActionMapping(key="a", action_type=ActionType.KEY_PRESS, action_name="move_left", description="     (-X  )", method_name="move_left"),
            ActionMapping(key="w", action_type=ActionType.KEY_PRESS, action_name="move_up", description="     (-Y  )", method_name="move_up"),
            ActionMapping(key="s", action_type=ActionType.KEY_PRESS, action_name="move_down", description="     (+Y  )", method_name="move_down"),
            ActionMapping(key="q", action_type=ActionType.KEY_PRESS, action_name="move_forward", description="    /   (-Z  )", method_name="move_forward"),
            ActionMapping(key="e", action_type=ActionType.KEY_PRESS, action_name="move_backward", description="    /   (+Z  )", method_name="move_backward"),
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
    game = SnakeGameAdapter(grid_size=5, max_steps=50)

    #      
    game.init()
    print("Initial state:", game.get_state())

    #     
    game.move_right()
    # pygame.display.set_mode((800, 600))
    # game.render()
    # pygame.display.flip()
    # pygame.time.wait(2000)