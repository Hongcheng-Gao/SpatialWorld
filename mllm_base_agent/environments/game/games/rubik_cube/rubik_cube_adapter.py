"""
       
        MLLM   
"""
import pygame
import sys
import os
from typing import Dict, Any, List, Optional

#           
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#          Python   
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
from data.rubik_cube.scramble_data import INITIAL_CONFIGS
#           
try:
    #         （        ）
    from .main import is_cube_solved
    from .geometry import get_init_points
    from .constants import WIDTH, HEIGHT, MOVE_KEY_MAP, ROTATE_KEY_MAP, CW, ACW, MOVE, MOVE2LAYERS, ROTATE
    from .constants import F, B, L, R, U, D, COLORS
    from .utilities import RubikUtilities
    from .solver import RubikSolver
    from .rubik import Rubik
    from .main import draw_orientation, draw_key_indicator
except ImportError:
    #         ，         
    try:
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)

        from main import is_cube_solved
        from geometry import get_init_points
        from constants import WIDTH, HEIGHT, MOVE_KEY_MAP, ROTATE_KEY_MAP, CW, ACW, MOVE, MOVE2LAYERS, ROTATE
        from constants import F, B, L, R, U, D, COLORS
        from utilities import RubikUtilities
        from solver import RubikSolver
        from rubik import Rubik
        from main import draw_orientation, draw_key_indicator
    except ImportError as e:
        print(f"Warning: Failed to import Rubik's Cube components: {e}")
        #              
        WIDTH, HEIGHT = 500, 500
        F, B, L, R, U, D = 'FRONT', 'BACK', 'LEFT', 'RIGHT', 'UP', 'DOWN'
        CW, ACW = 'CLOCKWISE', 'ANTICLOCKWISE'
        MOVE, ROTATE = 'MOVE', 'ROTATE'
        COLORS = {}

from copy import deepcopy


class RubikCubeGame:
    """        """

    def __init__(self, config_name: str = "simple"):
        """
               

        Args:
            config_name:        ("simple", "medium", "hard", "random")
        """
        self.config_name = config_name
        self.screen = None
        self.rubik = None
        self.points = None
        self.centers = None
        self.edges = None
        self.corners = None

        #     
        self.F = F
        self.B = B
        self.L = L
        self.R = R
        self.U = U
        self.D = D
        self.CW = CW
        self.ACW = ACW
        self.MOVE = MOVE
        self.ROTATE = ROTATE

        #     
        self.moves_count = 0
        self.game_over = False
        self.start_time = 0

        #     
        self.animation_in_progress = False
        self.current_angle = 0
        self.rotation_points = None
        self.rotation_axis = None
        self.rotation_face = None
        self.rotation_direction = None
        self.rotation_action = None
        self.initial_points = None

    def init(self):
        """     """
        try:
            #    pygame
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Rubik's Cube - MLLM Framework")

            #      
            self.rubik = Rubik()
            self.points, self.centers, self.edges, self.corners = get_init_points()

            #       
            self._apply_initial_config(self.config_name)

            #        
            self.moves_count = 0
            self.game_over = False
            self.start_time = pygame.time.get_ticks()

            return True

        except Exception as e:
            print(f"Failed to initialize Rubik's Cube: {e}")
            return False

    def update(self):
        """      """
        try:
            #     
            if self.animation_in_progress:
                self._animate()

            #         
            if not self.animation_in_progress and not self.game_over:
                if is_cube_solved(self.rubik):
                    self.game_over = True

        except Exception as e:
            print(f"Error in update: {e}")

    def render(self, screen=None):
        """      """
        try:
            if screen is None:
                screen = self.screen

            #     
            screen.fill((128, 128, 128))

            #     
            if self.animation_in_progress:
                self._animate()

            #     
            self._draw_rubik(screen)

            #        
            self._draw_orientation(screen)

            #        
            self._draw_key_indicator(screen)

            #       
            self._draw_game_info(screen)

            #       
            pygame.display.flip()

        except Exception as e:
            print(f"Error in render: {e}")

    def get_state(self) -> Dict[str, Any]:
        """      """
        current_time = pygame.time.get_ticks()
        time_elapsed = (current_time - self.start_time) / 1000.0

        return {
            "moves_count": self.moves_count,
            "time_elapsed": time_elapsed,
            "game_over": self.game_over,
            "cube_solved": is_cube_solved(self.rubik),
            "animation_in_progress": self.animation_in_progress,
            "config_name": self.config_name
        }

    def reset(self):
        """    """
        try:
            #     
            self.rubik = Rubik()
            self.points, self.centers, self.edges, self.corners = get_init_points()

            #         
            self._apply_initial_config(self.config_name)

            #       
            self.moves_count = 0
            self.game_over = False
            self.start_time = pygame.time.get_ticks()

            return True

        except Exception as e:
            print(f"Failed to reset Rubik's Cube: {e}")
            return False

    def move_forward(self):
        """     -          ，          """
        return True

    def turn_left(self):
        """     -          ，          """
        return True

    def turn_right(self):
        """     -          ，          """
        return True

    def rotate_face(self, face: str, direction: str = CW):
        """        """
        if self.animation_in_progress:
            return False

        if face not in [self.F, self.B, self.L, self.R, self.U, self.D]:
            return False

        if direction not in [self.CW, self.ACW]:
            return False

        try:
            self._init_move(direction, face, self.MOVE)
            self.moves_count += 1
            return True

        except Exception as e:
            print(f"Failed to rotate face: {e}")
            return False

    def rotate_cube(self, axis: str, direction: str = CW):
        """        

             45     ，          
                 45     ，        

        Args:
            axis:     ("RIGHT", "UP", "FRONT")
            direction:      ("CLOCKWISE", "ANTICLOCKWISE")
        """
        if self.animation_in_progress:
            return False

        if axis not in [self.F, self.U, self.R]:
            return False

        if direction not in [self.CW, self.ACW]:
            return False

        try:
            # 45      -        45 ，       
            rotation_angle = 45 if direction == self.CW else -45

            #      
            all_points = []
            for f in self.centers:
                all_points.extend(self.centers[f])
            for edge in self.edges:
                for f in edge:
                    all_points.extend(self.edges[edge][f])
            for corner in self.corners:
                for f in corner:
                    all_points.extend(self.corners[corner][f])

            #         
            rotation_axis = self._surf_mid_point(self.centers[axis])
            for p in all_points:
                p.rotate_ip(rotation_angle, rotation_axis)

            return True

        except Exception as e:
            print(f"Failed to rotate cube view: {e}")
            return False

    def _apply_initial_config(self, config_name: str):
        """      """
        config = INITIAL_CONFIGS.get(config_name, "simple")

        if config == "random":
            #     
            RubikUtilities.shuffle(self.rubik, 20)
        else:
            #         
            for direction, face in config:
                self.rubik.move(direction, face)

    def _init_move(self, direction, face, action):
        """     """
        self.animation_in_progress = True
        self.rotation_action = action
        self.rotation_direction = direction
        self.rotation_face = face
        self.current_angle = 0

        #         
        self.rotation_points, self.rotation_axis = self._moving_points_on_rotation(face, action)
        self.initial_points = deepcopy(self.rotation_points)

    def _moving_points_on_rotation(self, face, action):
        """        """
        points = []

        if action == self.MOVE:
            points.extend(self.centers[face])
            for edge in self.edges:
                if face in edge:
                    for f in edge:
                        points.extend(self.edges[edge][f])
            for corner in self.corners:
                if face in corner:
                    for f in corner:
                        points.extend(self.corners[corner][f])
        elif action == self.ROTATE:
            for f in self.centers:
                points.extend(self.centers[f])
            for edge in self.edges:
                for f in edge:
                    points.extend(self.edges[edge][f])
            for corner in self.corners:
                for f in corner:
                    points.extend(self.corners[corner][f])

        rotation_axis = self._surf_mid_point(self.centers[face])
        return points, rotation_axis

    def _surf_mid_point(self, surf):
        """      """
        from functools import reduce
        from operator import add
        v = reduce(add, surf, pygame.Vector3()) / 4
        return v

    def _animate(self):
        """    """
        step_angle = 5

        for p in self.rotation_points:
            p.rotate_ip(-step_angle if self.rotation_direction == self.CW else step_angle, self.rotation_axis)

        self.current_angle += step_angle

        #               
        if self.rotation_action == self.MOVE:
            #    ：90 
            target_angle = 90
        else:  # self.ROTATE
            #     ：45 
            target_angle = 45

        if self.current_angle >= target_angle:
            #     ，    
            for ip, rp in zip(self.initial_points, self.rotation_points):
                rp.update(ip)

            #       ，         ，        
            if self.rotation_action == self.MOVE:
                self.rubik.transform(self.rotation_direction, self.rotation_face, self.rotation_action)

            self.animation_in_progress = False

    def _draw_rubik(self, screen):
        """    """
        surfaces = []
        for face in self.centers:
            surfaces.append((self.rubik.get_colors(face), self.centers[face]))
        for edge in self.edges:
            for face, color in zip(edge, self.rubik.get_colors(edge)):
                surfaces.append((color, self.edges[edge][face]))
        for corner in self.corners:
            for face, color in zip(corner, self.rubik.get_colors(corner)):
                surfaces.append((color, self.corners[corner][face]))

        #   z    
        surfaces.sort(key=lambda v: self._surf_mid_point(v[1]).z)

        for color, surf in surfaces:
            self._draw_surface(screen, color, surf)

    def _draw_surface(self, screen, color, surf):
        """     """
        try:
            from geometry import z_orientation, xy_projection
            if z_orientation(surf) > 0:
                pygame.draw.polygon(screen, color, xy_projection(surf))
                pygame.draw.polygon(screen, (128, 128, 128), xy_projection(surf), 1)
        except ImportError:
            #       ，      
            pygame.draw.polygon(screen, color, [(100, 100), (200, 100), (200, 200), (100, 200)])

    def _draw_orientation(self, screen):
        """       """
        try:
            draw_orientation(screen, self.rubik)
        except:
            pass  #        ，  

    def _draw_key_indicator(self, screen):
        """       """
        try:
            draw_key_indicator(screen, self.points, self.centers)
        except:
            pass  #        ，  

    def _draw_game_info(self, screen):
        """      """
        font = pygame.font.Font(None, 24)

        #       
        moves_text = font.render(f"Moves: {self.moves_count}", True, (255, 255, 255))
        screen.blit(moves_text, (10, HEIGHT - 60))

        #      --     ，   
        # current_time = pygame.time.get_ticks()
        # time_elapsed = (current_time - self.start_time) / 1000.0
        # time_text = font.render(f"Time: {time_elapsed:.1f}s", True, (255, 255, 255))
        # screen.blit(time_text, (10, HEIGHT - 30))

        #       
        if self.game_over:
            status_text = font.render("Solved!", True, (0, 255, 0))
            screen.blit(status_text, (WIDTH - 100, HEIGHT - 30))

    def close(self):
        """    """
        try:
            pygame.quit()
        except:
            pass


if __name__ == "__main__":
    #      
    game = RubikCubeGame("simple")
    if game.init():
        print("Rubik's Cube adapter initialized successfully")

        #     
        game.render()
        print("Rendering test completed")

        #       
        state = game.get_state()
        print(f"Game state: {state}")

        game.close()
    else:
        print("Failed to initialize Rubik's Cube adapter")