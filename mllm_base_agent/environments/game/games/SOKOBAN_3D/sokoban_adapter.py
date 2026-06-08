"""
3D         -  3D       MLLM    
"""
import pygame
import sys
import os
import time
from typing import Dict, Any, Optional

#          
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

#         
import Classes.Map as Map

#   OpenGL    
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *


class SokobanGame:
    """3D         """

    def __init__(self, level: int = 0, game_mode: int = 0):
        """
           3D     

        Args:
            level:      (0-4)
            game_mode:      (0=    , 1=    )
        """
        self.level = level
        self.game_mode = game_mode
        self.map_builder = None
        self.current_map = None
        self.game_won = False
        self.steps_taken = 0
        self.max_steps = 500  #       

        #    pygame    
        self.screen = None
        self.font = None
        self.small_font = None

    def init(self):
        """     """
        #    pygame（       ）
        if not pygame.get_init():
            pygame.init()

        #     OpenGL    （    ）
        self.use_opengl = False
        try:
            #       OpenGL
            from pygame.locals import DOUBLEBUF, OPENGL
            #     OpenGL  
            self.screen = pygame.display.set_mode((800, 600), DOUBLEBUF | OPENGL)
            self.use_opengl = True
            print("  OpenGL    ")
        except Exception as e:
            # OpenGL   ，        
            self.screen = pygame.Surface((800, 600))
            print(f"OpenGL   ，      : {e}")

        #      
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)

        #        
        self.map_builder = Map.Map_Builder()
        self.map_builder.set_mode(self.game_mode)
        self.map_builder.current_level = self.level

        #     
        self.current_map = self.map_builder.build_map()
        if self.current_map == "DONE":
            raise RuntimeError("Failed to build map - game mode completed")

        self.game_won = False
        self.steps_taken = 0

        return True

    def update(self):
        """      """
        #        -           
        if self.current_map and hasattr(self.current_map, 'tiles'):
            #             (tile value 3)
            goal_x, goal_y = self.current_map.goals.x, self.current_map.goals.y
            if self.current_map.tiles[goal_y][goal_x] == 3:
                self.game_won = True

        #       
        if self.steps_taken >= self.max_steps:
            self.game_won = False  #     

    def render(self, screen=None):
        """       -   OpenGL   2D  """
        #          ，          
        render_screen = screen if screen is not None else self.screen
        if not render_screen:
            return

        #     
        if self.use_opengl:
            #   OpenGL  
            self._render_opengl_view()
            #  OpenGL   ，         
            #           
        else:
            #   2D    
            render_screen.fill((0, 0, 0))
            self._render_2d_view(render_screen)

        #       ，      
        if self.game_won:
            self._render_victory(render_screen)

    def _render_opengl_view(self):
        """  OpenGL  3D  """
        if not self.current_map or not hasattr(self.current_map, 'tiles'):
            return

        #   OpenGL  
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        #          （      ）
        gluPerspective(45, 800/600, 0.1, 50.0)
        glTranslate(1, 0, -5)
        glRotatef(45, 1, 0, 0)
        glOrtho(0, 800, 0, 800, 0, 1000)
        glEnable(GL_DEPTH_TEST)

        #     
        draw_map(self.current_map)

        #     
        pygame.display.flip()

    def _render_2d_view(self, screen):
        """  2D    """
        if not self.current_map or not hasattr(self.current_map, 'tiles'):
            return

        tiles = self.current_map.tiles
        tile_size = 30
        margin = 50

        #            
        map_width = len(tiles[0]) * tile_size
        map_height = len(tiles) * tile_size
        start_x = (800 - map_width) // 2
        start_y = (600 - map_height) // 2

        #       
        for y in range(len(tiles)):
            for x in range(len(tiles[0])):
                tile_x = start_x + x * tile_size
                tile_y = start_y + y * tile_size

                #           
                tile_value = tiles[y][x]
                if tile_value == 0:  #   
                    color = (200, 200, 200)
                elif tile_value == 1:  #  
                    color = (100, 100, 100)
                elif tile_value == 2:  #   
                    color = (255, 0, 0)
                elif tile_value == 3:  #   
                    color = (255, 165, 0)  #   
                elif tile_value == 4:  #   
                    color = (0, 255, 0)
                else:
                    color = (255, 255, 255)

                #     
                pygame.draw.rect(screen, color, (tile_x, tile_y, tile_size, tile_size))
                pygame.draw.rect(screen, (50, 50, 50), (tile_x, tile_y, tile_size, tile_size), 1)

        #       
        info_text = f"Level: {self.level} | Steps: {self.steps_taken}/{self.max_steps}"
        info_surface = self.font.render(info_text, True, (255, 255, 255))
        screen.blit(info_surface, (20, 20))

        #       
        status_text = "Status: " + ("Completed!" if self.game_won else "In Progress")
        status_surface = self.font.render(status_text, True, (255, 255, 255))
        screen.blit(status_surface, (20, 50))

    def _render_victory(self, screen):
        """      """
        victory_overlay = pygame.Surface((800, 600), pygame.SRCALPHA)
        victory_overlay.fill((0, 0, 0, 200))
        screen.blit(victory_overlay, (0, 0))

        victory_text = self.font.render("VICTORY!", True, (255, 255, 0))
        screen.blit(victory_text, (400 - victory_text.get_width() // 2, 250))

        stats_text = f"Level {self.level} completed in {self.steps_taken} steps!"
        stats_surface = self.small_font.render(stats_text, True, (255, 255, 255))
        screen.blit(stats_surface, (400 - stats_surface.get_width() // 2, 300))

    def get_state(self) -> Dict[str, Any]:
        """      """
        if not self.current_map:
            return {}

        #       
        player_x = self.current_map.player.x if hasattr(self.current_map, 'player') else 0
        player_y = self.current_map.player.y if hasattr(self.current_map, 'player') else 0

        #       
        goal_x = self.current_map.goals.x if hasattr(self.current_map, 'goals') else 0
        goal_y = self.current_map.goals.y if hasattr(self.current_map, 'goals') else 0

        #       
        objective_x = self.current_map.objectives.x if hasattr(self.current_map, 'objectives') else 0
        objective_y = self.current_map.objectives.y if hasattr(self.current_map, 'objectives') else 0

        #             
        distance_to_goal = abs(objective_x - goal_x) + abs(objective_y - goal_y)

        return {
            #     
            "player_x": player_x,
            "player_y": player_y,

            #     
            "goal_x": goal_x,
            "goal_y": goal_y,

            #     
            "objective_x": objective_x,
            "objective_y": objective_y,

            #     
            "game_won": self.game_won,
            "steps_taken": self.steps_taken,
            "max_steps": self.max_steps,
            "level": self.level,

            #     
            "distance_to_goal": distance_to_goal,
            "game_over": self.game_won or self.steps_taken >= self.max_steps,
            "success": self.game_won,
            "steps_remaining": self.max_steps - self.steps_taken
        }

    def reset(self):
        """    """
        self.init()

    def move_up(self) -> bool:
        """    """
        if not self.current_map:
            return False

        try:
            self.current_map.player_move("up")
            self.steps_taken += 1
            return True
        except Exception as e:
            print(f"Move up failed: {e}")
            return False

    def move_down(self) -> bool:
        """    """
        if not self.current_map:
            return False

        try:
            self.current_map.player_move("down")
            self.steps_taken += 1
            return True
        except Exception as e:
            print(f"Move down failed: {e}")
            return False

    def move_left(self) -> bool:
        """    """
        if not self.current_map:
            return False

        try:
            self.current_map.player_move("left")
            self.steps_taken += 1
            return True
        except Exception as e:
            print(f"Move left failed: {e}")
            return False

    def move_right(self) -> bool:
        """    """
        if not self.current_map:
            return False

        try:
            self.current_map.player_move("right")
            self.steps_taken += 1
            return True
        except Exception as e:
            print(f"Move right failed: {e}")
            return False


# OpenGL     -       
def draw_map(map):
    """    """
    Cube_Size = 150
    center_x = len(map.tiles) / 2
    center_y = len(map.tiles[0])/ 2

    # Draw Plane
    for i in range(len(map.tiles)):
        for j in range(len(map.tiles[0])):
            draw_cube((j-center_x)*Cube_Size,-Cube_Size,(i-center_y)*-Cube_Size,Cube_Size,4)
            draw_plane((j-center_x)*Cube_Size,-Cube_Size,(i-center_y)*-Cube_Size,Cube_Size,0)

    # Draw Walls, Player, Objectives, and Goals
    for i in range(len(map.tiles)):
        for j in range(len(map.tiles[0])):
            if map.tiles[i][j] != 0 and map.tiles[i][j] != 4:
                draw_cube((j-center_x)*Cube_Size,0,(i-center_y)*-Cube_Size,Cube_Size,map.tiles[i][j])
                draw_plane((j-center_x)*Cube_Size,0,(i-center_y)*-Cube_Size,Cube_Size,map.tiles[i][j])
            elif map.tiles[i][j] == 4:
                draw_plane((j-center_x)*Cube_Size,0,(i-center_y)*-Cube_Size,Cube_Size,map.tiles[i][j])

def draw_cube(centerPosX, centerPosY, centerPosZ, edgeLength, mode):
    """     """
    halfSideLength = edgeLength * 0.5
    vertices = (
        # front face
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # top left
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # top right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength, # bottom right
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength, # bottom left

        # back face
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # top left
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # top right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # bottom right
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # bottom left

        # left face
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # top left
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # top right
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # bottom right
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength, # bottom left

        # right face
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # top left
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # top right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # bottom right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength, # bottom left

        # top face
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # top left
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # top right
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # bottom right
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # bottom left

        # bottom face
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength, # top left
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # top right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # bottom right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength  # bottom left
    )

    glPolygonMode(GL_FRONT_AND_BACK,GL_FILL)
    glBegin(GL_QUADS)
    if mode == 0:
        glColor3f(0,0,1)
    elif mode == 1:
        glColor3f(0,1,0)
    elif mode == 2:
        glColor3f(1,0,0)
    elif mode == 3:
        glColor3f(1,0,1)
    elif mode == 4:
        glColor3f(0.5,0.5,0.5)

    for x in range(24):
        glVertex3f(vertices[x*3],vertices[x*3 + 1],vertices[x * 3 + 2])
    glColor3f(0,0,0)
    glEnd()

def draw_plane(centerPosX, centerPosY, centerPosZ, edgeLength, mode):
    """    """
    halfSideLength = edgeLength * 0.5
    vertices = (
        # front face
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # top left
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # top right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength, # bottom right
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength, # bottom left

        # back face
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # top left
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # top right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # bottom right
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # bottom left

        # left face
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # top left
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # top right
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # bottom right
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength, # bottom left

        # right face
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # top left
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # top right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # bottom right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength, # bottom left

        # top face
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # top left
        centerPosX - halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # top right
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ - halfSideLength, # bottom right
        centerPosX + halfSideLength, centerPosY + halfSideLength, centerPosZ + halfSideLength, # bottom left

        # bottom face
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength, # top left
        centerPosX - halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # top right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ - halfSideLength, # bottom right
        centerPosX + halfSideLength, centerPosY - halfSideLength, centerPosZ + halfSideLength  # bottom left
    )

    glPolygonMode(GL_FRONT_AND_BACK,GL_LINE)
    glLineWidth(2)
    if mode == 1:
        glColor3f(0,0,0)
    elif mode == 0:
        glColor3f(1,1,1)
    elif mode == 4:
        glLineWidth(5)
        glColor3f(0,1,1)
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3,GL_FLOAT,0,vertices)
    glDrawArrays(GL_QUADS,0,24)
    glDisableClientState(GL_VERTEX_ARRAY)


def draw_text(text, x_pos, y_pos, z_pos, size):
    """    """
    #           ，              
    #           
    pass


#          
if __name__ == "__main__":
    game = SokobanGame(level=0)
    # Note: init() is called automatically by the Pygame input source
    # Do not call it here to avoid double initialization

    #     
    print("Initial state:", game.get_state())

    game.move_right()
    print("After moving right:", game.get_state())

    game.move_down()
    print("After moving down:", game.get_state())