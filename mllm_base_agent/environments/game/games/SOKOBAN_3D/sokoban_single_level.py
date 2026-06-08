#!/usr/bin/env python3
"""
   3D      -           ，         
      ：0-4 
"""

import os
import sys
import time
import random
from pathlib import Path

#        Python  
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

#       
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

#      
import Classes.Map as Map
import Classes.Score as Score

def save_frame(surface, frame_dir, frame_count):
    """     """
    try:
        filename = f"{frame_dir}/frame_{frame_count:06d}.png"
        pygame.image.save(surface, filename)
        return True
    except Exception as e:
        print(f"     : {e}")
        return False

def smart_random_strategy(hasil_build, builder):
    """      ：        """
    directions = ["up", "down", "left", "right"]

    #         
    random.shuffle(directions)

    for direction in directions:
        try:
            #     
            hasil_build.player_move(direction)
            print(f"    : {direction}")
            return True
        except Exception as e:
            #       ，         
            continue

    #        ，    
    print("         ，    ")
    hasil_build = builder.build_map()
    return False

def main(level=0):
    """
         

    Args:
        level:      (0-4)
    """

    #       
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    frame_dir = f"single_level_frames_{timestamp}"
    os.makedirs(frame_dir, exist_ok=True)

    #        ALSA  
    os.environ['SDL_AUDIODRIVER'] = 'dummy'

    # INIT PYGAME
    pygame.init()
    display = (800, 600)

    #       
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    #   OpenGL
    gluPerspective(45, display[0] / display[1], 0.1, 50.0)
    glTranslate(1,0,-5)
    glRotatef(45,1,0,0)
    glOrtho(0,800,0,800,0,1000)
    glEnable(GL_DEPTH_TEST)

    # INIT FONT
    pygame.font.init()

    # Debug Option
    enable_fps_counter = True

    # RENDER POSITION
    rotate_x = 0
    rotate_y = 0
    translate_x = 0
    translate_y = 0
    z_position = 0

    # MOUSE INPUTS
    mouse_rotate = False
    mouse_move = False

    # Set Game Mode
    game_mode = 0  #     

    # CREATE CLASSES
    builder = Map.Map_Builder()
    builder.set_mode(game_mode)

    #       
    builder.current_level = level

    print(f"     -     : {level}")
    hasil_build = builder.build_map()
    steps_history = []

    # Create Countdown Timer
    time_elapsed = 0
    clocktick = 0
    game_time = 30

    # MAIN GAME LOOP
    pygame.key.set_repeat(16,100)
    in_game = True
    clock = pygame.time.Clock()

    #     
    start_time = time.time()
    max_duration = 120  #       （ ）
    move_counter = 0
    frame_counter = 0
    stuck_counter = 0
    max_stuck_moves = 30  #           

    #       
    level_completed = False

    print("              ...")
    print(f"     : {frame_dir}")

    while in_game:
        #             
        if time.time() - start_time > max_duration:
            print("        ，    ")
            break

        clock.tick(10)  #     ，           
        clocktick += clock.get_rawtime()
        time_elapsed = clocktick // 1000

        #     （       ）
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                in_game = False
            elif event.type == KEYDOWN:
                if event.key == 27:   # ESC
                    in_game = False

        #        ，       
        if level_completed:
            frame_counter += 1
            if frame_counter >= 10:  #   10    
                print("    ，    ")
                break
            continue

        #       
        move_success = smart_random_strategy(hasil_build, builder)
        move_counter += 1

        if move_success:
            stuck_counter = 0  #        
        else:
            stuck_counter += 1

        #           ，    
        if stuck_counter >= max_stuck_moves:
            print("       ，    ")
            hasil_build = builder.build_map()
            stuck_counter = 0

        # CHECK GOAL -         
        if hasil_build.tiles[hasil_build.goals.y][hasil_build.goals.x] == 3:
            steps_history.append(hasil_build.player.steps)
            print(f"     {level}!          ")
            level_completed = True

        if time_elapsed >= game_time and game_mode == 1:
            hasil_build = "DONE"
        if hasil_build == "DONE":
            in_game = False

        if in_game:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            #     
            draw_map(hasil_build)

            # TRANSLATE OBJECT IF MOUSE MOVED
            glTranslatef(translate_x, translate_y, -z_position)
            glRotatef(rotate_y/20.,1,0,0)
            glRotatef(rotate_x/20.,0,1,0)

            # RESET ROTATE
            rotate_x = 0
            rotate_y = 0
            translate_x = 0
            translate_y = 0
            z_position = 0

            #       
            if enable_fps_counter:
                draw_text(f"FPS: {clock.get_fps():.1f}", -display[0], display[1], -5, 12)

            draw_text(f"    : {level}", -display[0], display[1] - 20, -5, 12)
            draw_text(f"    : {move_counter}", -display[0], display[1] - 40, -5, 12)
            draw_text(f"  :     ", -display[0], display[1] - 60, -5, 12)
            draw_text(f"  : {frame_counter}", -display[0], display[1] - 80, -5, 12)

            if level_completed:
                draw_text("    !", -display[0]//2, display[1]//2, -5, 32)

            if game_mode == 1:
                draw_text("Time Remaining: {}".format((game_time-time_elapsed)), -display[0], display[1]+display[1]//4, -5, 64)
                draw_text("Score : {}".format(builder.current_level), -display[0], display[1]+350, -5, 16)

            pygame.display.flip()

            #      
            glReadBuffer(GL_FRONT)
            pixels = glReadPixels(0, 0, display[0], display[1], GL_RGB, GL_UNSIGNED_BYTE)
            surface = pygame.image.fromstring(pixels, display, "RGB")
            surface = pygame.transform.flip(surface, False, True)  #   Y 

            if save_frame(surface, frame_dir, frame_counter):
                frame_counter += 1

            #       
            if frame_counter >= 500:  #     500 
                print("        ，    ")
                break
        else:
            pygame.quit()

    #     
    pygame.quit()

    print(f"\n      :")
    print(f"-      : {time.time() - start_time:.1f}  ")
    print(f"-      : {move_counter}")
    print(f"-     : {'  ' if level_completed else '   '}")
    print(f"-     : {frame_counter}")
    print(f"-    : {frame_dir}")

    #   ffmpeg       
    if frame_counter > 0:
        print(f"\n     ，   :")
        print(f"python frames_to_video.py")

    return level_completed

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
    font = pygame.font.Font ("Fonts/Roboto.ttf", size)
    textSurface = font.render(text, True, (255,255,255,255), (0,0,0,255))
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glRasterPos3d(x_pos, y_pos, z_pos)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='   3D     ')
    parser.add_argument('--level', type=int, default=0, choices=[0, 1, 2, 3, 4],
                       help='     (0-4)')

    args = parser.parse_args()

    print(f"        -     : {args.level}")
    print("      :")
    print("-    0: 15x15     ")
    print("-    1: 15x15     ")
    print("-    2: 22x11    ")
    print("-    3: 20x20     ")
    print("-    4: 25x20     ")
    print()

    main(args.level)