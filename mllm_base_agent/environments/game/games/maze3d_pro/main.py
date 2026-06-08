import pygame
import sys
import os 
from pathlib import Path
import math
import time 

# --- 1.       ---
pygame.init()

#     
WALL = '#'
EMPTY = ' '
START = 'S'
EXIT = 'E'
STAIR = 'T' 

#     
NORTH, SOUTH, EAST, WEST = 'NORTH', 'SOUTH', 'EAST', 'WEST'

#     
COLORS = {
    'wall': (160, 85, 60),
    'wall_shadow': (80, 40, 30),
    'floor': (50, 60, 65),
    'ceiling': (15, 15, 35),
    'stair_fill': (0, 100, 120),   
    'stair_edge': (0, 255, 255),   
    'text': (255, 255, 255),
    'ui_bg': (0, 0, 0, 200)
}

SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("3D Tower Final Fix V3")

font = pygame.font.Font(None, 28)
large_font = pygame.font.Font(None, 48)

#     
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data" / "maze3d_pro"

# --- 2.      ---
def darken(color, factor):
    """      """
    return tuple(max(0, int(c * factor)) for c in color)

# --- 3.   ：     ---
def load_tower(filename):
    floors = {}
    start_pos = (1, 1, 0)
    exits = {}
    
    current_z = -1 
    current_y = 0
    
    #         
    if not os.path.exists(filename):
        print(f"Error: File not found {filename}")
        return {0: {}}, (0,0,0), {}

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            raw_line = line.rstrip('\n\r')
            if not raw_line: continue

            #        
            if raw_line.strip().startswith(('===', '---')):
                current_z += 1
                current_y = 0
                floors[current_z] = {} 
                continue
            
            if current_z == -1: 
                current_z = 0
            
            if current_z not in floors:
                floors[current_z] = {}

            for x, char in enumerate(raw_line):
                pos = (x, current_y)
                if char == START:
                    start_pos = (x, current_y, current_z)
                    floors[current_z][pos] = EMPTY
                elif char == EXIT:
                    exits[current_z] = (x, current_y)
                    floors[current_z][pos] = EMPTY 
                elif char == STAIR:
                    floors[current_z][pos] = STAIR
                elif char == WALL:
                    floors[current_z][pos] = WALL
                else:
                    floors[current_z][pos] = EMPTY
            current_y += 1

    if not floors:
        floors[0] = {}

    return floors, start_pos, exits

# --- 4.   ：3D    ---
def draw_3d_view(surface, maze, px, py, pDir, exit_pos):
    #     
    pygame.draw.rect(surface, COLORS['ceiling'], (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT//2))
    pygame.draw.rect(surface, COLORS['floor'], (0, SCREEN_HEIGHT//2, SCREEN_WIDTH, SCREEN_HEIGHT//2))

    dir_vecs = {NORTH: (0,-1,1,0), SOUTH: (0,1,-1,0), EAST: (1,0,0,1), WEST: (-1,0,0,-1)}
    dfx, dfy, dsx, dsy = dir_vecs[pDir]

    def get_cell(x, y):
        if (x, y) == exit_pos: return EXIT
        return maze.get((x, y), WALL)

    MAX_DEPTH = 12
    cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
    
    rects = []
    for d in range(MAX_DEPTH + 1):
        scale = 4.0 if d == 0 else 1.0 / d
        w = 600 * scale
        h = 400 * scale
        rects.append((w, h))

    for d in range(MAX_DEPTH, 0, -1):
        w_far, h_far = rects[d]
        w_near, h_near = rects[d-1]
        
        xf_l, xf_r = cx - w_far//2, cx + w_far//2
        xn_l, xn_r = cx - w_near//2, cx + w_near//2
        yf_t, yf_b = cy - h_far//2, cy + h_far//2
        yn_t, yn_b = cy - h_near//2, cy + h_near//2

        def check(dist_f, dist_s):
            tx, ty = px + dfx * dist_f + dsx * dist_s, py + dfy * dist_f + dsy * dist_s
            return get_cell(tx, ty)

        fog = max(0.4, 1.0 - d * 0.08)

        def draw_poly(pts, color, is_outline=True):
            c = darken(color, fog)
            pygame.draw.polygon(surface, c, pts)
            if is_outline:
                pygame.draw.polygon(surface, darken(COLORS['wall_shadow'], fog), pts, 1)

        def draw_stair_frame(pts, thickness=3):
            #           ，    ，   VLM     
            c_fill = darken(COLORS['stair_fill'], fog)
            c_edge = darken(COLORS['stair_edge'], max(fog, 0.8))
            pygame.draw.polygon(surface, c_fill, pts)
            pygame.draw.polygon(surface, c_edge, pts, thickness)
            if len(pts) == 4:
                pygame.draw.line(surface, c_edge, pts[0], pts[2], 2)
                pygame.draw.line(surface, c_edge, pts[1], pts[3], 2)

        #   
        cell_l = check(d-1, -1)
        pts_left = [(xn_l, yn_t), (xf_l, yf_t), (xf_l, yf_b), (xn_l, yn_b)]
        if cell_l == WALL:
            draw_poly(pts_left, COLORS['wall'])
        elif cell_l == STAIR:
            draw_stair_frame(pts_left)

        #   
        cell_r = check(d-1, 1)
        pts_right = [(xf_r, yf_t), (xn_r, yn_t), (xn_r, yn_b), (xf_r, yf_b)]
        if cell_r == WALL:
            draw_poly(pts_right, COLORS['wall'])
        elif cell_r == STAIR:
            draw_stair_frame(pts_right)

        #    
        cell_f = check(d, 0)
        pts_front = [(xf_l, yf_t), (xf_r, yf_t), (xf_r, yf_b), (xf_l, yf_b)]

        if cell_f == WALL:
            draw_poly(pts_front, COLORS['wall_shadow'])
        elif cell_f == EXIT:
            pygame.draw.polygon(surface, (255, 200, 0), pts_front)
            pygame.draw.polygon(surface, (255, 255, 255), pts_front, 3)
        elif cell_f == STAIR:
            draw_stair_frame(pts_front, thickness=3)

# --- 5.     ---
def main():
    tower_data, player_pos, all_exits = load_tower(r".")
    
    px, py, pz = player_pos
    pDir = NORTH
    won = False
    clock = pygame.time.Clock()

    while True:
        current_map = tower_data.get(pz, {})
        current_exit = all_exits.get(pz, (-1, -1))
        tile_under_foot = current_map.get((px, py), EMPTY)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            
            if event.type == pygame.KEYDOWN and not won:
                # ---      ---
                dx, dy = 0, 0
                turn = None
                if event.key == pygame.K_w:
                    move_map = {NORTH:(0,-1), SOUTH:(0,1), EAST:(1,0), WEST:(-1,0)}
                    dx, dy = move_map[pDir]
                elif event.key == pygame.K_s:
                    move_map = {NORTH:(0,1), SOUTH:(0,-1), EAST:(-1,0), WEST:(1,0)}
                    dx, dy = move_map[pDir]
                elif event.key == pygame.K_a: turn = 'LEFT'
                elif event.key == pygame.K_d: turn = 'RIGHT'

                if turn == 'LEFT': pDir = {NORTH:WEST, WEST:SOUTH, SOUTH:EAST, EAST:NORTH}[pDir]
                elif turn == 'RIGHT': pDir = {NORTH:EAST, EAST:SOUTH, SOUTH:WEST, WEST:NORTH}[pDir]
                
                # ---           ---
                if dx != 0 or dy != 0:
                    nx, ny = px + dx, py + dy
                    target = current_map.get((nx, ny), WALL)
                    
                    if (nx, ny) == current_exit:
                        px, py = nx, ny
                        won = True
                    elif target == EMPTY or target == STAIR:
                        px, py = nx, ny

                # ---      ---
                if tile_under_foot == STAIR:
                    if event.key == pygame.K_q:
                        if (pz + 1) in tower_data and tower_data[pz+1].get((px, py)) != WALL:
                            pz += 1
                            print(f"UP to Floor {pz+1}")
                    elif event.key == pygame.K_e:
                        if (pz - 1) in tower_data and tower_data[pz-1].get((px, py)) != WALL:
                            pz -= 1
                            print(f"DOWN to Floor {pz+1}")

        screen.fill((0, 0, 0))
        draw_3d_view(screen, current_map, px, py, pDir, current_exit)

        # UI
        overlay = pygame.Surface((SCREEN_WIDTH, 60))
        overlay.set_alpha(180)
        overlay.fill((0,0,0))
        screen.blit(overlay, (0,0))

        # [  ]     FLOOR   ，    POS   DIR
        info = f"POS: {px},{py} | DIR: {pDir}"
        screen.blit(font.render(info, True, COLORS['text']), (20, 20))

        if tile_under_foot == STAIR and not won:
            tips = []
            if (pz + 1) in tower_data: tips.append("[Q] UP")
            if (pz - 1) in tower_data: tips.append("[E] DOWN")
            if tips:
                t_surf = large_font.render(" | ".join(tips), True, (0, 255, 255))
                screen.blit(t_surf, (SCREEN_WIDTH//2 - t_surf.get_width()//2, SCREEN_HEIGHT - 80))

        # ---        ---
        if won:
            victory_bg = pygame.Surface((SCREEN_WIDTH, 100))
            victory_bg.set_alpha(200)
            victory_bg.fill((0,0,0))
            screen.blit(victory_bg, (0, SCREEN_HEIGHT//2 - 50))
            
            msg = large_font.render("MISSION COMPLETE!", True, (255, 215, 0))
            screen.blit(msg, (SCREEN_WIDTH//2 - msg.get_width()//2, SCREEN_HEIGHT//2 - 15))
            
            pygame.display.flip()
            
            pygame.time.wait(3000)
            print("Game Over: Player Escaped")
            pygame.quit()
            sys.exit()

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()