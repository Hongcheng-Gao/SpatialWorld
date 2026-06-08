"""
Maze 3D - Pygame Version (High Contrast Edition)
   maze3d_pygame_v3.py      
            ，   v3     
"""

import pygame
import sys
import os
import math

# ---     Pygame ---
pygame.init()

# ---      ---
WALL = '#'
EMPTY = ' '
START = 'S'
EXIT = 'E'
NORTH = 'NORTH'
SOUTH = 'SOUTH'
EAST = 'EAST'
WEST = 'WEST'

# ==========================================
#          (High Contrast & Clarity)
# ==========================================

#   ：    (  /  )，          
WALL_COLORS = {
    'base': (160, 85, 60),          #       
    'shadow': (80, 40, 30),         #      
    'highlight': (200, 140, 100),   #      
    'outline': (40, 20, 10)         #        (  )
}

#   ：    (   )，        
FLOOR_COLORS = {
    'base': (50, 60, 65),           #      
    'tile1': (70, 80, 85),          #       
    'tile2': (40, 50, 55),          #       
    'grid': (90, 100, 110)          #        ，     
}

#    ：    (   )，    
CEILING_COLORS = {
    'base': (15, 15, 35),           #        
    'beam': (30, 30, 60)            #           
}

#     ：  /  ，    
EXIT_COLORS = {
    'glow': (255, 215, 0),          #     
    'core': (255, 140, 0),          #     
    'ambient': (100, 80, 0)         #    
}

# UI   ：    
UI_COLORS = {
    'bg': (20, 20, 20, 220),        #        
    'text': (255, 255, 255),        #     
    'accent': (0, 255, 255),        #     
    'compass': (255, 50, 50)        #     
}

# ==========================================

#     
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

#      （       ， v3    ）
VIEW_WIDTH = 700
VIEW_HEIGHT = 500

#     （           ）
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Maze 3D - High Contrast Edition")

#   
font = pygame.font.Font(None, 28)
small_font = pygame.font.Font(None, 22)
large_font = pygame.font.Font(None, 64)

#        
animation_time = 0

def load_maze(filename):
    """      """
    maze = {}
    px = py = exitx = exity = None
    width = height = 0

    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            height = len(lines)
            width = len(lines[0].rstrip()) if lines else 0

            for y, line in enumerate(lines):
                line = line.rstrip()
                for x, char in enumerate(line):
                    if char in (WALL, EMPTY):
                        maze[(x, y)] = char
                    elif char == START:
                        px, py = x, y
                        maze[(x, y)] = EMPTY
                    elif char == EXIT:
                        exitx, exity = x, y
                        maze[(x, y)] = EMPTY
    except Exception as e:
        print(f"Error loading maze: {e}")
        return None, None, None, None, None, 0, 0

    if px is None or py is None:
        print("No start position found in maze")
        return None, None, None, None, None, 0, 0

    if exitx is None or exity is None:
        print("No exit found in maze")
        return None, None, None, None, None, 0, 0

    return maze, px, py, exitx, exity, width, height

def darken_color(color, factor):
    """     """
    return tuple(max(0, int(c * factor)) for c in color)

def blend_colors(c1, c2, ratio):
    """      """
    return tuple(max(0, min(255, int(c1[i] * (1 - ratio) + c2[i] * ratio))) for i in range(3))

def draw_gradient_poly(surface, points, color_top, color_bottom):
    """       """
    if len(points) < 3:
        return

    #       
    y_values = [p[1] for p in points]
    y_min, y_max = min(y_values), max(y_values)
    height = y_max - y_min

    if height == 0:
        pygame.draw.polygon(surface, color_top, points)
        return

    #         
    for y in range(int(y_min), int(y_max)):
        ratio = (y - y_min) / height
        color = blend_colors(color_top, color_bottom, ratio)

        #    y           
        intersections = []
        n = len(points)
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]

            if (y1 <= y <= y2) or (y2 <= y <= y1):
                if y2 != y1:
                    x = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
                    intersections.append(x)

        if len(intersections) >= 2:
            intersections.sort()
            for i in range(0, len(intersections), 2):
                if i + 1 < len(intersections):
                    x1, x2 = intersections[i], intersections[i + 1]
                    pygame.draw.line(surface, color, (x1, y), (x2, y))

# ==========================================
# v3       
# ==========================================


def draw_floor_perspective(surface, rect):
    horizon_y = rect.centery
    floor_height = rect.height // 2
    
    near_gray = 80   
    far_gray = 20    
    

    for y_offset in range(floor_height):

       
        prog = y_offset / floor_height
        
        current_v = far_gray + (near_gray - far_gray) * (prog ** 0.7)
        color = (int(current_v), int(current_v), int(current_v))
        
        current_y = horizon_y + y_offset
        pygame.draw.line(surface, color, (0, current_y), (rect.width, current_y))

def draw_ceiling_perspective(surface, rect):
    """     """
    horizon_y = rect.centery
    pygame.draw.rect(surface, CEILING_COLORS['base'], (0, 0, rect.width, horizon_y))

    #      
    for i in range(8):
        prog = (i / 8) ** 1.5
        y = int(horizon_y * (1 - prog))
        brightness = 0.3 + 0.4 * prog
        color = darken_color(CEILING_COLORS['beam'], brightness)
        height = max(1, int(8 * prog))
        pygame.draw.rect(surface, color, (0, y, rect.width, height))


def draw_wall_segment(surface, points, distance, is_exit=False):

    fog_factor = max(0.45, 1.0 - (distance * 0.07))

    if is_exit:
        pulse = (math.sin(animation_time * 0.1) + 1) / 2
        base_color = blend_colors(EXIT_COLORS['core'], EXIT_COLORS['glow'], pulse * 0.5)
        pygame.draw.polygon(surface, base_color, points)
    else:

        near_color = WALL_COLORS['base']
        far_color = darken_color(WALL_COLORS['shadow'], fog_factor)

        xs = [p[0] for p in points]
        x_min, x_max = int(min(xs)), int(max(xs))
        cx = SCREEN_WIDTH // 2  

        is_front_wall = len(set([int(x) for x in xs])) <= 2
        
        if is_front_wall:
            pygame.draw.polygon(surface, far_color, points)
        else:
            draw_x_min = max(0, x_min)
            draw_x_max = min(SCREEN_WIDTH, x_max)
            
            for x in range(draw_x_min, draw_x_max + 1):

                ratio = 1.0 - (abs(x - cx) / cx)
                current_color = blend_colors(near_color, far_color, max(0, min(0.5, ratio)))

                y_intersects = []
                for i in range(len(points)):
                    p1, p2 = points[i], points[(i + 1) % len(points)]
                    if (p1[0] <= x <= p2[0]) or (p2[0] <= x <= p1[0]):
                        if p1[0] != p2[0]:
                            y = p1[1] + (p2[1] - p1[1]) * (x - p1[0]) / (p2[0] - p1[0])
                            y_intersects.append(y)
                
                if len(y_intersects) >= 2:
                    y_top, y_bottom = min(y_intersects), max(y_intersects)
                    pygame.draw.line(surface, current_color, (x, y_top), (x, y_bottom))

    outline_color = darken_color(WALL_COLORS['outline'], fog_factor)
    pygame.draw.polygon(surface, outline_color, points, 1)

    if not is_exit and distance < 4.0:
        brick_color = darken_color(WALL_COLORS['shadow'], fog_factor * 0.8)
        pygame.draw.lines(surface, brick_color, True, points, 1)
           

def draw_3d_view(screen, maze, px, py, pDir, exitx, exity):
    view_rect = pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT)
    draw_ceiling_perspective(screen, view_rect)
    draw_floor_perspective(screen, view_rect)

    cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
    dir_map = {NORTH: (0,-1,1,0), SOUTH: (0,1,-1,0), EAST: (1,0,0,1), WEST: (-1,0,0,-1)}
    dfx, dfy, dsx, dsy = dir_map[pDir]

    def get_cell(x, y):
        if (x, y) == (exitx, exity): return EXIT
        return maze.get((x, y), WALL)

    MAX_DEPTH = 10
    
    coords = []
    for d in range(MAX_DEPTH + 1):
        if d == 0:
            scale = 4.0  
        else:
            scale = 1.0 / d
        coords.append((600 * scale, 400 * scale))

    for d in range(MAX_DEPTH, 0, -1):
        w_f, h_f = coords[d]
        w_n, h_n = coords[d-1]

        xf_l, xf_r = int(cx - w_f//2), int(cx + w_f//2)
        yf_t, yf_b = int(cy - h_f//2), int(cy + h_f//2)
        
        xn_l, xn_r = int(cx - w_n//2), int(cx + w_n//2)
        yn_t, yn_b = int(cy - h_n//2), int(cy + h_n//2)

        def check(dist_f, dist_s):
            target_x = px + dfx * dist_f + dsx * dist_s
            target_y = py + dfy * dist_f + dsy * dist_s
            return get_cell(target_x, target_y)
        

        if check(d-1, 1) == EMPTY:
            # if check(d, 2) == WALL:
                off_n_r = xn_r + (xn_r - cx)
                off_f_r = xf_r + (xf_r - cx)
                pts_rr = [(off_f_r, yf_t), (off_n_r, yn_t), (off_n_r, yn_b), (off_f_r, yf_b)]
                draw_wall_segment(screen, pts_rr, d + 0.5, False)

        if check(d-1, -1) == EMPTY:
            # if check(d, -2) == WALL:
                off_n_l = xn_l - (cx - xn_l)
                off_f_l = xf_l - (cx - xf_l)
                pts_ll = [(off_n_l, yn_t), (off_f_l, yf_t), (off_f_l, yf_b), (off_n_l, yn_b)]
                draw_wall_segment(screen, pts_ll, d + 0.5, False)

        pos_f = (px + dfx * d, py + dfy * d)
        pos_l_side = (px + dfx * (d-1) - dsx, py + dfy * (d-1) - dsy)
        pos_r_side = (px + dfx * (d-1) + dsx, py + dfy * (d-1) + dsy)

        cell_f = get_cell(*pos_f)
        cell_l = get_cell(*pos_l_side)
        cell_r = get_cell(*pos_r_side)
        
        if cell_l == WALL or cell_l == EXIT:
            pts_l = [(xn_l, yn_t), (xf_l, yf_t), (xf_l, yf_b), (xn_l, yn_b)]
            draw_wall_segment(screen, pts_l, d, cell_l == EXIT)
        else:
            pos_l_deep = (px + dfx * d - dsx, py + dfy * d - dsy)
            if get_cell(*pos_l_deep) == WALL:
                pts_l_rect = [(xn_l, yf_t), (xf_l, yf_t), (xf_l, yf_b), (xn_l, yf_b)]
                draw_wall_segment(screen, pts_l_rect, d + 0.1, False)

        if cell_r == WALL or cell_r == EXIT:
            pts_r = [(xf_r, yf_t), (xn_r, yn_t), (xn_r, yn_b), (xf_r, yf_b)]
            draw_wall_segment(screen, pts_r, d, cell_r == EXIT)
        else:
            pos_r_deep = (px + dfx * d + dsx, py + dfy * d + dsy)
            if get_cell(*pos_r_deep) == WALL:
                pts_r_rect = [(xf_r, yf_t), (xn_r, yf_t), (xn_r, yf_b), (xf_r, yf_b)]
                draw_wall_segment(screen, pts_r_rect, d + 0.1, False)

        if cell_f == WALL or cell_f == EXIT:
            pts_f = [(xf_l, yf_t), (xf_r, yf_t), (xf_r, yf_b), (xf_l, yf_b)]
            draw_wall_segment(screen, pts_f, d, cell_f == EXIT)


def draw_ui(screen, px, py, pDir):
    """      """
    #      
    bar_height = 40
    bar_surf = pygame.Surface((SCREEN_WIDTH, bar_height), pygame.SRCALPHA)
    bar_surf.fill(UI_COLORS['bg'])
    screen.blit(bar_surf, (0, 0))

    #      
    bottom_height = 30
    bottom_surf = pygame.Surface((SCREEN_WIDTH, bottom_height), pygame.SRCALPHA)
    bottom_surf.fill(UI_COLORS['bg'])
    screen.blit(bottom_surf, (0, SCREEN_HEIGHT - bottom_height))

    #     
    pos_text = font.render(f"POS: ({px}, {py})", True, UI_COLORS['accent'])
    screen.blit(pos_text, (20, 10))

    #     
    dir_text = font.render(f"DIR: {pDir}", True, UI_COLORS['text'])
    screen.blit(dir_text, (SCREEN_WIDTH - 150, 10))

    #    
    compass_center = (SCREEN_WIDTH // 2, bar_height // 2)
    radius = 15
    pygame.draw.circle(screen, UI_COLORS['text'], compass_center, radius, 2)

    #      
    if pDir == NORTH:
        tip = (compass_center[0], compass_center[1] - radius + 2)
        left = (compass_center[0] - 8, compass_center[1] + 5)
        right = (compass_center[0] + 8, compass_center[1] + 5)
    elif pDir == SOUTH:
        tip = (compass_center[0], compass_center[1] + radius - 2)
        left = (compass_center[0] - 8, compass_center[1] - 5)
        right = (compass_center[0] + 8, compass_center[1] - 5)
    elif pDir == EAST:
        tip = (compass_center[0] + radius - 2, compass_center[1])
        left = (compass_center[0] - 5, compass_center[1] - 8)
        right = (compass_center[0] - 5, compass_center[1] + 8)
    elif pDir == WEST:
        tip = (compass_center[0] - radius + 2, compass_center[1])
        left = (compass_center[0] + 5, compass_center[1] - 8)
        right = (compass_center[0] + 5, compass_center[1] + 8)

    pygame.draw.polygon(screen, UI_COLORS['compass'], [tip, left, right])

    #     
    n_text = small_font.render("N", True, UI_COLORS['accent'])
    screen.blit(n_text, (compass_center[0] - 5, compass_center[1] - radius - 15))

# ==========================================
#        （     ）
# ==========================================

#      
draw_gradient_polygon = draw_gradient_poly
draw_floor_with_perspective = draw_floor_perspective
draw_ceiling_with_depth = draw_ceiling_perspective
draw_modern_ui = draw_ui

#   draw_brick_texture draw_textured_wall，           
def draw_brick_texture(screen, points, base_color, is_exit=False):
    """    ：      """
    #   draw_wall_segment     
    distance = 1.0  #     
    # base_color     ，        
    draw_wall_segment(screen, points, distance, is_exit)

def draw_textured_wall(screen, points, distance, is_exit=False):
    """    ：      """
    draw_wall_segment(screen, points, distance, is_exit)

# ==========================================
#        
# ==========================================

def create_sample_maze():
    """      """
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
    os.makedirs('games/maze3d', exist_ok=True)

    with open('games/maze3d/maze_sample.txt', 'w') as f:
        f.write(sample_maze)

    return 'games/maze3d/maze_sample.txt'

def main():
    """    -     """
    maze_file = create_sample_maze()

    result = load_maze(maze_file)
    if result[0] is None:
        print("Failed to load maze")
        return

    maze, px, py, exitx, exity, _, _ = result  # width height   
    pDir = NORTH

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_w:
                    #     
                    new_x, new_y = px, py
                    if pDir == NORTH:
                        new_y -= 1
                    elif pDir == SOUTH:
                        new_y += 1
                    elif pDir == EAST:
                        new_x += 1
                    elif pDir == WEST:
                        new_x -= 1

                    if maze.get((new_x, new_y), WALL) == EMPTY or (new_x, new_y) == (exitx, exity):
                        px, py = new_x, new_y
                elif event.key == pygame.K_a:
                    #   
                    pDir = {NORTH: WEST, WEST: SOUTH, SOUTH: EAST, EAST: NORTH}[pDir]
                elif event.key == pygame.K_d:
                    #   
                    pDir = {NORTH: EAST, EAST: SOUTH, SOUTH: WEST, WEST: NORTH}[pDir]

        #       
        global animation_time
        animation_time += 1

        #   
        screen.fill((5, 10, 15))
        draw_3d_view(screen, maze, px, py, pDir, exitx, exity)
        draw_ui(screen, px, py, pDir)

        #       
        if (px, py) == (exitx, exity):
            victory_font = pygame.font.Font(None, 72)
            victory_text = victory_font.render("VICTORY!", True, EXIT_COLORS['glow'])
            screen.blit(victory_text, (SCREEN_WIDTH // 2 - victory_text.get_width() // 2,
                                     SCREEN_HEIGHT // 2 - 36))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()