import pygame
import sys
import argparse
import math
import os

#          Python   
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from data.Block3D.level_data import LEVEL_DESIGNS

# --- 1.      ---
# LEVEL_DESIGNS = {
#     "0": {
#         "name": "   -     (X   )",
#         "blocks": [(2, 2, 0), (3, 2, 0)] 
#     },
#     "1": {
#         "name": "   -   ",
#         "blocks": [(2, 2, 0), (2, 3, 0), (2, 2, 1), (2, 2, 2)]
#     },
#     "2": {
#         "name": "   -   ",
#         "blocks": [
#             (1, 2, 0), (1, 2, 1), (1, 2, 2), 
#             (3, 2, 0), (3, 2, 1), (3, 2, 2), 
#             (1, 2, 3), (2, 2, 3), (3, 2, 3)
#         ]
#     },
#     "3": {
#         "name": "   -    ",
#         "blocks": [
#             (2, 2, 0), (2, 3, 0), (3, 3, 0), (3, 2, 0), 
#             (3, 1, 1), (2, 1, 1), (1, 1, 1), (1, 2, 1), 
#             (1, 3, 2), (2, 3, 2)
#         ]
#     }
# }

parser = argparse.ArgumentParser()
parser.add_argument('level', nargs='?', default='0', choices=list(LEVEL_DESIGNS.keys()))
args = parser.parse_args()
current_level_data = LEVEL_DESIGNS[args.level]

# --- 2.     ---
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 1100, 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption(f"3D Block Builder [Asymmetric View] - {current_level_data['name']}")

#     
WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
GRAY = (80, 80, 80)
GREEN = (50, 200, 50) 
BLUE = (60, 120, 220) 
RED = (220, 60, 60)   
HIGHLIGHT = (100, 200, 250)
GRID_COLOR = (70, 70, 75) 
EYE_COLOR = (255, 255, 0)

#       
SHADOW_COLOR = (30, 30, 35)       #       
RIM_LIGHT_COLOR = (255, 255, 220) #        (  )
GROUND_PLANE_COLOR = (45, 45, 50) #       

#        
COLOR_X = RED
COLOR_Y = GREEN
COLOR_Z = BLUE

BLOCK_SIZE = 40
GRID_SIZE = 6

# [  ]      
OFFSET_X = 640 
OFFSET_Y = 280

font = pygame.font.SysFont('Arial', 14)
bold_font = pygame.font.SysFont('Arial', 18, bold=True)
large_font = pygame.font.SysFont('Arial', 40, bold=True)
debug_font = pygame.font.SysFont('Consolas', 14, bold=True)

class Game:
    def __init__(self, level_data):
        self.blocks = {} 
        self.cursor = [2, 2, 0]
        self.target_blocks = {coord: True for coord in level_data["blocks"]}
        self.won = False

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and not self.won:
                if event.key == pygame.K_LEFT:   self.cursor[0] = max(0, self.cursor[0] - 1)
                elif event.key == pygame.K_RIGHT: self.cursor[0] = min(GRID_SIZE - 1, self.cursor[0] + 1)
                elif event.key == pygame.K_UP:    self.cursor[1] = max(0, self.cursor[1] - 1)
                elif event.key == pygame.K_DOWN:  self.cursor[1] = min(GRID_SIZE - 1, self.cursor[1] + 1)
                elif event.key == pygame.K_w:     self.cursor[2] = min(GRID_SIZE - 1, self.cursor[2] + 1)
                elif event.key == pygame.K_s:     self.cursor[2] = max(0, self.cursor[2] - 1)
                elif event.key == pygame.K_SPACE:
                    pos = tuple(self.cursor)
                    if pos in self.blocks: del self.blocks[pos]
                    else: self.blocks[pos] = True
                    if self.blocks == self.target_blocks: self.won = True

    def iso_project(self, x, y, z):
        # 1.     
        scale_x = 1.0
        scale_y = 0.85 

        # 2.     
        slope = 0.5

        screen_x = int(OFFSET_X + (x * scale_x - y * scale_y) * BLOCK_SIZE)
        screen_y = int(OFFSET_Y + (x * scale_x + y * scale_y) * (BLOCK_SIZE * slope) - (z * BLOCK_SIZE))
        
        return screen_x, screen_y

    def draw_drop_shadow(self, x, y):
        p1 = self.iso_project(x, y, 0.05) 
        p2 = self.iso_project(x+1, y, 0.05)
        p3 = self.iso_project(x+1, y+1, 0.05)
        p4 = self.iso_project(x, y+1, 0.05)
        pygame.draw.polygon(screen, SHADOW_COLOR, [p1, p2, p3, p4])

    def draw_cube(self, x, y, z, color, outline=False):
        def get_pt(dx, dy, dz):
            return self.iso_project(x + dx, y + dy, z + dz)

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
                pygame.draw.polygon(screen, HIGHLIGHT, pts, thickness)
        else:
            c_top = color
            c_right = (max(0,color[0]-40), max(0,color[1]-40), max(0,color[2]-40))
            c_left = (max(0,color[0]-80), max(0,color[1]-80), max(0,color[2]-80))
            
            pygame.draw.polygon(screen, c_top, pts_top)
            pygame.draw.polygon(screen, c_right, pts_right)
            pygame.draw.polygon(screen, c_left, pts_left)
            
            for pts in [pts_top, pts_right, pts_left]: 
                pygame.draw.polygon(screen, BLACK, pts, 1)

            pygame.draw.line(screen, RIM_LIGHT_COLOR, p_top_3, p_top_0, 2)
            pygame.draw.line(screen, RIM_LIGHT_COLOR, p_top_0, p_top_1, 2)
            pygame.draw.line(screen, RIM_LIGHT_COLOR, p_top_2, p_bot_2, 2)
            pygame.draw.line(screen, RIM_LIGHT_COLOR, p_top_1, p_top_2, 1)

    def draw_dashed_line(self, start_pos, end_pos, color=EYE_COLOR):
        x1, y1 = start_pos
        x2, y2 = end_pos
        dl = 10 
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0: return
        angle = math.atan2(y2 - y1, x2 - x1)
        segments = int(length / dl)
        for i in range(0, segments, 2):
            start = (x1 + math.cos(angle) * i * dl, y1 + math.sin(angle) * i * dl)
            end = (x1 + math.cos(angle) * (i + 1) * dl, y1 + math.sin(angle) * (i + 1) * dl)
            pygame.draw.line(screen, color, start, end, 2)

    def draw_eye_icon(self, pos, text):
        x, y = pos
        pygame.draw.circle(screen, EYE_COLOR, (int(x), int(y)), 8, 2)
        pygame.draw.circle(screen, EYE_COLOR, (int(x), int(y)), 3)
        screen.blit(debug_font.render(text, True, EYE_COLOR), (x + 15, y - 10))

    def draw_projection_lines(self):
        # [      ] 
        #                       “  ” 
        #           X   Y   ，        (0-6)      
        #        Z   ，        
        
        #       (       ，   )
        focus_x, focus_y, focus_z = -0.8, -0.5, 0.1
        
        #         (         )
        dist = 8.0

        # --- TOP VIEW (   ) ---
        #     ：     
        #     ：     
        #   ：        ，   Z=3.0    
        eye_top_start = self.iso_project(focus_x, focus_y, focus_z + dist-2)
        eye_top_end   = self.iso_project(focus_x, focus_y, focus_z) 
        self.draw_dashed_line(eye_top_start, eye_top_end)
        self.draw_eye_icon(eye_top_start, "TOP")

        # --- FRONT VIEW (    -     UI  YZ  ) ---
        #     ：  X       
        #     ：     
        #   ：              
        eye_front_start = self.iso_project(focus_x + dist, focus_y, focus_z)
        eye_front_end   = self.iso_project(focus_x, focus_y, focus_z)
        self.draw_dashed_line(eye_front_start, eye_front_end)
        self.draw_eye_icon(eye_front_start, "FRONT (along X)")

        # --- SIDE VIEW (    -     UI  XZ  ) ---
        #     ：  Y       
        #     ：     
        #   ：              
        #           Y   “  ”，     
        eye_side_start = self.iso_project(focus_x, focus_y + dist, focus_z)
        eye_side_end   = self.iso_project(focus_x, focus_y, focus_z)
        self.draw_dashed_line(eye_side_start, eye_side_end)
        self.draw_eye_icon(eye_side_start, "SIDE (along Y)")
        
        # [    ]           ，      
        ix, iy = eye_front_end
        pygame.draw.circle(screen, EYE_COLOR, (ix, iy), 4)
        pygame.draw.circle(screen, RED, (ix, iy), 2)

    def draw(self):
        screen.fill(BLACK)
        
        # --- 1.    UI    ---
        view_cfg = [
            ("TOP (XY)",   lambda x,y,z: ((GRID_SIZE-1)-y, x), "<- Green(Y)", "Red(X) v", "5", "0"),
            ("FRONT (YZ)", lambda x,y,z: ((GRID_SIZE-1)-y, (GRID_SIZE-1)-z), "<- Green(Y)", "Blue(Z) ^", "5", "0"),
            ("SIDE (XZ)",  lambda x,y,z: (x, (GRID_SIZE-1)-z), "Red(X) ->", "Blue(Z) ^", "0", "0") 
        ]
        
        pad, vs, y_off = 40, 15, 30
        for title, mapping, xlabel, ylabel, origin_x_label, origin_y_label in view_cfg:
            color = WHITE
            if "SIDE" in title: color = EYE_COLOR
            if "FRONT" in title: color = HIGHLIGHT
            
            screen.blit(bold_font.render(title, True, color), (pad, y_off - 20))
            pygame.draw.rect(screen, GRID_COLOR, (pad, y_off, GRID_SIZE*vs, GRID_SIZE*vs), 1)
            
            screen.blit(font.render(origin_x_label, True, GRAY), (pad-10, y_off))
            end_label = "0" if origin_x_label == "5" else "5"
            screen.blit(font.render(end_label, True, GRAY), (pad + GRID_SIZE*vs + 2, y_off))
            
            screen.blit(font.render(xlabel, True, GRAY), (pad + GRID_SIZE*vs//2 - 20, y_off + GRID_SIZE*vs + 5)) 
            screen.blit(font.render(ylabel, True, GRAY), (pad + 5, y_off + GRID_SIZE*vs//2))

            for (x,y,z) in self.target_blocks:
                u, v = mapping(x,y,z)
                pygame.draw.rect(screen, GRAY, (pad+u*vs, y_off+v*vs, vs-1, vs-1))
            y_off += GRID_SIZE*vs + 60
        
        pygame.draw.line(screen, WHITE, (220, 20), (220, SCREEN_HEIGHT-20), 2)

        # --- 2.    3D      ---
        
        # A.       
        ground_pts = [
            self.iso_project(0, 0, 0),
            self.iso_project(GRID_SIZE, 0, 0),
            self.iso_project(GRID_SIZE, GRID_SIZE, 0),
            self.iso_project(0, GRID_SIZE, 0)
        ]
        pygame.draw.polygon(screen, GROUND_PLANE_COLOR, ground_pts)

        # B.      
        for x in range(GRID_SIZE + 1):
            p1 = self.iso_project(x, 0, 0)
            p2 = self.iso_project(x, GRID_SIZE, 0)
            pygame.draw.line(screen, GRID_COLOR, p1, p2, 1)
        for y in range(GRID_SIZE + 1):
            p1 = self.iso_project(0, y, 0)
            p2 = self.iso_project(GRID_SIZE, y, 0)
            pygame.draw.line(screen, GRID_COLOR, p1, p2, 1)

        # C.       
        active_columns = set()
        for (bx, by, bz) in self.blocks:
            if bz >= 0: active_columns.add((bx, by))
            
        cx, cy, cz = self.cursor
        if cz > 0: active_columns.add((cx, cy))

        for (sx, sy) in active_columns:
            self.draw_drop_shadow(sx, sy)

        # D.          
        origin = self.iso_project(0, 0, 0)
        pygame.draw.line(screen, COLOR_X, origin, self.iso_project(7,0,0), 3)
        pygame.draw.line(screen, COLOR_Y, origin, self.iso_project(0,7,0), 3)
        pygame.draw.line(screen, COLOR_Z, origin, self.iso_project(0,0,7), 3)
        screen.blit(bold_font.render("X", True, COLOR_X), self.iso_project(7.2,0,0))
        screen.blit(bold_font.render("Y", True, COLOR_Y), self.iso_project(0,7.2,0))
        
        self.draw_projection_lines()

        # E.       
        draw_list = [{'c': k, 't': 'b'} for k in self.blocks] + [{'c': tuple(self.cursor), 't': 'c'}]
        draw_list.sort(key=lambda k: sum(k['c']))
        
        for item in draw_list:
            self.draw_cube(*item['c'], BLUE if item['t']=='b' else HIGHLIGHT, outline=(item['t']=='c'))

        # F. HUD   
        screen.blit(debug_font.render(f"Cursor: {tuple(self.cursor)}", True, WHITE), (SCREEN_WIDTH - 200, SCREEN_HEIGHT - 40))

        if self.won: screen.blit(large_font.render("VICTORY!", True, GREEN), (500, 50))
        pygame.display.flip()

game = Game(current_level_data)
clock = pygame.time.Clock()
while True:
    game.handle_input()
    game.draw()
    clock.tick(30)