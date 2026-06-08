import pygame
import random
import math

# ---      ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 5       
CUBE_SPACING = 80   
FOV = 600           

# ---      ---
BG_COLOR = (20, 20, 25)          

# 1.        (   ，    )
GRID_FRAME_COLOR = (80, 80, 100)
# 2.          (     ，      )
GRID_DOT_COLOR = (50, 50, 60)
# 3.         
PROJECTION_LINE_COLOR = (60, 60, 70)

WHITE = (220, 220, 220)
RED = (255, 80, 80)           
GREEN = (80, 255, 80)        
HEAD_COLOR = (120, 255, 120) 
BLUE = (80, 80, 255)         
YELLOW = (255, 220, 0)       

class Point3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.z == other.z

    def copy(self):
        return Point3D(self.x, self.y, self.z)

class Snake3DGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        #     
        pygame.display.set_caption("3D Snake - Start with Body (Len 2)")
        self.clock = pygame.time.Clock()
        
        self.font = pygame.font.SysFont('Arial', 14, bold=True)
        self.large_font = pygame.font.SysFont('Arial', 24, bold=True)
        self.title_font = pygame.font.SysFont('Arial', 40, bold=True)

        #       
        self.angle_x = 0.5  
        self.angle_y = 0.6 
        
        self.reset_game()

    # ---        ---
    def reset_game(self):
        mid = GRID_SIZE // 2
        start_y = GRID_SIZE - 1
        start_z = mid
        
        #         2 (1  + 1 )
        target_length = 2
        self.snake = []
        
        #        (+X)，      X         
        # i=0    ，i=1    
        for i in range(target_length):
            pos = Point3D(mid - i, start_y, start_z)
            self.snake.append(pos)

        self.last_move_dir = Point3D(1, 0, 0) 
        # generate_food           
        self.food = self.generate_food()
        self.score = 0
        self.game_over = False
    # -------------------

    def generate_food(self):
        while True:
            x = random.randint(0, GRID_SIZE - 1)
            y = random.randint(0, GRID_SIZE - 1)
            z = random.randint(0, GRID_SIZE - 1)
            p = Point3D(x, y, z)
            collision = False
            for segment in self.snake:
                if p == segment:
                    collision = True
                    break
            if not collision:
                return p

    def try_move(self, dx, dy, dz):
        if self.game_over: return
        #     ：              ，   
        if (self.last_move_dir.x + dx == 0 and 
            self.last_move_dir.y + dy == 0 and 
            self.last_move_dir.z + dz == 0):
            return 

        self.last_move_dir = Point3D(dx, dy, dz)
        self.step(dx, dy, dz)

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and self.game_over: self.reset_game()

                if not self.game_over:
                    #      WASD    
                    if event.key == pygame.K_LEFT or event.key == pygame.K_a:  self.try_move(-1, 0, 0)
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d: self.try_move(1, 0, 0)
                    elif event.key == pygame.K_UP:    self.try_move(0, -1, 0) 
                    elif event.key == pygame.K_DOWN:  self.try_move(0, 1, 0)
                    # W/S       Z  
                    elif event.key == pygame.K_w:     self.try_move(0, 0, -1) 
                    elif event.key == pygame.K_s:     self.try_move(0, 0, 1)  
        return True

    def update(self, dt):
        #             ，      
        pass 

    def step(self, dx, dy, dz):
        head = self.snake[0]
        new_head = Point3D(head.x + dx, head.y + dy, head.z + dz)

        #       
        if (new_head.x < 0 or new_head.x >= GRID_SIZE or
            new_head.y < 0 or new_head.y >= GRID_SIZE or
            new_head.z < 0 or new_head.z >= GRID_SIZE):
            self.game_over = True
            return

        #        (                 )
        #   ：         ，        ，                 
        #          snake[:-1]，           
        for segment in self.snake[:-1]: 
            if new_head == segment:
                self.game_over = True
                return

        self.snake.insert(0, new_head) #         
        
        if new_head == self.food:
            #     ，     （    ）
            self.score += 1
            if len(self.snake) == GRID_SIZE ** 3:
                self.game_over = True 
            else:
                self.food = self.generate_food()
        else:
            #      ，    （      ）
            self.snake.pop()

    def project_point(self, x, y, z, offset_x=SCREEN_WIDTH/2, offset_y=SCREEN_HEIGHT/2, scale_mult=1.0):
        # 3D    
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

        camera_distance = 600 
        #         
        if camera_distance + z <= 1: scale_factor = 0 
        else: scale_factor = FOV / (camera_distance + z) * scale_mult
        
        px = int(x * scale_factor + offset_x)
        py = int(y * scale_factor + offset_y)
        
        return (px, py), scale_factor

    def project_grid(self, point):
        #                 
        x = (point.x - (GRID_SIZE - 1) / 2) * CUBE_SPACING
        y = (point.y - (GRID_SIZE - 1) / 2) * CUBE_SPACING
        z = (point.z - (GRID_SIZE - 1) / 2) * CUBE_SPACING
        return self.project_point(x, y, z)

    def apply_depth_shading(self, color, scale):
        #     （    ）      
        factor = min(1.2, max(0.4, scale * 0.8))
        r = int(min(255, color[0] * factor))
        g = int(min(255, color[1] * factor))
        b = int(min(255, color[2] * factor))
        return (r, g, b)

    def draw_grid_lattice(self):
        G_MAX = GRID_SIZE - 1
        
        # 1.       
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                for z in range(GRID_SIZE):
                    if x == 0 or x == G_MAX or y == 0 or y == G_MAX or z == 0 or z == G_MAX:
                        p = Point3D(x, y, z)
                        pos, scale = self.project_grid(p)
                        radius = max(1, int(2 * scale))
                        color = self.apply_depth_shading(GRID_DOT_COLOR, scale)
                        pygame.draw.circle(self.screen, color, pos, radius)

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
            pos1, _ = self.project_grid(p_start)
            pos2, _ = self.project_grid(p_end)
            pygame.draw.line(self.screen, GRID_FRAME_COLOR, pos1, pos2, 2)

    def draw_snake_body_lines(self):
        #       
        if len(self.snake) < 2: return
        for i in range(len(self.snake) - 1):
            p1 = self.snake[i]
            p2 = self.snake[i+1]
            pos1, scale1 = self.project_grid(p1)
            pos2, scale2 = self.project_grid(p2)
            
            base_color = (60, 200, 60)
            final_color = self.apply_depth_shading(base_color, (scale1+scale2)/2)
            width = max(1, int(6 * ((scale1+scale2)/2)))
            pygame.draw.line(self.screen, final_color, pos1, pos2, width)

    def draw_element(self, point, color, base_radius=11, is_head=False, draw_projections=False):
        #       （      ）
        pos, scale = self.project_grid(point)
        G_MAX = GRID_SIZE - 1
        
        if draw_projections:
            planes = [
                Point3D(0, point.y, point.z),       
                Point3D(G_MAX, point.y, point.z),   
                Point3D(point.x, 0, point.z),       
                Point3D(point.x, G_MAX, point.z),   
                Point3D(point.x, point.y, 0),       
                Point3D(point.x, point.y, G_MAX),   
            ]
            proj_color = PROJECTION_LINE_COLOR
            for target_p in planes:
                if target_p == point: continue
                target_pos, _ = self.project_grid(target_p)
                pygame.draw.line(self.screen, proj_color, pos, target_pos, 1)
                pygame.draw.circle(self.screen, (proj_color[0]+30, proj_color[1]+30, proj_color[2]+30), target_pos, 3)

        radius = int(base_radius * scale)
        if radius < 3: radius = 3
        
        shadow_pos = (pos[0] + 3, pos[1] + 3)
        pygame.draw.circle(self.screen, (10, 10, 10), shadow_pos, radius)

        final_color = self.apply_depth_shading(color, scale)
        pygame.draw.circle(self.screen, final_color, pos, radius)
        
        if is_head:
            pygame.draw.circle(self.screen, WHITE, pos, radius, 2)
            highlight_pos = (pos[0] - radius//3, pos[1] - radius//3)
            pygame.draw.circle(self.screen, (255, 255, 255), highlight_pos, max(1, radius//4))

    def draw_direction_guides(self):
        if self.game_over: return
        head = self.snake[0]
        head_pos_screen, _ = self.project_grid(head)
        
        directions = [
            (1, 0, 0, "R", RED), (-1, 0, 0, "L", RED),
            (0, 1, 0, "D", GREEN), (0, -1, 0, "U", GREEN),
            (0, 0, 1, "S", BLUE), (0, 0, -1, "W", BLUE)
        ]

        for dx, dy, dz, label, color in directions:
            nx, ny, nz = head.x + dx, head.y + dy, head.z + dz
            #     
            if (nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE or nz < 0 or nz >= GRID_SIZE):
                continue
            #        
            is_body = False
            #         ，    ，                   
            for seg in self.snake: 
                if seg.x == nx and seg.y == ny and seg.z == nz:
                    is_body = True
                    break
            if is_body: continue
            
            #        
            hx = (head.x - (GRID_SIZE - 1) / 2) * CUBE_SPACING
            hy = (head.y - (GRID_SIZE - 1) / 2) * CUBE_SPACING
            hz = (head.z - (GRID_SIZE - 1) / 2) * CUBE_SPACING
            tx = hx + dx * CUBE_SPACING * 0.8
            ty = hy + dy * CUBE_SPACING * 0.8
            tz = hz + dz * CUBE_SPACING * 0.8
            text_pos_screen, _ = self.project_point(tx, ty, tz)
            
            pygame.draw.line(self.screen, color, head_pos_screen, text_pos_screen, 2)
            
            text_surf = self.font.render(label, True, WHITE)
            text_rect = text_surf.get_rect(center=text_pos_screen)
            pygame.draw.rect(self.screen, (0,0,0), text_rect.inflate(4,4))
            self.screen.blit(text_surf, text_rect)

    def draw_axis_gizmo(self):
        origin = (60, SCREEN_HEIGHT - 60)
        axis_length = 30
        axes = [(axis_length, 0, 0, "X", RED), (0, axis_length, 0, "Y", GREEN), (0, 0, axis_length, "Z", BLUE)]
        for x, y, z, label, color in axes:
            end_pos, _ = self.project_point(x, y, z, offset_x=origin[0], offset_y=origin[1], scale_mult=1.0)
            pygame.draw.line(self.screen, color, origin, end_pos, 3)
            self.screen.blit(self.font.render(label, True, color), end_pos)

    def draw(self):
        self.screen.fill(BG_COLOR)
        
        # 1.     （ + ）
        self.draw_grid_lattice()
        
        # 2.            (Painter's Algorithm)
        draw_list = []
        draw_list.append({'type': 'food', 'obj': self.food, 'draw_proj': True})
        for i, segment in enumerate(self.snake):
            is_head = (i == 0)
            #           
            draw_list.append({'type': 'snake', 'obj': segment, 'is_head': is_head, 'draw_proj': is_head})

        #         
        def get_z_depth(item):
            p = item['obj']
            x = (p.x - (GRID_SIZE - 1) / 2) * CUBE_SPACING
            z = (p.z - (GRID_SIZE - 1) / 2) * CUBE_SPACING
            cos_y = math.cos(self.angle_y)
            z_rot = z * cos_y
            return z_rot

        #      
        draw_list.sort(key=get_z_depth, reverse=True)

        #       
        self.draw_snake_body_lines()

        #        
        for item in draw_list:
            if item['type'] == 'food':
                self.draw_element(item['obj'], YELLOW, 14, draw_projections=True)
            elif item['type'] == 'snake':
                color = HEAD_COLOR if item['is_head'] else GREEN
                self.draw_element(item['obj'], color, 12, item['is_head'], draw_projections=item['draw_proj'])

        #  UI  
        self.draw_direction_guides() 
        self.draw_axis_gizmo()

        score_text = self.large_font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_text, (20, 20))
        
        help_text = self.font.render("Arrows/WASD: Move X/Y | W/S (if not mapped to Y): Move Z", True, (150, 150, 150))
        self.screen.blit(help_text, (20, SCREEN_HEIGHT - 30))

        if self.game_over:
            cx, cy = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
            bg_rect = pygame.Rect(0, 0, 320, 180)
            bg_rect.center = (cx, cy)
            pygame.draw.rect(self.screen, (20, 20, 25), bg_rect)
            pygame.draw.rect(self.screen, WHITE, bg_rect, 2)
            
            title = self.title_font.render("GAME OVER", True, RED)
            score = self.large_font.render(f"Final Score: {self.score}", True, YELLOW)
            restart = self.font.render("Press 'R' to Restart", True, WHITE)
            
            self.screen.blit(title, title.get_rect(center=(cx, cy - 40)))
            self.screen.blit(score, score.get_rect(center=(cx, cy + 10)))
            self.screen.blit(restart, restart.get_rect(center=(cx, cy + 50)))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(60)
            running = self.handle_input()
            self.update(dt)
            self.draw()
        pygame.quit()

if __name__ == "__main__":
    game = Snake3DGame()
    game.run()