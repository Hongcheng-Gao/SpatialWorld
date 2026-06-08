import os
import random
import collections
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent
WALL = '#'
EMPTY = ' '
START = 'S'
EXIT = 'E'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. Maze generation (DFS)
# ==========================================

def _generate_single_attempt(width, height, straightness, dead_end_removal, difficulty):
    grid = [[WALL for _ in range(width)] for _ in range(height)]
    
    sx, sy = 1, 1
    ex, ey = width - 2, height - 2
    
    obstacle_coords = set()

    # -------------------------------------------------
    # Pattern A: protruding obstacles
    # -------------------------------------------------
    if width >= 15:
        raw_len = int(min(width, height) * (0.05 + difficulty * 0.15))
        safe_limit = min(width, height) // 4 
        protrude_len = max(2, min(raw_len, safe_limit))

        start_top_x = 5
        start_left_y = 5
        end_bottom_x = width - 6
        end_right_y = height - 6

        if start_top_x < width - 2:
            for i in range(protrude_len): obstacle_coords.add((start_top_x, 1 + i))
        if start_left_y < height - 2:
            for i in range(protrude_len): obstacle_coords.add((1 + i, start_left_y))
        if end_bottom_x > 2:
            for i in range(protrude_len): obstacle_coords.add((end_bottom_x, (height - 2) - i))
        if end_right_y > 2:
            for i in range(protrude_len): obstacle_coords.add(((width - 2) - i, end_right_y))

    # -------------------------------------------------
    # Pattern B: sparse random obstacles
    # -------------------------------------------------
    if width > 13:
        area = width * height
        density_base = 0.005 if width < 21 else 0.01
        num_random = int(area * density_base * (1 + difficulty * 2))
        num_random = min(num_random, area // 15)

        for _ in range(num_random):
            cx = random.randrange(3, width - 3, 2)
            cy = random.randrange(3, height - 3, 2)
            if (abs(cx-sx) + abs(cy-sy) > 4) and (abs(cx-ex) + abs(cy-ey) > 4):
                obstacle_coords.add((cx, cy))

    # -------------------------------------------------
    # Pattern C: DFS carving
    # -------------------------------------------------
    grid[sy][sx] = START
    visited = set([(sx, sy)]) | obstacle_coords
    
    stack = [(sx, sy, 0, 0)]
    directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
    
    while stack:
        cx, cy, last_dx, last_dy = stack[-1]
        neighbors = []
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if 1 <= nx < width-1 and 1 <= ny < height-1:
                if (nx, ny) not in visited:
                    neighbors.append((nx, ny, dx, dy))
        
        if neighbors:
            same_dir_n = next((n for n in neighbors if n[2] == last_dx and n[3] == last_dy), None)
            if same_dir_n and random.random() < straightness:
                chosen = same_dir_n
            else:
                chosen = random.choice(neighbors)
            
            nx, ny, dx, dy = chosen
            grid[cy + dy//2][cx + dx//2] = EMPTY
            grid[ny][nx] = EMPTY
            visited.add((nx, ny))
            stack.append((nx, ny, dx, dy))
        else:
            stack.pop()

    grid[ey][ex] = EXIT
    
    # -------------------------------------------------
    #    D:     (     )
    # -------------------------------------------------
    if dead_end_removal > 0:
        dead_ends = []
        for y in range(1, height-1):
            for x in range(1, width-1):
                if grid[y][x] == EMPTY:
                    w_count = sum(1 for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)] if grid[y+dy][x+dx] == WALL)
                    if w_count == 3: dead_ends.append((x, y))
        
        random.shuffle(dead_ends)
        remove_count = int(len(dead_ends) * dead_end_removal)
        for i in range(remove_count):
            dx, dy = dead_ends[i]
            valid_n = []
            for nx, ny in [(0,1),(0,-1),(1,0),(-1,0)]:
                tx, ty = dx+nx, dy+ny
                if 1 <= tx < width-1 and 1 <= ty < height-1:
                    if grid[ty][tx] == WALL and (tx, ty) not in obstacle_coords:
                        valid_n.append((nx, ny))
            if valid_n:
                wx, wy = random.choice(valid_n)
                grid[dy+wy][dx+wx] = EMPTY

    return grid, (sx, sy), (ex, ey), obstacle_coords

# ==========================================
# 2.     
# ==========================================

def solve_bfs(grid, width, height, start, end):
    """BFS   ，            """
    queue = collections.deque([[start]])
    visited = set([start])
    while queue:
        path = queue.popleft()
        cx, cy = path[-1]
        if (cx, cy) == end: return path
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = cx+dx, cy+dy
            if 0<=nx<width and 0<=ny<height and grid[ny][nx]!=WALL and (nx,ny) not in visited:
                visited.add((nx,ny))
                new_path = list(path)
                new_path.append((nx,ny))
                queue.append(new_path)
    return []

def generate_maze_robust(width, height, straightness, removal, difficulty):
    """           """
    attempt = 0
    while True:
        attempt += 1
        grid, start, end, obstacles = _generate_single_attempt(width, height, straightness, removal, difficulty)
        if solve_bfs(grid, width, height, start, end):
            return grid, start, end, obstacles
        if attempt > 30:
            return _generate_single_attempt(width, height, straightness, removal, 0.0)

def hash_grid(grid, start, end):
    """
              
        ：        +      +      
              ，        ，        
    """
    grid_str = "".join(["".join(row) for row in grid])
    return f"{grid_str}|{start}|{end}"

# ==========================================
# 3.            
# ==========================================

def save_unique_level(index, width, height, straightness, removal, difficulty, seen_hashes):
    MAX_RETRIES = 50 #            
    
    for attempt in range(MAX_RETRIES):
        # 1.       
        grid, start, end, obstacles = generate_maze_robust(width, height, straightness, removal, difficulty)
        
        # 2.         (   5x5        )
        #             (<10)，           ，         /  
        if width == height and width < 10:
            ops = random.choice(['none', 'flip_h', 'flip_v', 'transpose'])
            
            if ops != 'none':
                grid_np = np.array([list(r) for r in grid]) #    numpy
                
                if ops == 'flip_h':
                    grid_np = np.fliplr(grid_np)
                    #     : x' = w - 1 - x
                    start = (width - 1 - start[0], start[1])
                    end = (width - 1 - end[0], end[1])
                    
                elif ops == 'flip_v':
                    grid_np = np.flipud(grid_np)
                    #     : y' = h - 1 - y
                    start = (start[0], height - 1 - start[1])
                    end = (end[0], height - 1 - end[1])
                    
                elif ops == 'transpose':
                    grid_np = grid_np.T
                    #     : x' = y, y' = x
                    start = (start[1], start[0])
                    end = (end[1], end[0])
                
                grid = grid_np.tolist() #    list
        
        # 3.     
        current_hash = hash_grid(grid, start, end)
        
        if current_hash not in seen_hashes:
            seen_hashes.add(current_hash)
            
            # ---      ---
            filename_base = f"Level_{index+1:02d}"
            with open(os.path.join(OUTPUT_DIR, f"{filename_base}.txt"), 'w', encoding='utf-8') as f:
                for line in grid: f.write("".join(line) + "\n")

            # ---    ---
            path = solve_bfs(grid, width, height, start, end)
            matrix = np.zeros((height, width))
            for y in range(height):
                for x in range(width):
                    #   ：       obstacles       ，         obstacles，         width
                    if (x, y) in obstacles and width >= 13: 
                        matrix[y][x] = 0.3 #      
                    elif grid[y][x] == WALL: 
                        matrix[y][x] = 0.0 #    
                    else: 
                        matrix[y][x] = 1.0 #    
            
            plt.figure(figsize=(8, 8))
            plt.imshow(matrix, cmap='gray', interpolation='nearest')
            
            if path:
                px, py = zip(*path)
                plt.plot(px, py, color='red', linewidth=2, alpha=0.6)
                plt.scatter(start[0], start[1], c='lime', s=100, label='Start') #     
                plt.scatter(end[0], end[1], c='orange', s=100, label='Exit')    #     

            plt.title(f"Level {index+1}: {width}x{height} | Obstacles: {len(obstacles)}")
            plt.axis('off')
            plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_base}.png"), bbox_inches='tight')
            plt.close()
            
            return True #     

    #            
    print(f"  [  ] Level {index+1} ({width}x{height}):    {MAX_RETRIES}              ，         ")
    return True #     True        ，     

# ==========================================
# 4.    
# ==========================================

def main():
    print(f"🚀            (      )...")
    TOTAL_LEVELS = 20
    MIN_SIZE, MAX_SIZE = 5, 13
    
    #           ，         
    seen_hashes = set()
    
    for i in range(TOTAL_LEVELS):
        t = i / (TOTAL_LEVELS - 1) # 0.0 -> 1.0
        
        #     
        size = int(MIN_SIZE + t * (MAX_SIZE - MIN_SIZE))
        if size % 2 == 0: size += 1 
        
        #     
        difficulty = t 
        straightness = 0.85 - t * 0.5 
        removal = 0.5 - t * 0.4 
        
        save_unique_level(i, size, size, straightness, removal, difficulty, seen_hashes)
        print(f"  [   {i+1}/{TOTAL_LEVELS}]     {size}x{size}   ")

    print(f"\n✅     ！      : {OUTPUT_DIR}")

if __name__ == "__main__":
    main()