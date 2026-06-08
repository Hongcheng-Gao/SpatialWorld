import os
import random
import collections
import matplotlib.pyplot as plt
import numpy as np

# ---    ---
OUTPUT_DIR = r"." #         
WALL = '#'
EMPTY = ' '
START = 'S'
EXIT = 'E'
STAIR = 'T'  #     

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 1.        (     )
# ==========================================

def _generate_single_floor(width, height, straightness, dead_end_removal, difficulty, 
                           entry_pos, exit_stair_pos=None):
    """
           
    entry_pos:     （     S，      T）
    exit_stair_pos:            （       None）
    """
    grid = [[WALL for _ in range(width)] for _ in range(height)]
    
    sx, sy = entry_pos
    
    obstacle_coords = set()

    # ---    A:        (     ) ---
    if width >= 15:
        raw_len = int(min(width, height) * (0.05 + difficulty * 0.15))
        safe_limit = min(width, height) // 4 
        protrude_len = max(2, min(raw_len, safe_limit))
        
        #          ，           
        potential_obstacles = []
        # ... (       ，       ) ...
        #              ，                ，
        #                 DFS          

    # ---    B:      ---
    if width > 13:
        area = width * height
        density_base = 0.005 if width < 21 else 0.01
        num_random = int(area * density_base * (1 + difficulty * 2))
        num_random = min(num_random, area // 15)

        for _ in range(num_random):
            cx = random.randrange(3, width - 3, 2)
            cy = random.randrange(3, height - 3, 2)
            
            #    ：       ，       ，        
            dist_start = abs(cx-sx) + abs(cy-sy)
            dist_exit = 999
            if exit_stair_pos:
                dist_exit = abs(cx-exit_stair_pos[0]) + abs(cy-exit_stair_pos[1])

            if dist_start > 4 and dist_exit > 4:
                obstacle_coords.add((cx, cy))

    # ---    C: DFS    ---
    #          
    #   ：     （S T）       ，         
    
    #      
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
            grid[cy + dy//2][cx + dx//2] = EMPTY #   
            grid[ny][nx] = EMPTY                 #   
            visited.add((nx, ny))
            stack.append((nx, ny, dx, dy))
        else:
            stack.pop()

    # ---    D:             ---
    # DFS     ，         
    #         exit_stair_pos (    )    ，     
    #    DFS      （            ），         
    if exit_stair_pos:
        ex, ey = exit_stair_pos
        #        ，             
        #             ，   DFS      
        #      ，    
        grid[ey][ex] = EMPTY
        #             (         )
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            if grid[ey+dy][ex+dx] == EMPTY:
                break
        else:
            #        ，     
            grid[ey][ex-1] = EMPTY

    # ---    E:       ---
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
            #            （       ，       ）
            valid_n = []
            for nx, ny in [(0,1),(0,-1),(1,0),(-1,0)]:
                tx, ty = dx+nx, dy+ny
                if 1 <= tx < width-1 and 1 <= ty < height-1:
                    if grid[ty][tx] == WALL and (tx, ty) not in obstacle_coords:
                        valid_n.append((nx, ny))
            if valid_n:
                wx, wy = random.choice(valid_n)
                grid[dy+wy][dx+wx] = EMPTY

    return grid, obstacle_coords

# ==========================================
# 2. 3D      
# ==========================================

def generate_3d_tower(width, height, straightness, removal, difficulty, num_floors=3):
    """
             
    """
    floors_data = [] #        grid
    all_obstacles = []
    
    # 1.          
    # Floor 0: Start (1,1) -> Stair_Up (Random)
    # Floor 1: Stair_Down (== Floor 0 Stair_Up) -> Stair_Up (Random)
    # ...
    # Floor Top: Stair_Down (== Floor N-1 Stair_Up) -> End (Random somewhere)
    
    #          ，      
    possible_xs = list(range(1, width-1, 2))
    possible_ys = list(range(1, height-1, 2))
    
    def get_random_pos(exclude_pos=None):
        while True:
            px = random.choice(possible_xs)
            py = random.choice(possible_ys)
            if exclude_pos and (abs(px - exclude_pos[0]) + abs(py - exclude_pos[1]) < 4):
                continue #       ，       
            return (px, py)

    #      
    connections = [] # [(start_pos, end_pos), ...]
    
    current_start = (1, 1) # 1     
    
    for i in range(num_floors):
        is_top = (i == num_floors - 1)
        
        if is_top:
            #           ，             
            #               'E'
            target_pos = None 
        else:
            #              
            target_pos = get_random_pos(exclude_pos=current_start)
            
        connections.append((current_start, target_pos))
        
        #               （      ）
        current_start = target_pos

    # 2.      
    for i in range(num_floors):
        entry, exit_stair = connections[i]
        
        #       
        grid, obstacles = _generate_single_floor(
            width, height, straightness, removal, difficulty,
            entry_pos=entry,
            exit_stair_pos=exit_stair
        )
        
        #     
        #       
        if i == 0:
            grid[entry[1]][entry[0]] = START # 1     Start
        else:
            grid[entry[1]][entry[0]] = STAIR #              

        #            
        if exit_stair:
            grid[exit_stair[1]][exit_stair[0]] = STAIR

        floors_data.append(grid)
        all_obstacles.append(obstacles)

    # 3.        (E)
    #   ：       ，               
    valid_exit_placed = False
    attempt_count = 0
    
    while not valid_exit_placed and attempt_count < 100:
        target_floor_idx = random.randint(0, num_floors - 1)
        #          
        candidates = []
        target_grid = floors_data[target_floor_idx]
        for y in range(height):
            for x in range(width):
                char = target_grid[y][x]
                if char == EMPTY:
                    #                (      )
                    entry_pos = connections[target_floor_idx][0]
                    if abs(x - entry_pos[0]) + abs(y - entry_pos[1]) > width // 3:
                        candidates.append((x, y))
        
        if candidates:
            ex, ey = random.choice(candidates)
            target_grid[ey][ex] = EXIT
            valid_exit_placed = True
            print(f"   ->        {target_floor_idx+1}   ({ex}, {ey})")
        
        attempt_count += 1

    return floors_data, all_obstacles

# ==========================================
# 3.       
# ==========================================

def save_3d_level(index, width, height, straightness, removal, difficulty):
    #     ：2   4  
    num_floors = random.choice([2, 3, 4])
    if width < 15: num_floors = 2 #        
    
    floors, obstacles = generate_3d_tower(width, height, straightness, removal, difficulty, num_floors)
    
    filename_base = f"Level_{index+1:02d}"
    txt_path = os.path.join(OUTPUT_DIR, f"{filename_base}.txt")
    
    #    TXT
    #   ：      ===   ---   
    with open(txt_path, 'w', encoding='utf-8') as f:
        for i, grid in enumerate(floors):
            if i > 0:
                f.write("===\n") #     
            for line in grid:
                f.write("".join(line) + "\n")

    #      (        )
    fig, axes = plt.subplots(1, num_floors, figsize=(4 * num_floors, 4))
    if num_floors == 1: axes = [axes]
    
    for i, ax in enumerate(axes):
        grid = floors[i]
        obs = obstacles[i]
        
        matrix = np.zeros((height, width))
        start_pt = None
        exit_pt = None
        stair_pts = []
        
        for y in range(height):
            for x in range(width):
                char = grid[y][x]
                if (x, y) in obs: matrix[y][x] = 0.3 #    
                elif char == WALL: matrix[y][x] = 0.0 #  
                else: matrix[y][x] = 1.0 #  
                
                if char == START: start_pt = (x, y)
                elif char == EXIT: exit_pt = (x, y)
                elif char == STAIR: stair_pts.append((x, y))

        ax.imshow(matrix, cmap='gray', interpolation='nearest')
        
        if start_pt: ax.scatter(start_pt[0], start_pt[1], c='lime', s=100, label='Start', zorder=5)
        if exit_pt: ax.scatter(exit_pt[0], exit_pt[1], c='orange', s=100, label='Exit', zorder=5)
        if stair_pts:
            xs, ys = zip(*stair_pts)
            ax.scatter(xs, ys, c='cyan', s=80, marker='s', label='Stair', zorder=5)
            
        ax.set_title(f"Floor {i+1}")
        ax.axis('off')
    
    plt.suptitle(f"Level {index+1}: {width}x{height} x {num_floors} Floors", fontsize=14)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_base}.png"), bbox_inches='tight')
    plt.close()

def main():
    print(f"🚀      3D      (     )...")
    TOTAL_LEVELS = 20
    MIN_SIZE, MAX_SIZE = 5,11  #            
    
    for i in range(TOTAL_LEVELS):
        t = i / (TOTAL_LEVELS - 1)
        
        size = int(MIN_SIZE + t * (MAX_SIZE - MIN_SIZE))
        if size % 2 == 0: size += 1
        
        difficulty = t 
        straightness = 0.90 - t * 0.6 
        removal = 0.6 - t * 0.5 
        
        save_3d_level(i, size, size, straightness, removal, difficulty)
        print(f"  [   {i+1}/{TOTAL_LEVELS}]     Level {i+1}")

    print(f"\n✅     ！      : {OUTPUT_DIR}")

if __name__ == "__main__":
    main()