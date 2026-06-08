import os
import random
import matplotlib.pyplot as plt
import numpy as np

# ---    ---
OUTPUT_DIR = r"." #         
WALL = '#'
EMPTY = ' '
START = 'S'
EXIT = 'E'
STAIR = 'T'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 1.        (      )
# ==========================================

def _generate_single_floor(width, height, straightness, dead_end_removal, difficulty, 
                           entry_pos, target_pos=None, bounds=None):
    """
           
    bounds: (min_x, max_x, min_y, max_y)       （       ）
    """
    grid = [[WALL for _ in range(width)] for _ in range(height)]
    
    #          
    if bounds is None:
        min_x, max_x, min_y, max_y = 0, width, 0, height
    else:
        min_x, max_x, min_y, max_y = bounds

    sx, sy = entry_pos
    visited = set([(sx, sy)])
    
    # ---    A: DFS    ---
    stack = [(sx, sy, 0, 0)]
    #          ，    
    base_directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
    
    while stack:
        cx, cy, last_dx, last_dy = stack[-1]
        neighbors = []
        random.shuffle(base_directions) 
        
        for dx, dy in base_directions:
            nx, ny = cx + dx, cy + dy
            #     ：     bounds
            if min_x < nx < max_x - 1 and min_y < ny < max_y - 1:
                if (nx, ny) not in visited:
                    neighbors.append((nx, ny, dx, dy))
        
        if neighbors:
            #     ：      ，       
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

    # ---    B:        （  /  ）      ---
    if target_pos:
        ex, ey = target_pos
        #              DFS   （       ），        
        #         ：       ，                  
        grid[ey][ex] = EMPTY
        if (ex, ey) not in visited:
            #            
            found_conn = False
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                tx, ty = ex+dx, ey+dy
                if min_x < tx < max_x-1 and min_y < ty < max_y-1:
                    if grid[ty][tx] == EMPTY:
                        grid[ey][ex] = EMPTY #     
                        #        ，         ，          
                        found_conn = True
                        break
            #       ，      ，        
            if not found_conn:
                #           ，     
                if ex > min_x + 1: grid[ey][ex-1] = EMPTY

    # ---    C:       ---
    if dead_end_removal > 0:
        dead_ends = []
        for y in range(min_y + 1, max_y - 1):
            for x in range(min_x + 1, max_x - 1):
                if grid[y][x] == EMPTY:
                    w_count = sum(1 for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)] if grid[y+dy][x+dx] == WALL)
                    if w_count == 3: dead_ends.append((x, y))
        
        random.shuffle(dead_ends)
        remove_count = int(len(dead_ends) * dead_end_removal)
        for i in range(remove_count):
            dx, dy = dead_ends[i]
            #         
            valid_n = []
            for nx, ny in [(0,1),(0,-1),(1,0),(-1,0)]:
                tx, ty = dx+nx, dy+ny
                if min_x < tx < max_x - 1 and min_y < ty < max_y - 1:
                    if grid[ty][tx] == WALL:
                        valid_n.append((nx, ny))
            if valid_n:
                wx, wy = random.choice(valid_n)
                grid[dy+wy][dx+wx] = EMPTY

    return grid

# ==========================================
# 2. 3D          
# ==========================================

def generate_bridge_tower(width, height, straightness, removal, difficulty):
    """
            ：1 Start -> 1    -> 2    -> 2    -> 1 Exit     
    """
    #      (       ，      ，        "  "   )
    #       Floor 1      ，         ，        2     
    num_floors = 2 
    floors_grids = []

    # 1.       (  Floor 0   )
    #    split_x        ，                   
    split_x = (width // 2) 
    if split_x % 2 != 0: split_x += 1 #       
    if split_x >= width - 2: split_x = width - 3 #      

    # 2.        
    #     ：   x          
    def get_valid_pos(x_min, x_max, y_min, y_max):
        candidates = []
        for y in range(y_min, y_max, 2):
            for x in range(x_min, x_max, 2):
                candidates.append((x, y))
        return random.choice(candidates)

    # --- Floor 0     (Start Zone) ---
    #   : x [0, split_x]
    start_pos = (1, 1) 
    stair_up_pos = get_valid_pos(1, split_x, 1, height-1)
    
    #            
    while stair_up_pos == start_pos:
        stair_up_pos = get_valid_pos(1, split_x, 1, height-1)

    # --- Floor 0     (Exit Zone) ---
    #   : x [split_x, width]
    stair_down_landing_pos = get_valid_pos(split_x + 1, width-1, 1, height-1)
    exit_pos = get_valid_pos(split_x + 1, width-1, 1, height-1)
    
    while exit_pos == stair_down_landing_pos:
        exit_pos = get_valid_pos(split_x + 1, width-1, 1, height-1)

    # ==========================
    #    Floor 0 (  )
    # ==========================
    #             
    grid_floor0 = [[WALL for _ in range(width)] for _ in range(height)]
    
    # A.        
    left_sub_grid = _generate_single_floor(width, height, straightness, removal, difficulty,
                                           entry_pos=start_pos, target_pos=stair_up_pos,
                                           bounds=(0, split_x, 0, height))
    
    # B.         (   entry      )
    right_sub_grid = _generate_single_floor(width, height, straightness, removal, difficulty,
                                            entry_pos=stair_down_landing_pos, target_pos=exit_pos,
                                            bounds=(split_x, width, 0, height))
    
    # C.      Grid
    #     
    for y in range(height):
        for x in range(1, split_x): #            split_x
            grid_floor0[y][x] = left_sub_grid[y][x]
    #     
    for y in range(height):
        for x in range(split_x + 1, width - 1):
            grid_floor0[y][x] = right_sub_grid[y][x]
            
    #       
    grid_floor0[start_pos[1]][start_pos[0]] = START
    grid_floor0[stair_up_pos[1]][stair_up_pos[0]] = STAIR #    
    grid_floor0[stair_down_landing_pos[1]][stair_down_landing_pos[0]] = STAIR #     
    grid_floor0[exit_pos[1]][exit_pos[0]] = EXIT

    # ==========================
    #    Floor 1 (  /   )
    # ==========================
    #         floor0   stair_up_pos   stair_down_landing_pos
    #            ，   split_x
    
    grid_floor1 = _generate_single_floor(width, height, straightness, removal, difficulty,
                                         entry_pos=stair_up_pos, target_pos=stair_down_landing_pos,
                                         bounds=None) #    ，    
    
    #       (      ，              )
    grid_floor1[stair_up_pos[1]][stair_up_pos[0]] = STAIR #     
    grid_floor1[stair_down_landing_pos[1]][stair_down_landing_pos[0]] = STAIR #     

    return [grid_floor0, grid_floor1]

# ==========================================
# 3.        (    ，    )
# ==========================================

def save_bridge_level(index, width, height, straightness, removal, difficulty):
    #          ，       11   
    if width < 9: width = 9
    
    floors = generate_bridge_tower(width, height, straightness, removal, difficulty)
    num_floors = len(floors)

    filename_base = f"Level_Bridge_{index+1:02d}"
    txt_path = os.path.join(OUTPUT_DIR, f"{filename_base}.txt")
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        for i, grid in enumerate(floors):
            if i > 0: f.write("===\n")
            for line in grid:
                f.write("".join(line) + "\n")

    #   
    fig, axes = plt.subplots(1, num_floors, figsize=(5 * num_floors, 5))
    if num_floors == 1: axes = [axes]
    
    for i, ax in enumerate(axes):
        grid = floors[i]
        matrix = np.zeros((height, width))
        start_pt, exit_pt = None, None
        stair_pts = []
        
        for y in range(height):
            for x in range(width):
                char = grid[y][x]
                if char == WALL: matrix[y][x] = 0.0
                else: matrix[y][x] = 1.0 #  
                
                if char == START: start_pt = (x, y)
                elif char == EXIT: exit_pt = (x, y)
                elif char == STAIR: stair_pts.append((x, y))

        ax.imshow(matrix, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        
        #            (  Floor 0)
        if i == 0:
            split_x = width // 2
            if split_x % 2 != 0: split_x += 1
            #               
            ax.axvline(x=split_x, color='red', linestyle='--', alpha=0.5, label='Divider')

        if start_pt: ax.scatter(start_pt[0], start_pt[1], c='lime', s=120, edgecolors='black', label='Start', zorder=10)
        if exit_pt: ax.scatter(exit_pt[0], exit_pt[1], c='red', s=120, edgecolors='black', label='Exit', zorder=10)
        
        #            (      )
        #    ：   T   ，   T   
        if stair_pts:
            xs, ys = zip(*stair_pts)
            ax.scatter(xs, ys, c='cyan', s=80, marker='s', edgecolors='black', label='Stair', zorder=5)
            
        ax.set_title(f"Floor {i+1} {'(Bridge)' if i==1 else '(Split)'}")
        ax.axis('off') #      
        # ax.legend(loc='upper right', fontsize='small') #       ，    

    plt.suptitle(f"Bridge Level {index+1}: Must go via Floor 2", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_base}.png"), bbox_inches='tight', dpi=100)
    plt.close()

def main():
    print(f"🚀      '   '    (      )...")
    TOTAL_LEVELS = 10
    MIN_SIZE, MAX_SIZE = 7, 13 #            
    
    for i in range(TOTAL_LEVELS):
        #     
        t = i / (TOTAL_LEVELS - 1) if TOTAL_LEVELS > 1 else 0
        size = int(MIN_SIZE + t * (MAX_SIZE - MIN_SIZE))
        if size % 2 == 0: size += 1
        
        #     
        difficulty = t 
        straightness = 0.8 - t * 0.4 
        removal = 0.5 - t * 0.3 #             
        
        save_bridge_level(i, size, size, straightness, removal, difficulty)
        print(f"  [   {i+1}/{TOTAL_LEVELS}]     {size}x{size}")

    print(f"\n✅     ！      : {OUTPUT_DIR}")

if __name__ == "__main__":
    main()