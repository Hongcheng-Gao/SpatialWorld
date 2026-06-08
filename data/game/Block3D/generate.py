import random

# --- A.           (    ) ---
STATIC_LEVELS = [
    {
        "name": "   -     (2 )",
        "blocks": [(2, 2, 0), (3, 2, 0)]
    },
    {
        "name": "   -    (4 )",
        "blocks": [(2, 2, 0), (2, 3, 0), (2, 2, 1), (2, 2, 2)]
    },
    {
        "name": "   -    (9 )",
        "blocks": [
            (1, 2, 0), (1, 2, 1), (1, 2, 2),
            (3, 2, 0), (3, 2, 1), (3, 2, 2),
            (1, 2, 3), (2, 2, 3), (3, 2, 3)
        ]
    },
    {
        "name": "   -     (10 )",
        "blocks": [
            (2, 2, 0), (2, 3, 0), (3, 3, 0), (3, 2, 0),
            (3, 1, 1), (2, 1, 1), (1, 1, 1), (1, 2, 1),
            (1, 3, 2), (2, 3, 2)
        ]
    },
    {
        "name": "   -       (8 )",
        "blocks": [
            (2, 2, 0), (3, 2, 0), (2, 3, 0), (3, 3, 0),
            (2, 2, 1), (3, 2, 1), (2, 3, 1), (3, 3, 1)
        ]
    },
    {
        "name": "   -     (15 )",
        "blocks": [
            (1, 1, 0), (2, 1, 0), (3, 1, 0),
            (1, 2, 0), (2, 2, 0), (3, 2, 0),
            (1, 3, 0), (2, 3, 0), (3, 3, 0),
            (2, 1, 1), (1, 2, 1), (2, 2, 1), (3, 2, 1), (2, 3, 1),
            (2, 2, 2)
        ]
    },
    {
        "name": "   -    H (10 )",
        "blocks": [
            (1, 2, 0), (1, 2, 1), (1, 2, 2), (1, 2, 3),
            (3, 2, 0), (3, 2, 1), (3, 2, 2), (3, 2, 3),
            (2, 2, 1), (2, 2, 2)
        ]
    },
    {
        "name": "   -    (16 )",
        "blocks": [
            (1, 1, 0), (2, 1, 0), (3, 1, 0),
            (1, 3, 0), (2, 3, 0), (3, 3, 0),
            (1, 2, 0), (3, 2, 0),
            (1, 1, 1), (2, 1, 1), (3, 1, 1),
            (1, 3, 1), (2, 3, 1), (3, 3, 1),
            (1, 2, 1), (3, 2, 1)
        ]
    },
    {
        "name": "   -      (8 )",
        "blocks": [
            (2, 2, 0), (2, 2, 1),
            (2, 2, 2), (2, 2, 3),
            (1, 2, 2), (3, 2, 2),
            (2, 1, 2), (2, 3, 2)
        ]
    }
]

# --- B.          ---
def generate_random_level_data(num_blocks, grid_size=6):
    """             """
    blocks = set()
    start_pos = (random.randint(1, grid_size-2), random.randint(1, grid_size-2), 0)
    blocks.add(start_pos)
    
    attempts = 0
    while len(blocks) < num_blocks and attempts < 1000:
        attempts += 1
        base = random.choice(list(blocks))
        bx, by, bz = base
        
        directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        dx, dy, dz = random.choice(directions)
        nx, ny, nz = bx + dx, by + dy, bz + dz
        
        if 0 <= nx < grid_size and 0 <= ny < grid_size and 0 <= nz < grid_size:
            blocks.add((nx, ny, nz))
            
    return sorted(list(blocks))

# --- C.     ：         ---
def export_to_file(filename="level_data.py"):
    # 1.              
    all_levels = []
    
    # 1.1       
    all_levels.extend(STATIC_LEVELS)
    
    # 1.2           (  5      )
    #        5 , 12 , 18 , 20       
    random_counts = [1,1,1,3,3,3,4,4,4,5,5] 
    for count in random_counts:
        blocks = generate_random_level_data(count)
        all_levels.append({
            "name": f"     -      ({count} )",
            "blocks": blocks
        })

    # 2. [    ]           
    # Python   sort     ，           ，          
    all_levels.sort(key=lambda x: len(x['blocks']))

    # 3.     ，      ID (1, 2, 3...)
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Auto-generated Level Data\n")
        f.write("# Sorted by block count (Easy -> Hard)\n\n")
        f.write("LEVEL_DESIGNS = {\n")

        for index, data in enumerate(all_levels):
            #    enumerate   index+1        ID ( 1  )
            new_id = str(index + 1)
            block_count = len(data["blocks"])

            f.write(f'    "{new_id}": {{\n')
            #             ，         
            f.write(f'        "name": "[Lv.{new_id}] {data["name"]} ( {block_count} )",\n')
            f.write(f'        "blocks": [\n')
            
            #         
            blocks = data["blocks"]
            line_buffer = ""
            for b in blocks:
                s = str(b) + ", "
                if len(line_buffer) + len(s) > 80:
                    f.write(f'            {line_buffer}\n')
                    line_buffer = s
                else:
                    line_buffer += s
            if line_buffer:
                f.write(f'            {line_buffer}\n')
                
            f.write(f'        ]\n')
            f.write(f'    }},\n')
            
        f.write("}\n")
    
    print(f"     {filename}!")
    print(f"    {len(all_levels)}    ，             ")
    print(f"          ：python game.py 1, python game.py 2 ...")

if __name__ == "__main__":
    export_to_file()