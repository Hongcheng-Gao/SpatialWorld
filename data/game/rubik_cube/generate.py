import random
import os

def generate_data_and_save(num_samples=20, filename="scramble_data.py"):
    """
               ->       ->        
    """
    DIRECTIONS = ["CLOCKWISE", "ANTICLOCKWISE"]
    FACES = ["FRONT", "RIGHT", "UP", "LEFT", "DOWN", "BACK"]
    
    # 1.         
    raw_sequences = []
    for _ in range(num_samples):
        #      1   4
        steps = random.randint(1, 4)
        sequence = []
        for _ in range(steps):
            #       ，    
            face = random.choice(FACES)
            direction = random.choice(DIRECTIONS)
            sequence.append((direction, face))
        raw_sequences.append(sequence)
    
    # 2.     （  ）      
    #         ，"1"        ，         
    raw_sequences.sort(key=lambda x: len(x))

    # 3.       
    file_content = '"""\n       (      ，       )\n"""\n\n'
    file_content += 'INITIAL_CONFIGS = {\n'

    for i, moves in enumerate(raw_sequences):
        #      "1", "2", "3"... ( 1  )
        file_content += f'    "{i + 1}": [\n'
        for move in moves:
            file_content += f'        {repr(move)},\n'
        file_content += '    ],\n'

    file_content += '}\n'

    # 4.     
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(file_content)
        print(f"✅     ！")
        print(f"📄    : {filename}")
        print(f"📊    : {num_samples}   (   1->4    )")
    except IOError as e:
        print(f"❌     : {e}")

if __name__ == "__main__":
    #             
    generate_data_and_save(num_samples=20)