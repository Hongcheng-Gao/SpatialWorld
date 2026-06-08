import pygame
from geometry import get_init_points
from constants import WIDTH, HEIGHT, MOVE_KEY_MAP, ROTATE_KEY_MAP, CW, ACW, MOVE, MOVE2LAYERS, ROTATE, OPPOSITE
from constants import SAVE_POSITION, RESET_POSITION, SAVE_KEY_MAP
from constants import FUNCTIONAL_KEY_MAP, SOLVE, NEXT_STEP, SHUFFLE
from constants import F, B, L, R, U, D
from constants import ROTATIONAL_SPEED, COLORS, EDGES, CORNERS
# from init_configs import INITIAL_CONFIGS
import os
import sys
#          Python   
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
from data.rubik_cube.scramble_data import INITIAL_CONFIGS
from utilities import RubikUtilities
from solver import RubikSolver
from copy import deepcopy
from geometry import z_orientation, xy_projection
from functools import reduce
from operator import add
from rubik import Rubik


def init_mouse_drag(points):
    dragging = False

    def handle_mouse_drag(event):
        nonlocal dragging
        if not any(pygame.mouse.get_pressed()):
            dragging = False

        if event.type == pygame.MOUSEBUTTONUP:
            dragging = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            dragging = True
            # Ignore initial relative value.
            pygame.mouse.get_rel()
        elif event.type == pygame.MOUSEMOTION and dragging:
            # Handle mouse drag HERE
            x, y = pygame.mouse.get_rel()
            for point in points:
                point.rotate_x_ip(y)
                point.rotate_y_ip(x)

    return handle_mouse_drag


def handle_save_points(points):
    saved_points = deepcopy(points)

    def save_positions():
        nonlocal saved_points
        saved_points = deepcopy(points)

    def reset_positions():
        for p, sp in zip(points, saved_points):
            p.update(sp)

    return save_positions, reset_positions


def init_handle_keys(init_move, save_positions, reset_positions):
    def handle_key_event(event):
        if event.type == pygame.KEYDOWN:
            keys = pygame.key.get_pressed()
            direction = ACW if keys[pygame.K_RSHIFT] or keys[pygame.K_LSHIFT] else CW
            if event.key in MOVE_KEY_MAP:
                face = MOVE_KEY_MAP[event.key]
                if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
                    init_move(direction, face, MOVE2LAYERS)
                else:
                    init_move(direction, face, MOVE)
            if event.key in ROTATE_KEY_MAP:
                face = ROTATE_KEY_MAP[event.key]
                init_move(direction, face, ROTATE)
            if event.key in SAVE_KEY_MAP:
                if SAVE_KEY_MAP[event.key] == SAVE_POSITION:
                    save_positions()
                elif SAVE_KEY_MAP[event.key] == RESET_POSITION:
                    reset_positions()

    return handle_key_event


def draw_surface(win, color, surf):
    if z_orientation(surf) > 0:
        pygame.draw.polygon(win, color, xy_projection(surf))
        pygame.draw.polygon(win, (128, 128, 128), xy_projection(surf), 1)


def surf_mid_point(surf):
    v = reduce(add, surf, pygame.Vector3()) / 4
    return v


def moving_points_on_move(face, centers, edges, corners):
    # The points that will be rotated when given face is rotated.
    points = []
    points.extend(centers[face])
    for edge in edges:
        if face in edge:
            for f in edge:
                points.extend(edges[edge][f])
    for corner in corners:
        if face in corner:
            for f in corner:
                points.extend(corners[corner][f])
    rotation_axis = surf_mid_point(centers[face])
    return points, rotation_axis


def moving_points_on_rotation(face, action, centers, edges, corners):
    # The points that will be rotated when given face is rotated.
    points = []
    if action == MOVE:
        points.extend(centers[face])
        for edge in edges:
            if face in edge:
                for f in edge:
                    points.extend(edges[edge][f])
        for corner in corners:
            if face in corner:
                for f in corner:
                    points.extend(corners[corner][f])
    elif action == MOVE2LAYERS:
        opp_face = OPPOSITE[face]
        for f in centers:
            if f != opp_face:
                points.extend(centers[f])
        for edge in edges:
            if opp_face not in edge:
                for f in edge:
                    points.extend(edges[edge][f])
        for corner in corners:
            if opp_face not in corner:
                for f in corner:
                    points.extend(corners[corner][f])
    elif action == ROTATE:
        for f in centers:
            points.extend(centers[f])
        for edge in edges:
            for f in edge:
                points.extend(edges[edge][f])
        for corner in corners:
            for f in corner:
                points.extend(corners[corner][f])

    rotation_axis = surf_mid_point(centers[face])
    return points, rotation_axis


def animation(rubik, centers, edges, corners):
    # Implements the animation for move.
    # Also completes the move by applying it on the rubik's after animation is complete.
    running = False
    step_angle = 5
    current_angle = 0
    rotation_points, rotation_axis = None, None
    rotation_face, rotation_direction = None, None
    rotation_action = None
    initial_points = None

    def in_progress():
        return running

    def init_move(direction, face, action):
        nonlocal rotation_points, rotation_axis, running, initial_points
        nonlocal rotation_face, rotation_direction, current_angle, rotation_action
        running = True
        rotation_action = action
        rotation_direction = direction
        rotation_face = face
        current_angle = 0
        rotation_points, rotation_axis = moving_points_on_rotation(face, action, centers, edges, corners)
        initial_points = deepcopy(rotation_points)

    def animate():
        nonlocal rotation_points, current_angle, running, initial_points
        for p in rotation_points:
            p.rotate_ip(-step_angle if rotation_direction == CW else step_angle, rotation_axis)
        current_angle = current_angle + step_angle
        if current_angle >= 90:
            for ip, rp in zip(initial_points, rotation_points):
                rp.update(ip)
            rubik.transform(rotation_direction, rotation_face, rotation_action)
            running = False

    return in_progress, init_move, animate


def draw_rubik(win, rubik, centers, edges, corners):
    surfaces = []
    for face in centers:
        surfaces.append((rubik.get_colors(face), centers[face]))
    for edge in edges:
        for face, color in zip(edge, rubik.get_colors(edge)):
            surfaces.append((color, edges[edge][face]))
    for corner in corners:
        # print(corner, rubik.get_colors(corner))
        for face, color in zip(corner, rubik.get_colors(corner)):
            surfaces.append((color, corners[corner][face]))

    # Sort the surfaces according to their average z axis
    # hinge that this will surfaces on top to be drawn later on.
    surfaces.sort(key=lambda v: surf_mid_point(v[1]).z)

    for color, surf in surfaces:
        draw_surface(win, color, surf)


def draw_orientation(win, rubik):
    w, h = WIDTH / 3.5, HEIGHT / 4
    f = lambda *v: tuple(int(i) for i in v)
    surf = pygame.Surface(f(w, h))
    surf.fill((128, 128, 128))
    rect = surf.get_rect()
    rect.bottomright = (WIDTH, HEIGHT)
    faces = {F: f(w / 5, h / 4, w / 5, h / 4), L: f(0, h / 4, w / 5, h / 4), U: f(w / 5, 0, w / 5, h / 4),
             D: f(w / 5, h / 2, w / 5, h / 4),
             R: f(2 * w / 5, h / 4, w / 5, h / 4), B: f(3 * w / 5, h / 4, w / 5, h / 4)}
    for f in faces:
        pygame.draw.rect(surf, rubik.get_colors(f), faces[f])
    win.blit(surf, rect)


def draw_key_indicator(win, points, centers):
    """
                      ，            
                
    """
    #     
    key_face_map = {
        'F': F, 'B': B, 'L': L, 'R': R, 'U': U, 'D': D
    }

    #                     
    face_positions = {}
    face_visible = {}

    for face in centers:
        center_point = surf_mid_point(centers[face])
        screen_pos = xy_projection([center_point])[0]
        face_positions[face] = screen_pos

        #        ：z_orientation > 0         
        face_visible[face] = z_orientation(centers[face]) > 0

    #           
    font = pygame.font.Font(None, 24)  #       

    for key, face in key_face_map.items():
        if face in face_positions and face_visible[face]:
            screen_pos = face_positions[face]

            #        -          
            key_text = font.render(f"{key}", True, (255, 255, 0))  #     
            text_rect = key_text.get_rect(center=(screen_pos.x, screen_pos.y))

            #        
            bg_rect = text_rect.inflate(12, 8)
            bg_surf = pygame.Surface((bg_rect.width, bg_rect.height), pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, 180))  #           
            win.blit(bg_surf, bg_rect)

            #     
            win.blit(key_text, text_rect)




def is_cube_solved(rubik):
    """
              
    """
    #               
    for face in [F, B, L, R, U, D]:
        center_color = rubik.get_colors(face)
        if center_color != COLORS[face]:
            return False

    #              （    ）
    for edge in EDGES:
        edge_colors = rubik.get_colors(edge)
        expected_colors = tuple(COLORS[f] for f in edge)
        if edge_colors != expected_colors:
            return False

    #              （    ）
    for corner in CORNERS:
        corner_colors = rubik.get_colors(corner)
        expected_colors = tuple(COLORS[f] for f in corner)
        if corner_colors != expected_colors:
            return False

    return True




def apply_initial_config(rubik, config_name):
    """
             
    """
    config = INITIAL_CONFIGS[config_name]

    if config == "random":
        #     
        RubikUtilities.shuffle(rubik, 20)
    else:
        #         
        for direction, face in config:
            rubik.move(direction, face)


def show_game_over_screen(win, moves_count, time_elapsed):
    """
            
    """
    font = pygame.font.Font(None, 48)
    small_font = pygame.font.Font(None, 24)

    #         
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 200))
    win.blit(overlay, (0, 0))

    #       
    congrats = font.render("Congratulations! Cube Solved!", True, (255, 255, 0))
    win.blit(congrats, (WIDTH//2 - congrats.get_width()//2, HEIGHT//2 - 100))

    #       
    moves_text = small_font.render(f"Moves: {moves_count}", True, (255, 255, 255))
    # time_text = small_font.render(f"Time: {time_elapsed:.1f} seconds", True, (255, 255, 255))

    win.blit(moves_text, (WIDTH//2 - moves_text.get_width()//2, HEIGHT//2 - 20))
    # win.blit(time_text, (WIDTH//2 - time_text.get_width()//2, HEIGHT//2 + 10))

    #       
    continue_text = small_font.render("Press any key to continue...", True, (200, 200, 200))
    win.blit(continue_text, (WIDTH//2 - continue_text.get_width()//2, HEIGHT//2 + 60))

    pygame.display.update()

    #       
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                waiting = False
                return True

    return True


def init_rotation_handler(points):
    """
             ，        45 
    """
    rotation_angle = 45  #     45 
    key_states = {
        pygame.K_UP: False,
        pygame.K_DOWN: False,
        pygame.K_LEFT: False,
        pygame.K_RIGHT: False,
        pygame.K_LEFTBRACKET: False,
        pygame.K_RIGHTBRACKET: False
    }

    def handle_rotation_event(event):
        """        """
        if event.type == pygame.KEYDOWN:
            if event.key in key_states:
                key_states[event.key] = True

                #       
                if event.key == pygame.K_UP:
                    for p in points:
                        p.rotate_x_ip(-rotation_angle)
                elif event.key == pygame.K_DOWN:
                    for p in points:
                        p.rotate_x_ip(rotation_angle)
                elif event.key == pygame.K_LEFT:
                    for p in points:
                        p.rotate_y_ip(-rotation_angle)
                elif event.key == pygame.K_RIGHT:
                    for p in points:
                        p.rotate_y_ip(rotation_angle)
                elif event.key == pygame.K_LEFTBRACKET:
                    for p in points:
                        p.rotate_z_ip(rotation_angle)
                elif event.key == pygame.K_RIGHTBRACKET:
                    for p in points:
                        p.rotate_z_ip(-rotation_angle)

        elif event.type == pygame.KEYUP:
            if event.key in key_states:
                key_states[event.key] = False

    return handle_rotation_event


def shuffle_generator():
    for _ in range(50):
        yield RubikUtilities.random_move()


def init_functional_keys(rubik, init_move):
    def handle_functional_keys(event):
        if event.type == pygame.KEYDOWN and event.key in FUNCTIONAL_KEY_MAP:
            keys = pygame.key.get_pressed()
            shift_pressed = keys[pygame.K_RSHIFT] or keys[pygame.K_LSHIFT]
            if not shift_pressed:
                if FUNCTIONAL_KEY_MAP[event.key] == SHUFFLE:
                    RubikUtilities.shuffle(rubik, 50)
                elif FUNCTIONAL_KEY_MAP[event.key] == NEXT_STEP:
                    for direction, face in RubikSolver.solve_next_step(rubik, D):
                        rubik.move(direction, face)
                elif FUNCTIONAL_KEY_MAP[event.key] == SOLVE:
                    for direction, face in RubikSolver.solve(rubik, D):
                        rubik.move(direction, face)
            else:
                init_function(FUNCTIONAL_KEY_MAP[event.key])

    running = False
    generator = ()

    def init_function(func):
        nonlocal running, generator
        if func == SHUFFLE:
            running = True
            generator = shuffle_generator()
        elif func == NEXT_STEP:
            running = True
            generator = RubikSolver.solve_next_step(rubik, D)
        elif func == SOLVE:
            running = True
            generator = RubikSolver.solve(rubik, D)

    def continue_function():
        nonlocal running
        assert running
        try:
            direction, face = next(generator)
            init_move(direction, face, MOVE)
        except StopIteration:
            running = False

    def in_progress():
        return running

    return handle_functional_keys, in_progress, continue_function


def mainloop(config_name="simple"):
    """
         
    config_name:        ("simple", "medium", "hard", "random")
    """
    #       
    if config_name not in INITIAL_CONFIGS:
        print(f"Warning: Invalid config name '{config_name}', using 'simple' instead")
        config_name = "test"

    pygame.init()
    clock = pygame.time.Clock()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Rubik's Cube - {config_name}")

    #     
    run = True
    game_over = False
    moves_count = 0
    start_time = pygame.time.get_ticks()

    #      
    rubik = Rubik()
    points, centers, edges, corners = get_init_points()

    #       
    apply_initial_config(rubik, config_name)

    #         
    save_positions, reset_positions = handle_save_points(points)
    handle_mouse_drag = init_mouse_drag(points)
    in_progress_animation, init_move, animate = animation(rubik, centers, edges, corners)
    handle_functional_keys, in_progress_function, continue_function = init_functional_keys(rubik, init_move)
    handle_key_event = init_handle_keys(init_move, save_positions, reset_positions)
    handle_rotation_event = init_rotation_handler(points)

    #      
    def wrapped_init_move(direction, face, action):
        nonlocal moves_count
        moves_count += 1
        init_move(direction, face, action)

    #                  
    handle_key_event = init_handle_keys(wrapped_init_move, save_positions, reset_positions)
    handle_functional_keys, in_progress_function, continue_function = init_functional_keys(rubik, wrapped_init_move)

    while run:
        clock.tick(60)

        #       
        current_time = pygame.time.get_ticks()
        time_elapsed = (current_time - start_time) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
                    (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE) or \
                    (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                run = False
            if not in_progress_animation() and not in_progress_function() and not game_over:
                handle_mouse_drag(event)
                handle_key_event(event)
                handle_functional_keys(event)
                handle_rotation_event(event)

        if in_progress_animation():
            animate()
        elif in_progress_function():
            continue_function()

        #         
        if not game_over and not in_progress_animation() and not in_progress_function():
            if is_cube_solved(rubik):
                game_over = True
                #         
                if not show_game_over_screen(win, moves_count, time_elapsed):
                    run = False
                else:
                    #       ，      
                    return mainloop()

        win.fill((128, 128, 128))
        draw_orientation(win, rubik)
        draw_rubik(win, rubik, centers, edges, corners)
        draw_key_indicator(win, points, centers)

        #       
        if not game_over:
            font = pygame.font.Font(None, 24)
            moves_text = font.render(f"Moves: {moves_count}", True, (255, 255, 255))
            # time_text = font.render(f"Time: {time_elapsed:.1f}s", True, (255, 255, 255))
            win.blit(moves_text, (10, HEIGHT - 60))
            # win.blit(time_text, (10, HEIGHT - 30))

        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    import sys

    #     
    config = "7"

    #        
    if len(sys.argv) > 1:
        config = sys.argv[1]

    print(f"Starting Rubik's Cube with config: {config}")
    print("Available configs: simple, medium, hard, random")

    mainloop(config)