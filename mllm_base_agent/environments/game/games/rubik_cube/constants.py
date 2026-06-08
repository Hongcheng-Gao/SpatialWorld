from pygame import Vector2
import pygame

F, B = 'FRONT', 'BACK'
R, L = 'RIGHT', 'LEFT'
U, D = 'UP', 'DOWN'

CW, ACW = 'CLOCKWISE', 'ANTICLOCKWISE'
f = lambda *i: frozenset(i)

# Constants used by geometry.py
OFFSET = Vector2(250, 250)
SCALE = Vector2(100, 100)

WIDTH, HEIGHT = (500, 500)

WHITE = (255, 255, 255)
RED = (137, 18, 20)
BLUE = (13, 72, 172)
ORANGE = (255, 85, 37)
GREEN = (25, 155, 76)
YELLOW = (254, 213, 47)

COLORS = {
    F: RED,
    B: ORANGE,
    U: YELLOW,
    D: WHITE,
    R: GREEN,
    L: BLUE,
}

COLORS_MAP = {
    WHITE: 'WHITE',
    RED: 'RED',
    BLUE: 'BLUE',
    GREEN: 'GREEN',
    YELLOW: 'YELLOW',
    ORANGE: 'ORANGE',
}

FACES = {
    F: ((L, U, F), (L, D, F), (R, D, F), (R, U, F)),
    B: ((L, D, B), (L, U, B), (R, U, B), (R, D, B)),
    R: ((R, D, B), (R, U, B), (R, U, F), (R, D, F)),
    L: ((L, D, F), (L, U, F), (L, U, B), (L, D, B)),
    U: ((R, U, F), (R, U, B), (L, U, B), (L, U, F)),
    D: ((R, D, F), (L, D, F), (L, D, B), (R, D, B)),
}

CENTERS = [F, B, L, R, U, D]
EDGES = [(R, U), (R, D), (R, F), (R, B),
         (L, U), (L, D), (L, F), (L, B),
         (U, F), (U, B), (D, F), (D, B)]
CORNERS = [(R, U, F), (R, U, B), (R, D, F), (R, D, B),
           (L, U, F), (L, U, B), (L, D, F), (L, D, B)]

OPPOSITE = {
    F: B,
    B: F,
    R: L,
    L: R,
    U: D,
    D: U
}

NEIGHBORS = {
    F: (L, R, U, D),
    B: (L, R, U, D),
    L: (U, D, F, B),
    R: (U, D, F, B),
    U: (L, R, F, B),
    D: (L, R, F, B),
}

MOVE_KEY_MAP = {
    pygame.K_f: F,
    pygame.K_b: B,
    pygame.K_l: L,
    pygame.K_r: R,
    pygame.K_u: U,
    pygame.K_d: D,
}

ROTATE_KEY_MAP = {
    pygame.K_x: R,
    pygame.K_y: U,
    pygame.K_z: F,
}

SHUFFLE = 'SHUFFLE'
NEXT_STEP = 'NEXT_STEP'
SOLVE = 'SOLVE'
FUNCTIONAL_KEY_MAP = {
    pygame.K_h: SHUFFLE,
    pygame.K_j: NEXT_STEP,
    pygame.K_k: SOLVE,
}

ROTATE = 'ROTATE'
MOVE = 'MOVE'
MOVE2LAYERS = 'MOVE2LAYERS'

SAVE_POSITION = 'SAVE_POSITION'
RESET_POSITION = 'RESET_POSITION'

SAVE_KEY_MAP = {
    pygame.K_s: SAVE_POSITION,
    pygame.K_i: RESET_POSITION,
}

ALL_MOVES = [
    (CW, F), (CW, B), (CW, R), (CW, L), (CW, U), (CW, D),
    (ACW, F), (ACW, B), (ACW, R), (ACW, L), (ACW, U), (ACW, D)
]


ROTATIONAL_SPEED = 5