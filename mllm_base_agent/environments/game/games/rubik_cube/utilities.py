from constants import ALL_MOVES
from constants import CENTERS, EDGES, CORNERS
from constants import D, CW, ACW
from constants import NEIGHBORS, OPPOSITE
from random import choice


class RubikUtilities:
    @staticmethod
    def shuffle(rubik, steps):
        for _ in range(steps):
            direction, face = RubikUtilities.random_move()
            rubik.move(direction, face)

    @staticmethod
    def random_move():
        return choice(ALL_MOVES)

    @staticmethod
    def is_solved(rubik):
        color_map = {f: rubik.get_colors(f) for f in CENTERS}
        print(color_map)
        for corner in CORNERS:
            if tuple(color_map[f] for f in corner) == rubik.get_colors(corner):
                continue
            print('OUT OF PLACE CORNER:', corner, 'COLOR MAP:', color_map)
            return False
        for edge in EDGES:
            if tuple(color_map[f] for f in edge) == rubik.get_colors(edge):
                continue
            print('OUT OF PLACE EDGE:', edge, 'COLOR MAP:', color_map)
            return False
        return True

    @staticmethod
    def is_bottom_cross_solved(rubik, bottom):
        bottom_color = rubik.get_colors(positions=bottom)
        bottom_edges = list(filter(lambda _edge: bottom_color in _edge.colors,
                                   (rubik.get_edge(positions=positions) for positions in EDGES)))
        for edge in bottom_edges:
            if not edge.colors == tuple(rubik.get_colors(f) for f in edge.positions):
                return False
        return True

    @staticmethod
    def is_bottom_layer_solved(rubik, bottom):
        if not RubikUtilities.is_bottom_cross_solved(rubik, bottom):
            return False
        bottom_color = rubik.get_colors(positions=bottom)
        bottom_corners = list(filter(lambda _corner: bottom_color in _corner.colors,
                                   (rubik.get_corner(positions=positions) for positions in CORNERS)))
        for corner in bottom_corners:
            if not corner.colors == tuple(rubik.get_colors(f) for f in corner.positions):
                return False
        return True

    @staticmethod
    def is_middle_layer_solved(rubik, bottom):
        if not RubikUtilities.is_bottom_layer_solved(rubik, bottom):
            return False
        bottom_color = rubik.get_colors(bottom)
        top_color = rubik.get_colors(OPPOSITE[bottom])
        for edge in (rubik.get_edge(positions=positions) for positions in EDGES):
            if bottom_color not in edge.colors and top_color not in edge.colors:
                if not edge.colors == tuple(rubik.get_colors(f) for f in edge.positions):
                    return False
        return True

    @staticmethod
    def is_top_cross_solved(rubik, bottom):
        if not RubikUtilities.is_middle_layer_solved(rubik, bottom):
            return False
        top_side = OPPOSITE[bottom]
        top_color = rubik.get_colors(top_side)
        for edge in (rubik.get_edge(positions=positions) for positions in EDGES):
            if top_color in edge.colors:
                if not edge.colors[edge.positions.index(top_side)] == top_color:
                    return False

        return True

    @staticmethod
    def is_top_edges_solved(rubik, bottom):
        if not RubikUtilities.is_top_cross_solved(rubik, bottom):
            return False
        top_color = rubik.get_colors(OPPOSITE[bottom])
        for edge in (rubik.get_edge(positions=positions) for positions in EDGES):
            if top_color in edge.colors:
                if not edge.colors == tuple(map(rubik.get_colors, edge.positions)):
                    return False
        return True

    @staticmethod
    def is_positioned_top_corners(rubik, bottom):
        if not RubikUtilities.is_top_edges_solved(rubik, bottom):
            return False
        top_color = rubik.get_colors(OPPOSITE[bottom])
        for corner in (rubik.get_corner(positions=positions) for positions in CORNERS):
            if top_color in corner.colors:
                if not set(corner.colors) == set(map(rubik.get_colors, corner.positions)):
                    return False
        return True

    @staticmethod
    def is_oriented_top_corners(rubik, bottom):
        # This is same as asking if rubik's cube is solved.
        # Use the is_solved method as it is more efficient.
        if not RubikUtilities.is_top_edges_solved(rubik, bottom):
            return False
        top_color = rubik.get_colors(OPPOSITE[bottom])
        for corner in (rubik.get_corner(positions=positions) for positions in CORNERS):
            if top_color in corner.colors:
                if not corner.colors == tuple(map(rubik.get_colors, corner.positions)):
                    return False
        return True


class RubikSolver:
    @staticmethod
    def make_bottom_cross(rubik, base=D):
        # Makes a cross at the 'base' side.
        # Assumed it is the down-side for naming variables with respect to it
        up_side = OPPOSITE[base]
        nbr_c = [rubik.get_center(position=f) for f in NEIGHBORS[base]]
        # Get the edges that make up the cross.
        cross_edges = [rubik.get_edge(colors=(rubik.get_colors(positions=base), c.color)) for c in nbr_c]
        for index, edge in enumerate(cross_edges):
            if edge.colors == tuple(map(rubik.get_colors, edge.positions)):
                # if edge already in place, move on to the next edge.
                continue

            # STEP 1: Bring the cross edges to top
            if base in edge.positions:
                # if it is no the bottom, rotate twice the corresponding side to bring in on top.
                face = edge.positions[0 if edge.positions[1] == base else 1]
                rubik.move(CW, face, 2)

            if up_side not in edge.positions:
                # if it is in the middle layer, bring it on top without affecting the bottom pieces
                # as they may already be aligned.
                top_rotated = False
                face = edge.positions[0]
                for _ in range(4):
                    rubik.move(CW, face)
                    if up_side in edge.positions and not top_rotated:
                        rubik.move(CW, up_side)
                        top_rotated = True

            # STEP 2: Move the edge right above where it needs to be by rotating the top.
            while nbr_c[index].position not in edge.positions:
                rubik.move(CW, up_side)

            # STEP 3: Move it to the right place.
            # Ensure the edge has right orientation after moving.
            if (nbr_c[index].color == edge.colors[0] and nbr_c[index].position == edge.positions[0]) or \
                    (nbr_c[index].color == edge.colors[1] and nbr_c[index].position == edge.positions[1]):
                # If right orientation, simply rotating twice is enough.
                rubik.move(CW, nbr_c[index].position, 2)
            else:
                # Otherwise, do the following.
                rubik.move(CW, up_side)
                face = edge.positions[0 if edge.positions[1] == up_side else 1]
                rubik.move(CW, face)
                rubik.move(ACW, nbr_c[index].position)
                rubik.move(ACW, face)

        # Post Condition: Bottom cross must be done.
        assert RubikUtilities.is_bottom_cross_solved(rubik, base)

    @staticmethod
    def make_bottom_corners(rubik, base=D):
        corners = [rubik.get_corner(colors=tuple(map(rubik.get_colors, c))) for c in CORNERS if base in c]
        for corner in corners:
            # First move the corner to top later.
            if base in corner.positions:
                top_rotated = False
                return_direction = None
                face = corner.positions[0 if corner.positions[1] == base else 1]  # Any face would do
                for index in range(4):
                    rubik.move(CW, face)
                    if OPPOSITE[base] in corner.positions and not top_rotated:
                        if index == 0:
                            rubik.move(CW, OPPOSITE[base])
                            return_direction = ACW
                            top_rotated = True
                        if index == 2:
                            rubik.move(ACW, OPPOSITE[base])
                            return_direction = CW
                            top_rotated = True
                rubik.move(return_direction, OPPOSITE[base])
            # Move it to the right place on top
            while set(corner.colors) != set(
                    rubik.get_colors(p if p != OPPOSITE[base] else base) for p in corner.positions):
                rubik.move(CW, OPPOSITE[base])
            # Now put it in place.
            face = None
            while corner.colors != tuple(map(rubik.get_colors, corner.positions)):
                positions = corner.positions
                rubik.move(CW, OPPOSITE[base])
                if face is None:
                    face = next(p for p in positions if p not in corner.positions)
                rubik.move(CW, face)
                rubik.move(ACW, OPPOSITE[base])
                rubik.move(ACW, face)
        for corner in corners:
            assert corner.colors == tuple(map(rubik.get_colors, corner.positions))

    @staticmethod
    def solve_middle_layer(rubik, base=D):
        # Algorithm definitions
        def left_algorithm(left, up, front):
            # Algorithm: U' L' U L U F U' F'
            rubik.move(ACW, up)
            rubik.move(ACW, left)
            rubik.move(CW, up)
            rubik.move(CW, left)
            rubik.move(CW, up)
            rubik.move(CW, front)
            rubik.move(ACW, up)
            rubik.move(ACW, front)

        def right_algorithm(right, up, front):
            # Algorithm: U R U' R' U' F' U F
            rubik.move(CW, up)
            rubik.move(CW, right)
            rubik.move(ACW, up)
            rubik.move(ACW, right)
            rubik.move(ACW, up)
            rubik.move(ACW, front)
            rubik.move(CW, up)
            rubik.move(CW, front)

        up_side = OPPOSITE[base]
        # The edges which need to be placed in order, without impacting the lower layers
        edges = []
        for e in EDGES:
            edge = rubik.get_edge(positions=e)
            if rubik.get_center(position=base).color not in edge.colors and \
                    rubik.get_center(position=up_side).color not in edge.colors:
                edges.append(edge)

        for index, edge in enumerate(edges):
            # Make sure that the edge is on top.
            # If not push any other available edge to occupy its place
            if up_side not in edge.positions:
                # may be possible it is already in place.
                if edge.colors == tuple(rubik.get_colors(f) for f in edge.positions):
                    continue
                # get the edge out
                # any side of the edge
                some_face = edge.positions[0]
                other_face = edge.positions[1]
                rubik.move(CW, some_face)
                if up_side in edge.positions:
                    rubik.move(ACW, some_face)
                    left_algorithm(other_face, up_side, some_face)
                else:
                    rubik.move(ACW, some_face)
                    right_algorithm(other_face, up_side, some_face)

            assert up_side in edge.positions

            front_color = edge.colors[0 if edge.positions[1] == up_side else 1]
            up_color = edge.colors[0 if edge.positions[0] == up_side else 1]
            front_side = rubik.get_center(color=front_color).position
            # Now move the edge around the top such that center and edge have a common color
            # so as to apply the algorithms later on.
            while front_side not in edge.positions:
                rubik.move(CW, up_side)

            # Now figure out which direction to apply the algorithm
            # simply move top side to determine the orientations to apply algorithm.
            # if after moving to top clockwise, determine if edge has a common color with the center.
            # and take action accordingly.
            rubik.move(CW, up_side)
            if rubik.get_center(color=up_color).position in edge.positions:
                rubik.move(ACW, up_side)
                left_side = rubik.get_center(color=up_color).position
                left_algorithm(left_side, up_side, front_side)
            else:
                rubik.move(ACW, up_side)
                right_side = rubik.get_center(color=up_color).position
                right_algorithm(right_side, up_side, front_side)

        for edge in edges:
            assert edge.colors == tuple(rubik.get_colors(f) for f in edge.positions)

    @staticmethod
    def solve_top_cross(rubik, base=D):
        def algorithm(right, up, front):
            # Algorithm: F R U R' U' F'
            rubik.move(CW, front)
            rubik.move(CW, right)
            rubik.move(CW, up)
            rubik.move(ACW, right)
            rubik.move(ACW, up)
            rubik.move(ACW, front)

        up_side = OPPOSITE[base]
        up_color = rubik.get_colors(up_side)
        edges = list(filter(lambda _edge: up_side in _edge.positions, [rubik.get_edge(positions=e) for e in EDGES]))

        top_facing_edges = lambda: list(
            filter(lambda _edge: _edge.colors[_edge.positions.index(up_side)] == up_color, edges))

        if len(top_facing_edges()) == 0:
            # now orientation is required to apply the algorithm
            some_edge = edges[0]
            right_side = some_edge.positions[0 if some_edge.positions[1] == up_side else 1]
            rubik.move(CW, up_side)
            front_side = some_edge.positions[0 if some_edge.positions[1] == up_side else 1]
            rubik.move(ACW, up_side)

            # Apply the algorithm to obtain at least 2 edges
            algorithm(right_side, up_side, front_side)

        assert len(top_facing_edges()) > 0
        if len(top_facing_edges()) == 2:
            facing_edges = top_facing_edges()
            edge1_side = facing_edges[0].positions[0 if facing_edges[0].positions[1] == up_side else 1]
            edge2_side = facing_edges[1].positions[0 if facing_edges[1].positions[1] == up_side else 1]

            if OPPOSITE[edge1_side] != edge2_side:
                # L shape
                # Apply algorithm to convert it to line orientation.
                # First determine Orientation to apply algorithm
                rubik.move(CW, up_side)
                if edge2_side == facing_edges[0].positions[0 if facing_edges[0].positions[1] == up_side else 1]:
                    # If moving CW, edge1 moves to edge2. Then opposite of initial edge2 is front side.
                    rubik.move(ACW, up_side)
                    front_side, right_side = OPPOSITE[edge2_side], OPPOSITE[edge1_side]
                    algorithm(right_side, up_side, front_side)
                else:
                    rubik.move(ACW, up_side)
                    front_side, right_side = OPPOSITE[edge1_side], OPPOSITE[edge2_side]
                    algorithm(right_side, up_side, front_side)

            facing_edges = top_facing_edges()
            edge1_side = facing_edges[0].positions[0 if facing_edges[0].positions[1] == up_side else 1]
            edge2_side = facing_edges[1].positions[0 if facing_edges[1].positions[1] == up_side else 1]
            assert OPPOSITE[edge1_side] == edge2_side
            # Now there must be I shape

            rubik.move(CW, up_side)
            right_side = edge1_side
            front_side = facing_edges[0].positions[0 if facing_edges[0].positions[1] == up_side else 1]
            rubik.move(ACW, up_side)
            algorithm(right_side, up_side, front_side)

        # By the end of this step, top must contain a cross.
        assert len(top_facing_edges()) == 4

    @staticmethod
    def solve_top_edges(rubik, base=D):
        def algorithm(right, up):
            # Algorithm: R U R' U R U2 R' U
            rubik.move(CW, right)
            rubik.move(CW, up)
            rubik.move(ACW, right)
            rubik.move(CW, up)
            rubik.move(CW, right)
            rubik.move(CW, up, 2)
            rubik.move(ACW, right)
            rubik.move(CW, up)

        up_side = OPPOSITE[base]
        up_color = rubik.get_colors(up_side)
        edges = list(filter(lambda _edge: up_side in _edge.positions, [rubik.get_edge(positions=e) for e in EDGES]))

        def aligned_top_edges():
            return list(filter(lambda _edge: rubik.get_colors(
                _edge.positions[0 if _edge.positions[1] == up_side else 1]) in _edge.colors, edges))

        while len(aligned_top_edges()) <= 1:
            rubik.move(CW, up_side)

        aligned_edges = aligned_top_edges()
        if len(aligned_edges) == 4:
            # No need for any algorithm already aligned
            return

        assert len(aligned_edges) == 2

        edge1_side = aligned_edges[0].positions[0 if aligned_edges[0].positions[1] == up_side else 1]
        edge2_side = aligned_edges[1].positions[0 if aligned_edges[1].positions[1] == up_side else 1]
        if OPPOSITE[edge1_side] == edge2_side:
            # Aligned edges in opposite to each other.
            # apply algorithm to convert it into L pattern
            # any side would do for algorithm orientation not required
            algorithm(edge1_side, up_side)

        # Update the aligned top edges in case conversion took place.
        while len(aligned_top_edges()) <= 1:
            rubik.move(CW, up_side)

        aligned_edges = aligned_top_edges()
        edge1_side = aligned_edges[0].positions[0 if aligned_edges[0].positions[1] == up_side else 1]
        edge2_side = aligned_edges[1].positions[0 if aligned_edges[1].positions[1] == up_side else 1]
        assert OPPOSITE[edge1_side] != edge2_side

        # Not oriented top edge orientation must have L shape
        # Orientation is now required for the algorithm to complete this step.
        rubik.move(CW, up_side)
        right_side = edge1_side
        if edge2_side == aligned_edges[0].positions[0 if aligned_edges[0].positions[1] == up_side else 1]:
            right_side = edge2_side
        rubik.move(ACW, up_side)
        algorithm(right_side, up_side)

        assert len(aligned_top_edges()) == 4

    @staticmethod
    def solve_position_top_corners(rubik, base=D):
        def algorithm(left, right, up):
            # Algorithm: U R U' L' U R' U' L
            rubik.move(CW, up)
            rubik.move(CW, right)
            rubik.move(ACW, up)
            rubik.move(ACW, left)
            rubik.move(CW, up)
            rubik.move(ACW, right)
            rubik.move(ACW, up)
            rubik.move(CW, left)

        up_side = OPPOSITE[base]
        up_color = rubik.get_colors(up_side)
        corners = list(filter(lambda cnr: up_color in cnr.colors, [rubik.get_corner(positions=c) for c in CORNERS]))

        def aligned_top_corners():
            return list(filter(lambda cnr: set(cnr.colors) == set(rubik.get_colors(f) for f in cnr.positions), corners))

        if len(aligned_top_corners()) == 0:
            # Simply apply the algorithm.
            # Then one of them will auto align
            left_side = NEIGHBORS[up_side][0]
            right_side = OPPOSITE[left_side]
            algorithm(left_side, right_side, up_side)

        while len(aligned_top_corners()) == 1:
            aligned_corner = aligned_top_corners()[0]
            side1, side2 = list(filter(lambda f: f != up_side, aligned_corner.positions))
            rubik.move(CW, up_side)
            right_side = side2 if side1 in aligned_corner.positions else side1
            left_side = OPPOSITE[right_side]
            rubik.move(ACW, up_side)

            algorithm(left_side, right_side, up_side)

        assert len(aligned_top_corners()) == 4

    @staticmethod
    def solve_orient_top_corners(rubik, base=D):
        up_side = OPPOSITE[base]
        up_color = rubik.get_colors(up_side)
        down_side = base
        corners = list(filter(lambda cnr: up_color in cnr.colors, [rubik.get_corner(positions=c) for c in CORNERS]))

        def require_orientation():
            return list(filter(lambda cnr: cnr.colors != tuple(rubik.get_colors(f) for f in cnr.positions), corners))

        if not require_orientation():
            # If all are already oriented
            return

        # chose a top position to perform algorithm at.
        chosen_corner_index = 0
        corners_to_rotate = require_orientation()
        chosen_position = corners_to_rotate[chosen_corner_index].positions

        # Now identify the right side.
        side1, side2 = list(filter(lambda f: f != up_side, chosen_position))
        rubik.move(CW, up_side)
        # After rotation (destroys alignment) so  can't use require_orientation
        right_side = side2 if side1 in corners_to_rotate[chosen_corner_index].positions else side1
        rubik.move(ACW, up_side)

        # using reversed so that top layer is aligned after algorithm (as the chosen_corner_index = 0)
        for corner in reversed(require_orientation()):
            # rotate to move it to right place.
            while corner.positions != chosen_position:
                rubik.move(CW, up_side)

            # Apply algorithm till corner oriented.
            while not (up_side in corner.positions and corner.colors[corner.positions.index(up_side)] == up_color):
                rubik.move(ACW, right_side)
                rubik.move(ACW, down_side)
                rubik.move(CW, right_side)
                rubik.move(CW, down_side)

        # no top corner must now require any changes.
        assert not require_orientation()

    @staticmethod
    def solve_top_layer(rubik, base=D):
        RubikSolver.solve_top_cross(rubik, base)
        RubikSolver.solve_top_edges(rubik, base)
        RubikSolver.solve_position_top_corners(rubik, base)
        RubikSolver.solve_orient_top_corners(rubik, base)

    @staticmethod
    def solve(rubik, base=D):
        RubikSolver.make_bottom_cross(rubik, base)
        RubikSolver.make_bottom_corners(rubik, base)
        RubikSolver.solve_middle_layer(rubik, base)
        RubikSolver.solve_top_layer(rubik, base)
