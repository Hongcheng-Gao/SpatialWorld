from constants import EDGES, CORNERS
from constants import D, CW, ACW
from constants import NEIGHBORS, OPPOSITE
from itertools import chain
from utilities import RubikUtilities
from functools import wraps


def assert_conditions(pre_condition, post_condition):
    """
        This is a decorator applied to the methods of the RubikSolver to check if pre-condition (input is as expected)
        And post-condition (output is as expected) are met to check the validity of the input and correctness of
        the method.
    """

    def decorator(solver_method):
        @wraps(solver_method)
        def inner_function(rubik, base):
            if pre_condition:
                # Check validity of input
                assert pre_condition(rubik, base)

            for direction, face in solver_method(rubik, base):
                yield direction, face

            if post_condition:
                # Check correctness of output.
                assert post_condition(rubik, base)

        return inner_function

    return decorator


class RubikSolver:
    """
        This class contains methods for solving various stages of cube and for the complete cube.
        All of the methods verify if previous stage is complete, and then proceed, so don't use methods
        before completing previous stage. They are there to mainly enforce correctness.
        Also note all the methods are generators, which yield direction and side to move. So after obtaining the
        values, strictly apply the move before requesting the next move otherwise behaviour is unpredictable and
        will fail.
    """

    @staticmethod
    @assert_conditions(None, RubikUtilities.is_bottom_cross_solved)
    def solve_bottom_cross(rubik, base=D):
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
                yield CW, face
                yield CW, face

            if up_side not in edge.positions:
                # if it is in the middle layer, bring it on top without affecting the bottom pieces
                # as they may already be aligned.
                top_rotated = False
                face = edge.positions[0]
                for _ in range(4):
                    yield (CW, face)
                    if up_side in edge.positions and not top_rotated:
                        yield (CW, up_side)
                        top_rotated = True

            # STEP 2: Move the edge right above where it needs to be by rotating the top.
            while nbr_c[index].position not in edge.positions:
                yield (CW, up_side)

            # STEP 3: Move it to the right place.
            # Ensure the edge has right orientation after moving.
            if (nbr_c[index].color == edge.colors[0] and nbr_c[index].position == edge.positions[0]) or \
                    (nbr_c[index].color == edge.colors[1] and nbr_c[index].position == edge.positions[1]):
                # If right orientation, simply rotating twice is enough.
                yield CW, nbr_c[index].position
                yield CW, nbr_c[index].position
            else:
                # Otherwise, do the following.
                yield (CW, up_side)
                face = edge.positions[0 if edge.positions[1] == up_side else 1]
                yield (CW, face)
                yield (ACW, nbr_c[index].position)
                yield (ACW, face)

    @staticmethod
    @assert_conditions(RubikUtilities.is_bottom_cross_solved, RubikUtilities.is_bottom_layer_solved)
    def solve_bottom_corners(rubik, base=D):
        corners = [rubik.get_corner(colors=tuple(map(rubik.get_colors, c))) for c in CORNERS if base in c]
        for corner in corners:
            # First move the corner to top later.
            if base in corner.positions:
                top_rotated = False
                return_direction = None
                face = corner.positions[0 if corner.positions[1] == base else 1]  # Any face would do
                for index in range(4):
                    yield (CW, face)
                    if OPPOSITE[base] in corner.positions and not top_rotated:
                        if index == 0:
                            yield (CW, OPPOSITE[base])
                            return_direction = ACW
                            top_rotated = True
                        if index == 2:
                            yield (ACW, OPPOSITE[base])
                            return_direction = CW
                            top_rotated = True
                yield (return_direction, OPPOSITE[base])
            # Move it to the right place on top
            while set(corner.colors) != set(
                    rubik.get_colors(p if p != OPPOSITE[base] else base) for p in corner.positions):
                yield (CW, OPPOSITE[base])
            # Now put it in place.
            face = None
            while corner.colors != tuple(map(rubik.get_colors, corner.positions)):
                positions = corner.positions
                yield (CW, OPPOSITE[base])
                if face is None:
                    face = next(p for p in positions if p not in corner.positions)
                yield (CW, face)
                yield (ACW, OPPOSITE[base])
                yield (ACW, face)

    @staticmethod
    @assert_conditions(RubikUtilities.is_bottom_layer_solved, RubikUtilities.is_middle_layer_solved)
    def solve_middle_layer(rubik, base=D):
        # Algorithm definitions
        def left_algorithm(left, up, front):
            # Algorithm: U' L' U L U F U' F'
            for _direction, _face in [(ACW, up), (ACW, left), (CW, up), (CW, left), (CW, up), (CW, front), (ACW, up),
                                      (ACW, front)]:
                yield _direction, _face

        def right_algorithm(right, up, front):
            # Algorithm: U R U' R' U' F' U F
            for _direction, _face in [(CW, up), (CW, right), (ACW, up), (ACW, right), (ACW, up), (ACW, front), (CW, up),
                                      (CW, front)]:
                yield _direction, _face

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
                yield (CW, some_face)
                if up_side in edge.positions:
                    yield (ACW, some_face)
                    for direction, face in left_algorithm(other_face, up_side, some_face):
                        yield direction, face
                else:
                    yield (ACW, some_face)
                    for direction, face in right_algorithm(other_face, up_side, some_face):
                        yield direction, face

            assert up_side in edge.positions

            front_color = edge.colors[0 if edge.positions[1] == up_side else 1]
            up_color = edge.colors[0 if edge.positions[0] == up_side else 1]
            front_side = rubik.get_center(color=front_color).position
            # Now move the edge around the top such that center and edge have a common color
            # so as to apply the algorithms later on.
            while front_side not in edge.positions:
                yield (CW, up_side)

            # Now figure out which direction to apply the algorithm
            # simply move top side to determine the orientations to apply algorithm.
            # if after moving to top clockwise, determine if edge has a common color with the center.
            # and take action accordingly.
            yield (CW, up_side)
            if rubik.get_center(color=up_color).position in edge.positions:
                yield (ACW, up_side)
                left_side = rubik.get_center(color=up_color).position
                for direction, face in left_algorithm(left_side, up_side, front_side):
                    yield direction, face
            else:
                yield (ACW, up_side)
                right_side = rubik.get_center(color=up_color).position
                for direction, face in right_algorithm(right_side, up_side, front_side):
                    yield direction, face
        # POST CONDITION
        assert RubikUtilities.is_middle_layer_solved(rubik, base)

    @staticmethod
    @assert_conditions(RubikUtilities.is_middle_layer_solved, RubikUtilities.is_top_cross_solved)
    def solve_top_cross(rubik, base=D):
        def algorithm(right, up, front):
            # Algorithm: F R U R' U' F'
            for _direction, _face in [(CW, front), (CW, right), (CW, up), (ACW, right), (ACW, up), (ACW, front)]:
                yield _direction, _face

        up_side = OPPOSITE[base]
        up_color = rubik.get_colors(up_side)
        edges = list(filter(lambda _edge: up_side in _edge.positions, [rubik.get_edge(positions=e) for e in EDGES]))

        top_facing_edges = lambda: list(
            filter(lambda _edge: _edge.colors[_edge.positions.index(up_side)] == up_color, edges))

        if len(top_facing_edges()) == 0:
            # now orientation is required to apply the algorithm
            some_edge = edges[0]
            right_side = some_edge.positions[0 if some_edge.positions[1] == up_side else 1]
            yield (CW, up_side)
            front_side = some_edge.positions[0 if some_edge.positions[1] == up_side else 1]
            yield (ACW, up_side)

            # Apply the algorithm to obtain at least 2 edges
            for direction, face in algorithm(right_side, up_side, front_side):
                yield direction, face

        assert len(top_facing_edges()) > 0
        if len(top_facing_edges()) == 2:
            facing_edges = top_facing_edges()
            edge1_side = facing_edges[0].positions[0 if facing_edges[0].positions[1] == up_side else 1]
            edge2_side = facing_edges[1].positions[0 if facing_edges[1].positions[1] == up_side else 1]

            if OPPOSITE[edge1_side] != edge2_side:
                # L shape
                # Apply algorithm to convert it to line orientation.
                # First determine Orientation to apply algorithm
                yield (CW, up_side)
                if edge2_side == facing_edges[0].positions[0 if facing_edges[0].positions[1] == up_side else 1]:
                    # If moving CW, edge1 moves to edge2. Then opposite of initial edge2 is front side.
                    yield (ACW, up_side)
                    front_side, right_side = OPPOSITE[edge2_side], OPPOSITE[edge1_side]
                    for direction, face in algorithm(right_side, up_side, front_side):
                        yield direction, face
                else:
                    yield (ACW, up_side)
                    front_side, right_side = OPPOSITE[edge1_side], OPPOSITE[edge2_side]
                    for direction, face in algorithm(right_side, up_side, front_side):
                        yield direction, face

            facing_edges = top_facing_edges()
            edge1_side = facing_edges[0].positions[0 if facing_edges[0].positions[1] == up_side else 1]
            edge2_side = facing_edges[1].positions[0 if facing_edges[1].positions[1] == up_side else 1]
            assert OPPOSITE[edge1_side] == edge2_side
            # Now there must be I shape

            yield (CW, up_side)
            right_side = edge1_side
            front_side = facing_edges[0].positions[0 if facing_edges[0].positions[1] == up_side else 1]
            yield (ACW, up_side)
            for direction, face in algorithm(right_side, up_side, front_side):
                yield direction, face

    @staticmethod
    @assert_conditions(RubikUtilities.is_top_cross_solved, RubikUtilities.is_top_edges_solved)
    def solve_top_edges(rubik, base=D):
        def algorithm(right, up):
            # Algorithm: R U R' U R U2 R' U
            for _direction, _face in [(CW, right), (CW, up), (ACW, right), (CW, up), (CW, right), (CW, up), (CW, up),
                                      (ACW, right), (CW, up)]:
                yield _direction, _face

        up_side = OPPOSITE[base]
        edges = list(filter(lambda _edge: up_side in _edge.positions, [rubik.get_edge(positions=e) for e in EDGES]))

        def aligned_top_edges():
            return list(filter(lambda _edge: rubik.get_colors(
                _edge.positions[0 if _edge.positions[1] == up_side else 1]) in _edge.colors, edges))

        while len(aligned_top_edges()) <= 1:
            yield (CW, up_side)

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
            for direction, face in algorithm(edge1_side, up_side):
                yield direction, face

        # Update the aligned top edges in case conversion took place.
        while len(aligned_top_edges()) <= 1:
            yield (CW, up_side)

        aligned_edges = aligned_top_edges()
        edge1_side = aligned_edges[0].positions[0 if aligned_edges[0].positions[1] == up_side else 1]
        edge2_side = aligned_edges[1].positions[0 if aligned_edges[1].positions[1] == up_side else 1]
        assert OPPOSITE[edge1_side] != edge2_side

        # Not oriented top edge orientation must have L shape
        # Orientation is now required for the algorithm to complete this step.
        yield (CW, up_side)
        right_side = edge1_side
        if edge2_side == aligned_edges[0].positions[0 if aligned_edges[0].positions[1] == up_side else 1]:
            right_side = edge2_side
        yield (ACW, up_side)
        for direction, face in algorithm(right_side, up_side):
            yield direction, face

    @staticmethod
    @assert_conditions(RubikUtilities.is_top_edges_solved, RubikUtilities.is_positioned_top_corners)
    def solve_position_top_corners(rubik, base=D):
        def algorithm(left, right, up):
            # Algorithm: U R U' L' U R' U' L
            for _direction, _face in [(CW, up), (CW, right), (ACW, up), (ACW, left), (CW, up), (ACW, right), (ACW, up),
                                      (CW, left)]:
                yield _direction, _face

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
            for direction, face in algorithm(left_side, right_side, up_side):
                yield direction, face

        while len(aligned_top_corners()) == 1:
            aligned_corner = aligned_top_corners()[0]
            side1, side2 = list(filter(lambda f: f != up_side, aligned_corner.positions))
            yield (CW, up_side)
            right_side = side2 if side1 in aligned_corner.positions else side1
            left_side = OPPOSITE[right_side]
            yield (ACW, up_side)

            for direction, face in algorithm(left_side, right_side, up_side):
                yield direction, face

    @staticmethod
    @assert_conditions(RubikUtilities.is_positioned_top_corners, RubikUtilities.is_oriented_top_corners)
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
        yield (CW, up_side)
        # After rotation (destroys alignment) so  can't use require_orientation
        right_side = side2 if side1 in corners_to_rotate[chosen_corner_index].positions else side1
        yield (ACW, up_side)

        # using reversed so that top layer is aligned after algorithm (as the chosen_corner_index = 0)
        for corner in reversed(require_orientation()):
            # rotate to move it to right place.
            while corner.positions != chosen_position:
                yield (CW, up_side)

            # Apply algorithm till corner oriented.
            while not (up_side in corner.positions and corner.colors[corner.positions.index(up_side)] == up_color):
                yield (ACW, right_side)
                yield (ACW, down_side)
                yield (CW, right_side)
                yield (CW, down_side)


    @staticmethod
    def solve_top_layer(rubik, base=D):
        return chain(RubikSolver.solve_top_cross(rubik, base), RubikSolver.solve_top_edges(rubik, base),
                     RubikSolver.solve_position_top_corners(rubik, base),
                     RubikSolver.solve_orient_top_corners(rubik, base))

    @staticmethod
    def solve(rubik, base=D):
        return chain(RubikSolver.solve_bottom_layer(rubik, base), RubikSolver.solve_middle_layer(rubik, base),
                     RubikSolver.solve_top_layer(rubik, base))

    @staticmethod
    def solve_bottom_layer(rubik, base=D):
        return chain(RubikSolver.solve_bottom_cross(rubik, base), RubikSolver.solve_bottom_corners(rubik, base))

    @staticmethod
    def solve_next_step(rubik, base=D):
        """
            This method identifies the current stage the cube is solved to, then solves it to the next stage.
        """
        if not RubikUtilities.is_bottom_cross_solved(rubik, base):
            return RubikSolver.solve_bottom_cross(rubik, base)
        elif not RubikUtilities.is_bottom_layer_solved(rubik, base):
            return RubikSolver.solve_bottom_corners(rubik, base)
        elif not RubikUtilities.is_middle_layer_solved(rubik, base):
            return RubikSolver.solve_middle_layer(rubik, base)
        elif not RubikUtilities.is_top_cross_solved(rubik, base):
            return RubikSolver.solve_top_cross(rubik, base)
        elif not RubikUtilities.is_top_edges_solved(rubik, base):
            return RubikSolver.solve_top_edges(rubik, base)
        elif not RubikUtilities.is_positioned_top_corners(rubik, base):
            return RubikSolver.solve_position_top_corners(rubik, base)
        elif not RubikUtilities.is_oriented_top_corners(rubik, base):
            return RubikSolver.solve_orient_top_corners(rubik, base)
        return ()
