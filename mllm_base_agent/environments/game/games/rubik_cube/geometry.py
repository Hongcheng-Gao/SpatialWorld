from pygame import Vector2, Vector3
from collections import defaultdict
from constants import F, B, L, R, U, D
from constants import FACES, OFFSET, SCALE



# X-Axis -> RIGHT
# Y-Axis -> UP
# Z-Axis -> FRONT
CP = CornerPoints = {
    (R, U, F): Vector3(+1, +1, +1),
    (R, U, B): Vector3(+1, +1, -1),
    (R, D, F): Vector3(+1, -1, +1),
    (R, D, B): Vector3(+1, -1, -1),
    (L, U, F): Vector3(-1, +1, +1),
    (L, U, B): Vector3(-1, +1, -1),
    (L, D, F): Vector3(-1, -1, +1),
    (L, D, B): Vector3(-1, -1, -1),
}

def get_init_points():
    common = lambda l1, l2: tuple(x for x in l1 if x in l2)
    centers, edges, corners = {}, defaultdict(dict), defaultdict(dict)
    for f in FACES:
        centers[f] = (
            (2 * CP[FACES[f][0]] + CP[FACES[f][2]]) / 3,
            (2 * CP[FACES[f][1]] + CP[FACES[f][3]]) / 3,
            (CP[FACES[f][0]] + 2 * CP[FACES[f][2]]) / 3,
            (CP[FACES[f][1]] + 2 * CP[FACES[f][3]]) / 3
        )

        edges[common(FACES[f][0], FACES[f][1])][f] = (
            (2 * CP[FACES[f][0]] + CP[FACES[f][1]]) / 3,
            (CP[FACES[f][0]] + 2 * CP[FACES[f][1]]) / 3,
            (2 * CP[FACES[f][1]] + CP[FACES[f][3]]) / 3,
            (2 * CP[FACES[f][0]] + CP[FACES[f][2]]) / 3
        )

        edges[common(FACES[f][1], FACES[f][2])][f] = (
            (2 * CP[FACES[f][1]] + CP[FACES[f][3]]) / 3,
            (2 * CP[FACES[f][1]] + CP[FACES[f][2]]) / 3,
            (CP[FACES[f][1]] + 2 * CP[FACES[f][2]]) / 3,
            (CP[FACES[f][0]] + 2 * CP[FACES[f][2]]) / 3
        )
        edges[common(FACES[f][2], FACES[f][3])][f] = (
            (CP[FACES[f][1]] + 2 * CP[FACES[f][3]]) / 3,
            (CP[FACES[f][0]] + 2 * CP[FACES[f][2]]) / 3,
            (2 * CP[FACES[f][2]] + CP[FACES[f][3]]) / 3,
            (CP[FACES[f][2]] + 2 * CP[FACES[f][3]]) / 3
        )
        edges[common(FACES[f][3], FACES[f][0])][f] = (
            (2 * CP[FACES[f][0]] + CP[FACES[f][3]]) / 3,
            (2 * CP[FACES[f][0]] + CP[FACES[f][2]]) / 3,
            (CP[FACES[f][1]] + 2 * CP[FACES[f][3]]) / 3,
            (CP[FACES[f][0]] + 2 * CP[FACES[f][3]]) / 3
        )

        corners[FACES[f][0]][f] = (
            1 * CP[FACES[f][0]],
            (2 * CP[FACES[f][0]] + CP[FACES[f][1]]) / 3,
            (2 * CP[FACES[f][0]] + CP[FACES[f][2]]) / 3,
            (2 * CP[FACES[f][0]] + CP[FACES[f][3]]) / 3
        )
        corners[FACES[f][1]][f] = (
            (CP[FACES[f][0]] + 2 * CP[FACES[f][1]]) / 3,
            1 * CP[FACES[f][1]],
            (2 * CP[FACES[f][1]] + CP[FACES[f][2]]) / 3,
            (2 * CP[FACES[f][1]] + CP[FACES[f][3]]) / 3
        )
        corners[FACES[f][2]][f] = (
            (CP[FACES[f][0]] + 2 * CP[FACES[f][2]]) / 3,
            (CP[FACES[f][1]] + 2 * CP[FACES[f][2]]) / 3,
            1 * CP[FACES[f][2]],
            (2 * CP[FACES[f][2]] + CP[FACES[f][3]]) / 3
        )
        corners[FACES[f][3]][f] = (
            (CP[FACES[f][0]] + 2 * CP[FACES[f][3]]) / 3,
            (CP[FACES[f][1]] + 2 * CP[FACES[f][3]]) / 3,
            (CP[FACES[f][2]] + 2 * CP[FACES[f][3]]) / 3,
            1 * CP[FACES[f][3]]
        )
    points,surfaces = [], []
    for f in centers:
        points.extend(centers[f])
    for e in edges:
        for f in e:
            points.extend(edges[e][f])
    for c in corners:
        for f in c:
            points.extend(corners[c][f])


    return points, centers, edges, corners

def z_orientation(surf):
    return Vector3.cross(surf[1] - surf[0], surf[2] - surf[1]).z


def perspective_projection(pt):
    # Assuming camera has orientation along -z axis to easy calculations.
    # Thus the display surface is perpendicular to z-axis
    # display_surface_distance is the distance between camera and the display surface
    # http.
    camera = Vector3(0, 0, 6)
    display_surface_distance = 0.2
    projected_point = (pt - camera) * display_surface_distance/ (camera.z - pt.z)
    # Return the point after normalizing
    return projected_point * camera.z / display_surface_distance

def xy_projection(surf):
    perspective_points = tuple(map(perspective_projection, surf))
    return tuple(Vector2(pt.x * SCALE.x, -pt.y * SCALE.y) + OFFSET for pt in perspective_points)

