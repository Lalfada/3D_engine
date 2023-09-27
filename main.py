import pygame
import math as M
from copy import deepcopy
from vector import Vec3, matrix_mul
from camera import Camera

BACKGROUND_COLOR = "black"
HEIGHT = 720
WIDTH = 1280
SMALL_BORDER = min(HEIGHT, WIDTH)
GAME_FPS = 60
CAMERA_SPEED = 5

Z_NEAR = 0.1
Z_FAR = 1000.0
FOV = M.radians(90)
LIGHT_SOURCE = Vec3(0, 0, 1)

ASPECT_RATIO = HEIGHT / WIDTH
DISTANCE_RATIO = 1 / M.tan(FOV / 2.0)
Z_NORMALIZATION = Z_FAR / (Z_FAR - Z_NEAR)

PROJECTION_MATRIX = [
    [ASPECT_RATIO * DISTANCE_RATIO, 0.0, 0.0, 0.0],
    [0.0, ASPECT_RATIO * DISTANCE_RATIO, 0.0, 0.0],
    [0.0, 0.0, Z_NORMALIZATION, Z_NEAR * Z_NORMALIZATION],
    [0.0, 0.0, 1.0, 0.0]
]

# helper function to divide x and y component by z
def projection_divide(vec, w):
    return Vec3(vec.x / w, vec.y / w, vec.z) if w != 0 else vec

SCREEN_SCCALING_MATRIX = [
    [0.5 * SMALL_BORDER, 0.0, 0.0, 0.5 * WIDTH],
    [0.0, 0.5 * SMALL_BORDER, 0.0, 0.5 * HEIGHT],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
]

# thx wikipediaa
def get_xrotation_matrix(angle):
    return [
        [1.0, 0.0, 0.0],
        [0.0, M.cos(angle), -M.sin(angle)],
        [0, M.sin(angle), M.cos(angle)],
    ]

# thx wikipedia
def get_yrotation_matrix(angle):
    return [
        [M.cos(angle), 0.0, M.sin(angle)],
        [0.0, 1.0, 0.0],
        [- M.sin(angle), 0.0, M.cos(angle)],
    ]

def get_pointat_matrix(pos, target, up):
    new_forward = (target - pos).normalized()

    a = new_forward * new_forward.dot(up)
    new_up = (up - a).normalized()

    right = new_up.cross(new_forward).normalized()

    return [
        [right.x, new_up.x, new_forward.x, pos.x],
        [right.y, new_up.y, new_forward.y, pos.y],
        [right.z, new_up.z, new_forward.z, pos.z],
        [0.0, 0.0, 0.0, 1.0]
    ]

# only works for rotation and translations matrices
# to me it's a black box
# mat is a 4x4 list
def inverse_matrix(mat):
    A = Vec3(mat[0][0], mat[1][0], mat[2][0])
    B = Vec3(mat[0][1], mat[1][1], mat[2][1])
    C = Vec3(mat[0][2], mat[1][2], mat[2][2])
    T = Vec3(mat[0][3], mat[1][3], mat[2][3])

    return [
        [A.x, A.y, A.z, -T.dot(A)],
        [B.x, B.y, B.z, -T.dot(B)],
        [C.x, C.y, C.z, -T.dot(C)],
        [0.0, 0.0, 0.0, 1.0],
    ]



cube_verticies = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0), Vec3(0, 1, 0),
    Vec3(0, 0, 1), Vec3(1, 0, 1), Vec3(1, 1, 1), Vec3(0, 1, 1)]

cube_indices = [
    # south
    [0, 3, 2],
    [2, 1, 0],    

    # east
    [2, 6, 5],
    [5, 1, 2],
    
    # north
    [6, 7, 4],
    [4, 5, 6],
    
    # west
    [7, 3, 0],
    [0, 4, 7],    

    # top
    [3, 7, 6],
    [6, 2, 3],
    
    # bottom
    [0, 1, 5],
    [5, 4, 0],
]


def mesh_from_indices(vertices, indices):
    mesh = []
    for row in indices:
        new_row = []
        for k in row:
            new_row.append(deepcopy(vertices[k]))
        mesh.append(new_row)
    return mesh


# thx chat gpt
def meshdata_from_lines(lines):
    vertices, indices = [], []
    for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                # Parse vertex (v) lines
                vertex = Vec3(float(parts[1]), 
                    float(parts[2]), float(parts[3]))
                vertices.append(vertex)
            elif parts[0] == 'f':
                # Parse face (f) lines
                # Subtract 1 to convert to 0-based indexing
                tri = [int(part.split('/')[0]) - 1 for part in parts[1:]]
                indices.append(tri) 
    return vertices, indices


def mesh_from_obj(path):
    with open(path, 'r') as obj_file:
        lines = obj_file.readlines()
        vertices, indices = meshdata_from_lines(lines)
        return mesh_from_indices(vertices, indices)


def get_normal(tri):
    A = tri[1] - tri[0]
    B = tri[2] - tri[0]
    return A.cross(B)

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)
    
def luminosity_from_light(normal, light):
    # light.length() = 1 so no need to divide by it
    normalized_angle = -normal.dot(light) / normal.length() / M.pi
    luminosity = (normalized_angle + 0.5)* 255
    luminosity = clamp(int(luminosity), 0, 255)
    return luminosity


def draw_tri_lines(tri):
    pygame.draw.lines(screen, (255, 255, 255), True, [
        (tri[0].x, tri[0].y),
        (tri[1].x, tri[1].y),
        (tri[2].x, tri[2].y)
    ])


def draw_tri_fill(color, tri):
    pygame.draw.polygon(screen, color,  [
        (tri[0].x, tri[0].y),
        (tri[1].x, tri[1].y),
        (tri[2].x, tri[2].y),
    ])


def render(cam):
    # theta_x =  pygame.time.get_ticks() * 1e-3
    theta_x = 0.0
    theta_y =  theta_x * 0.5
    xrot_mat = get_xrotation_matrix(theta_x)
    yrot_mat = get_yrotation_matrix(theta_y)

    target = cam.pos + cam.get_lookdir()
    camera_matrix = get_pointat_matrix(cam.pos, target, cam.up)
    view_matrix = inverse_matrix(camera_matrix)

    mesh_to_compute = deepcopy(model_mesh)
    drawing_buffer = []

    for tri in mesh_to_compute:
        for i, vec in enumerate(tri):
            # rotate
            vec = vec.matrix_mul(xrot_mat)
            vec = vec.matrix_mul(yrot_mat)
            # translate
            vec.z += 7.0
            tri[i] = vec

        lum = luminosity_from_light(
            get_normal(tri), LIGHT_SOURCE)
        
        for i, vec in enumerate(tri):
            # camera tranform
            vec = vec.extended_matrix_mul(view_matrix)
            tri[i] = vec

        # only draw triangles facing towards the camera
        tri_normal = get_normal(tri)
        if tri[0].dot(tri_normal) >= 0:
            continue

        z_mid = (tri[0].z + tri[1].z + tri[2].z) / 3.0
        drawing_buffer.append((tri, z_mid, tri_normal, lum))

    # sort based on the distance to the camera
    # this is an implementation of the painter's algorithm
    sorted_drawing_buffer = sorted(drawing_buffer, reverse = True, key = 
        lambda v: v[1]
        )

    for v in sorted_drawing_buffer:
        (tri, _, tri_normal, lum) = v
        for i, vec in enumerate(tri):
            # projection
            vec = vec.extended_matrix_mul(
                PROJECTION_MATRIX, projection_divide)
            # scale to the screen size
            vec = vec.extended_matrix_mul(SCREEN_SCCALING_MATRIX)
            tri[i] = vec

        color = (lum, lum, lum)

        draw_tri_fill(color, tri)
        # draw_triangle_fill(tri)


def update(dt, cam, yo):
    # fill the screen with a color 
    # to wipe away anything from last frame
    screen.fill(BACKGROUND_COLOR)
    cam.update(dt, yo)
    render(cam)


def game_loop(cam):
    running = True
    clock = pygame.time.Clock()
    count = 0
    while running:
        count += 1
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        update(clock.get_time() * 1e-3, cam, count % 60 == 0)
        # flip() the display to put your work on screen
        pygame.display.flip()
        clock.tick(GAME_FPS)  # limits FPS to 60


if __name__ == "__main__":
    mesh_cube = mesh_from_indices(cube_verticies, cube_indices) 
    model_path = "models/axis.obj"
    model_mesh = mesh_from_obj(model_path)

    camera = Camera(pos = Vec3(0, 0, 0),
        lam = M.pi / 2,
        phi = 0.0,
        up = Vec3(0, 1, 0))

    # initial setup
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # core loop
    game_loop(camera)
    # quit
    pygame.quit()