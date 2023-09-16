import pygame
import math as M
from copy import deepcopy
from vector import Vec3, matrix_mul

BACKGROUND_COLOR = "black"
HEIGHT = 720
# WIDTH = 1280
WIDTH = HEIGHT
GAME_FPS = 60

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

# thx wikipedia
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
        newRow = []
        for k in row:
            newRow.append(deepcopy(vertices[k]))
        mesh.append(newRow)
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
                vertex = Vec3(float(parts[1]), float(parts[2]), float(parts[3]))
                vertices.append(vertex)
            elif parts[0] == 'f':
                # Parse face (f) lines
                tri = [int(part.split('/')[0]) - 1 for part in parts[1:]]  # Subtract 1 to convert to 0-based indexing
                indices.append(tri)
    return vertices, indices


# thx chat gpt
def mesh_from_obj(path):
    with open(path, 'r') as obj_file:
        lines = obj_file.readlines()
        vertices, indices = meshdata_from_lines(lines)
        
        return mesh_from_indices(vertices, indices)


def get_normal(tri):
    A = tri[1] - tri[0]
    B = tri[2] - tri[0]
    return A.cross(B)

    
def luminosity_from_light(normal, light):
    luminosity = -normal.dot(light)
    luminosity *= 255 / normal.length()
    luminosity = int(luminosity)
    luminosity = luminosity if luminosity >= 0 else 0
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


def update():
    theta_x =  pygame.time.get_ticks() * 1e-3
    theta_y =  theta_x * 0.5
    xrot_mat = get_xrotation_matrix(theta_x)
    yrot_mat = get_yrotation_matrix(theta_y)
    mesh_to_compute = deepcopy(model_mesh)

    drawing_buffer = []

    for tri in mesh_to_compute:
        # rotate and translate
        for i, vec in enumerate(tri):
            vec = vec.matrix_mul(xrot_mat)
            vec = vec.matrix_mul(yrot_mat)
            vec.z += 10.0

            tri[i] = vec

        # only draw triangles facing towards the camera
        tri_normal = get_normal(tri)
        if tri[0].dot(tri_normal) >= 0:
            continue   

        z_mid = (tri[0].z + tri[1].z + tri[2].z) / 3.0
        drawing_buffer.append((tri, z_mid, tri_normal))

    # sort based on the distance to the camera
    # this is an implementation of the painter's algorithm
    sorted_drawing_buffer = sorted(drawing_buffer, reverse = True, key = 
        lambda v: v[1]
        )

    for v in sorted_drawing_buffer:
        (tri, _, tri_normal) = v

        # projection and screen scaling
        for i, vec in enumerate(tri):
            vec = vec.extended_matrix_mul(PROJECTION_MATRIX)
            vec += Vec3(1, 1, 0)
            vec.x *= 0.5 * WIDTH
            vec.y *= 0.5 * HEIGHT
            tri[i] = vec

        lum = luminosity_from_light(tri_normal, LIGHT_SOURCE)
        color = (lum, lum, lum)

        draw_tri_fill(color, tri)
        # draw_triangle_fill(tri)


def game_loop():
    running = True
    clock = pygame.time.Clock()
    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # fill the screen with a color 
        # to wipe away anything from last frame
        screen.fill(BACKGROUND_COLOR)
        # render the game
        update()
        # flip() the display to put your work on screen
        pygame.display.flip()
        clock.tick(GAME_FPS)  # limits FPS to 60


if __name__ == "__main__":
    # mesh_cube = mesh_from_indices(cube_verticies, cube_indices) 
    model_path = "AirShip.obj"
    model_mesh = mesh_from_obj(model_path)

    # initial setup
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # core loop
    game_loop()
    # quit
    pygame.quit()