import pygame
import math as M
from copy import deepcopy
from vector import Vec3, matrix_mul
from camera import Camera
from image import Image_Data

BACKGROUND_COLOR = "black"
HEIGHT = 400
WIDTH = 600
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

SCREEN_BORDER_PLANES = [ 
    # a plane is a normal and a point
    (Vec3(1, 0, 0), Vec3(0, 0, 0)), # left
    (Vec3(0, 1, 0), Vec3(0, 0, 0)), # top
    (Vec3(-1, 0, 0), Vec3(WIDTH, HEIGHT, 0)), # right
    (Vec3(0, -1, 0), Vec3(WIDTH, HEIGHT, 0)), # bottom

]

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

    # bot
    [3, 7, 6],
    [6, 2, 3],
    
    # top
    [0, 1, 5],
    [5, 4, 0],
]

minimal_indices = [[0, 1, 5]]

image_vertices = [
    Vec3(0, 0, 1), # top left
    Vec3(1, 0, 1), # top right
    Vec3(0, 1, 1), # bot left
    Vec3(1, 1, 1), # bot right
]

cube_uv_indices = [
    # south
    [0, 2, 3],
    [3, 1, 0],    

    # east
    [0, 2, 3],
    [3, 1, 0],
    
    # north
    [0, 2, 3],
    [3, 1, 0],
    
    # west
    [0, 2, 3],
    [3, 1, 0],   

    # top
    [0, 2, 3],
    [3, 1, 0],
    
    # bottom
    [0, 2, 3],
    [3, 1, 0],
]

minimal_uv_indices = [[0, 2, 3]]


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

def on_normal_side(plane, vec):
    (normal, point) = plane
    return normal.dot(vec - point) >= 0


def lerp(a, b, t):
    return a * (1 - t) + b * t


# this function is taken from this thread
# https://stackoverflow.com/questions/5666222/
# 3d-line-plane-intersection#18543221
def intersect_edge_plane(plane, a, b, epsilon=1e-6):
    (vertex_a, uv_a) = a
    (vertex_b, uv_b) = b
    (normal, point) = plane

    delta = vertex_b - vertex_a
    dot = normal.dot(delta)

    if abs(dot) > epsilon:
        # The factor of the point between a -> b (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # Otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = vertex_a - point
        t = - normal.dot(w) / dot

        # use the factor of vertex space to interect in texture space
        return (lerp(vertex_a, vertex_b, t), lerp(uv_a, uv_b, t))

    return None


def unzip_to_tuple(zipped_tris):
    vertices = [zipped_tris[0][0], zipped_tris[1][0], zipped_tris[2][0]]
    uvs = [zipped_tris[0][1], zipped_tris[1][1], zipped_tris[2][1]]
    return (vertices, uvs)


def clip_on_plane(plane, tri_data):
    (tri, uv) = tri_data
    (normal, point) = plane

    normal_side = []
    other_side = []
    # treat vertex and its uv as one unit for the caclations
    vertecies_data = zip(tri, uv)
    for vertex_data in vertecies_data:
        if on_normal_side(plane, vertex_data[0]):
            normal_side.append(deepcopy(vertex_data))
        else:
            other_side.append(deepcopy(vertex_data))

    # each case implies somthing else for what should be kept
    # the order IS important !
    # it not respected it will mess with the normals
    n = len(normal_side)
    if n == 3: return [tri_data]
    if n == 0: return []
    if n == 1:
        # the unzip is for seperating the vertexs and uvs into
        # a vertex list and a uv list
        return [unzip_to_tuple([normal_side[0],
            intersect_edge_plane(plane, normal_side[0], other_side[0]),
            intersect_edge_plane(plane, normal_side[0], other_side[1])
            ])]
    else: # n == 2
        boundary1 = intersect_edge_plane(plane, 
            normal_side[0], other_side[0])
        boundary2 = intersect_edge_plane(plane, 
            normal_side[1], other_side[0])
        
        return [
            unzip_to_tuple([
                normal_side[0],
                normal_side[1], 
                boundary1,
            ]),
            unzip_to_tuple([
                normal_side[1],
                boundary2, 
                boundary1,
            ]),
        ]
    
def clip_multiple_planes(planes, tri):
    to_clip = [tri]
    for plane in planes:
        # clip all tris, we get a 2D list of tris
        clipped_tris = [clip_on_plane(plane, tri) for tri in to_clip]
        # turn the list into a 1D list of tris
        to_clip = []
        for row in clipped_tris:
            for tri in row:
                to_clip.append(tri)
    return to_clip


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

    
def luminosity_from_light(normal, light):
    # light.length() = 1 so no need to divide by it
    normalized_angle = -normal.dot(light) / normal.length() / M.pi
    luminosity = (normalized_angle + 0.5)* 255
    luminosity = clamp(int(luminosity), 0, 255)
    return luminosity


def draw_tri_lines(tri):
    pygame.draw.lines(screen, (0, 0, 0), True, [
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


def get_slope(a, b):
    # value won't be used anyways in that case
    # so need to return None (my IDE doesn't like it)
    if b.y - a.y == 0: return 0 
    return (b.x - a.x) / (b.y - a.y)


def draw_pixel(x, y, col):
    # pygame.draw.rect(screen, col, (x, y, 1, 1))
    screen.set_at((x, y), col)


def rasterize_trapezoid(image, a, b, a_uv, b_uv, slope_ab, slope_ac, ac_start):
    delta_y = round(b.y - a.y) + 1
    for i in range(delta_y):
        y = round(a.y) + i
        start_x = round(a.x + i * slope_ab)
        end_x = round(ac_start + i * slope_ac)

        if start_x > end_x:
            start_x, end_x = end_x, start_x
        
        ty = (i + 1) / delta_y
        if not (0 <= ty <= 1): print(f"ty: {ty}")
        v = a_uv.y * (1 - ty) + b_uv.y * ty

        delta_x = end_x - start_x + 1
        for j in range(delta_x):
            x = start_x + j

            tx = (j + 1) / delta_x
            if not (0 <= tx <= 1): print(f"tx: {tx}")
            u = a_uv.x * (1 - tx) + b_uv.x * tx
            if not (0 <= u <= 1) or not (0 <= v <= 1):
                print(f"""a: [{a}]\na_uv: [{a_uv}]\nb: [{b}]\n
                    b_uv: [{b_uv}]\nty: {ty}; tx: {tx}; v: {v}; u: {u}""")

            col = image.pixel_from_uv(u, v)
            # draw_pixel(x, y, col)
            draw_pixel(x, y, (255, 255, 255))


def rasterize_trapezoid2(image, pixels, a, b, c, d, a_uv, b_uv, c_uv, d_uv):
    delta_y = round(c.y - a.y) + 1
    for i in range(delta_y):
        y = round(a.y) + i
        ty = i / (delta_y - 1) if delta_y - 1 != 0 else 0.5

        start_x = round(lerp(a.x, c.x, ty))
        end_x = round(lerp(b.x, d.x, ty))
        start_uv = lerp(a_uv, c_uv, ty)
        end_uv = lerp(b_uv, d_uv, ty)

        if start_x > end_x:
            start_x, end_x = end_x, start_x
            # I forgot this line and spent 2 hours debugging
            # grrornrnornronrnrono
            start_uv, end_uv = end_uv, start_uv
        
        if not (0 <= ty <= 1): print(f"ty: {ty}")

        delta_x = end_x - start_x + 1
        for j in range(delta_x):
            x = start_x + j

            tx = (j + 1) / delta_x
            if not (0 <= tx <= 1): print(f"tx: {tx}")
            uv = lerp(start_uv, end_uv, tx)
            # if not (0 <= u <= 1) or not (0 <= v <= 1):
                # print(f"""a: [{a}]\na_uv: [{a_uv}]\nb: [{b}]
# b_uv: [{b_uv}]\nty: {ty}; tx: {tx}; v: {v}; u: {u}""")

            col = image.pixel_from_uv(uv.x / uv.z, uv.y / uv.z)
            # pixels[y][x] = col
            draw_pixel(x, y, col)
            # draw_pixel(x, y, (255, 255, 255))


def rasterize(tris_data, image, pixels, should_print):
    zipped_tris = list(zip(tris_data[0], tris_data[1]))
    sorted_tris = sorted(zipped_tris, key=
        lambda x: x[0].y)
    
    a, a_uv = sorted_tris[0][0], sorted_tris[0][1]
    b, b_uv = sorted_tris[1][0], sorted_tris[1][1]
    c, c_uv = sorted_tris[2][0], sorted_tris[2][1]

    slope_ab = get_slope(a, b)
    slope_ac = get_slope(a, c)
    slope_bc = get_slope(b, c)

    # rasterize_trapezoid(image, a, b, a_uv, b_uv, slope_ab, slope_ac, a.x)
    # ac_start =  round(a.x + round(b.y - a.y) * slope_ac)
    # rasterize_trapezoid(image, b, c, b_uv, c_uv, slope_bc, slope_ac,
    #     ac_start)
    
    # d is the point which y value is equal to b's y value
    # and lie on the line betwen a and c
    t = (b.y - a.y) / (c.y - a.y) if c.y - a.y != 0 else 0
    d, d_uv = a + t * (c - a), a_uv + t * (c_uv - a_uv)

    if b.x > d.x:
        b, d = b, d

    rasterize_trapezoid2(image, pixels, a, a, b, d, a_uv, a_uv, b_uv, d_uv)
    rasterize_trapezoid2(image, pixels, b, d, c, c, b_uv, d_uv, c_uv, c_uv)

    if False:
        print(f"""
a: {a}
b: {b}
c: {c}
d: {d}
a_uv: {a_uv}
b_uv: {b_uv}
c_uv: {c_uv}
d_uv: {d_uv}
t {t}


""")
    

def render(cam, pixels, should_print):
    # theta_x =  pygame.time.get_ticks() * 1e-3
    theta_x = 0.0
    theta_y =  theta_x * 0.5
    xrot_mat = get_xrotation_matrix(theta_x)
    yrot_mat = get_yrotation_matrix(theta_y)

    target = cam.pos + cam.get_lookdir()
    camera_matrix = get_pointat_matrix(cam.pos, target, cam.up)
    view_matrix = inverse_matrix(camera_matrix)

    mesh_to_compute = deepcopy(cube_data)
    image = flustered_girl
    drawing_buffer = []

    for data in mesh_to_compute:
        (tri, uv) = data
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
            # camera tranform ie view matrix
            vec, _ = vec.extended_matrix_mul(view_matrix)
            tri[i] = vec
        
        # only draw triangles facing towards the camera
        # tri[0] is just a vertex, but it is also
        # the vector from the camera to the vertex since
        # we are in view space
        tri_normal = get_normal(tri)
        if tri[0].dot(tri_normal) >= 0:
            continue

        # clip on close plane
        clipped_tris = clip_on_plane((Vec3(0, 0, 1), 
                Vec3(0, 0, Z_NEAR)), (tri, uv))
        for clipped_tri_data in clipped_tris:
            (clipped_tri, uv) = clipped_tri_data

            z_mid = (clipped_tri[0].z + clipped_tri[1].z 
                + clipped_tri[2].z) / 3.0
            drawing_buffer.append((clipped_tri, uv, z_mid, lum))

    # sort based on the distance to the camera
    # this is an implementation of the painter's algorithm
    sorted_drawing_buffer = sorted(drawing_buffer, reverse = True, key = 
        lambda v: v[2]
        )

    for v in sorted_drawing_buffer:
        (tri, uv, _, lum) = v
        for i, vec in enumerate(tri):
            # projection
            vec, w = vec.extended_matrix_mul(
                PROJECTION_MATRIX, projection_divide)
            uv[i] /= w # important for texture perspective

            # scale to the screen size
            vec, _ = vec.extended_matrix_mul(SCREEN_SCCALING_MATRIX)
            tri[i] = vec

        clipped_tris = clip_multiple_planes(SCREEN_BORDER_PLANES, (tri, uv))
        for clipped_tri_data in clipped_tris:
            (clipped_tri, uv) = clipped_tri_data

            color = (lum, lum, lum)

            rasterize(clipped_tri_data, image, pixels, should_print)
            # draw_tri_fill(color, clipped_tri)
            draw_tri_lines(clipped_tri)

"""
def flush(pixel_colors):
    surface = pygame.Surface((WIDTH, HEIGHT))
    pixels = pygame.PixelArray(surface)

    # Assign colors to each pixel
    for y in range(HEIGHT):
        for x in range(WIDTH):
            pixels[x][y] = pixel_colors[y][x]

    # Delete the pixel array to apply the changes
    del pixels
"""

def render_text(what, color, where):
    font = pygame.font.Font('Roboto-Regular.ttf', 20)
    text = font.render(what, 1, pygame.Color(color))
    screen.blit(text, where)


def update(dt, clock, cam, should_print):
    # fill the screen with a color 
    # to wipe away anything from last frame
    pixels = [
        [(0, 0, 0) for _ in range(WIDTH)] for _ in range(HEIGHT)
    ]
    screen.fill(BACKGROUND_COLOR)
    cam.update(dt)
    render(cam, pixels, should_print)
    # flush(pixels)
    # show fps
    render_text(f"FPS: {int(clock.get_fps())}", (255,255,120), (0,0))


def game_loop(cam):
    running = True
    clock = pygame.time.Clock()
    count = 0
    should_print = False
    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        should_print = count % 60 == 0
        count += 1

        update(clock.get_time() * 1e-3, clock, cam, should_print)
        # flip() the display to put your work on screen
        pygame.display.flip()
        clock.tick(GAME_FPS)  # limits FPS to 60


if __name__ == "__main__":
    cube_mesh = mesh_from_indices(cube_verticies, cube_indices)
    cube_uv = mesh_from_indices(image_vertices, cube_uv_indices)
    cube_data = zip(cube_mesh, cube_uv)
    minimal_mesh = mesh_from_indices(cube_verticies, minimal_indices)
    minimal_uv = mesh_from_indices(image_vertices, minimal_uv_indices)
    minimal_data = zip(minimal_mesh, minimal_uv)

    model_path = "models/axis.obj"
    model_mesh = mesh_from_obj(model_path)
    flustered_girl = Image_Data("flustered_girl.png")

    # initial setup
    pygame.init()
    # screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # camera has to be iniated after pygame.init
    camera = Camera(pos = Vec3(0, 0, 0),
        lam = M.pi / 2,
        phi = 0.0,
        up = Vec3(0, 1, 0),
        mouse_pos = (WIDTH / 2, HEIGHT / 2))
    
    pygame.mouse.set_visible(False)

    # core loop
    game_loop(camera)
    # quit
    pygame.quit()

"""
tri = [
    Vec3(0, -3, 0),
    Vec3(0, 0, 0),
    Vec3(0, 1, 0),
]
uv = [Vec3(0, 0, 0),
    Vec3(1, 0, 0),
    Vec3(2, -0, 0),]
flustered_girl = Image_Data("flustered_girl.png")
rasterize((tri, uv), flustered_girl, [])
"""