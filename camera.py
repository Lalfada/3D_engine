from vector import Vec3
import math as M
import pygame

def get_mouse_pos():
    (mouse_x, mouse_y) = pygame.mouse.get_pos()
    return Vec3(mouse_x, mouse_y, 0) 

class Camera():
    """A camera holding informations usefull to the view
    transform and which can be controlled by the user"""


    def __init__(self, pos, lam, phi, up, mouse_pos):
        self.pos = pos # Vec3
        self.up = up.normalized() # Vec3
        self.speed = 10.0
        self.lam = lam # angle of orientation on xz plane, rad
        self.phi = phi # angle of orientation from xz plane, rad
        self.omega = M.radians(90) # rotate speed, rad/s
        self.mouse_pos = mouse_pos

    def get_lookdir(self):
        return Vec3.from_angles(self.lam, self.phi)
    
    def mouse_rotate(self, dt, keys):
        new_mouse_pos = pygame.mouse.get_pos()
        dx = new_mouse_pos[0] - self.mouse_pos[0]
        pygame.mouse.set_pos(self.mouse_pos)

        self.lam += dx * dt * -1e-1
    
    def rotate(self, dt, keys):
        self.mouse_rotate(dt, keys)
        
        delta_angle = 0
        if keys[pygame.K_LEFT]:
            self.lam += self.omega * dt
        if keys[pygame.K_RIGHT]:
            self.lam -= self.omega * dt
        
        self.lam %= M.pi * 2.0

    def translate(self, dt, keys):
        speed = Vec3(0, 0, 0)
        lookdir = self.get_lookdir()
        right = self.up.cross(lookdir).normalized()
        
        # y axis is facing down
        if keys[pygame.K_UP] or keys[pygame.K_SPACE]:
            speed -= self.up
        if keys[pygame.K_DOWN] or keys[pygame.K_LALT]:
            speed += self.up

        if keys[pygame.K_d]:
            speed += right
        if keys[pygame.K_a]:
            speed -= right

        if keys[pygame.K_w]:
            speed += lookdir
        if keys[pygame.K_s]:
            speed -= lookdir

        self.pos += speed.normalized() * self.speed * dt

    def update(self, dt):
        # that input polling method comes from chat gpt 
        keys = pygame.key.get_pressed()
        self.rotate(dt, keys)
        self.translate(dt, keys)