from vector import Vec3
import math as M
import pygame

class Camera():
    """A camera holding informations usefull to the view
    transform and which can be controlled by the user"""

    def __init__(self, pos, lam, phi, up):
        self.pos = pos # Vec3
        self.up = up.normalized() # Vec3
        self.speed = 10.0
        self.lam = lam # angle of orientation on xz plane, rad
        self.phi = phi # angle of orientation from xz plane, rad
        self.omega = M.radians(90) # rotate speed, rad/s

    def get_lookdir(self):
        return Vec3.from_angles(self.lam, self.phi)

    def rotate(self, dt, keys):
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

    def update(self, dt, yo):
        # that input polling method comes from chat gpt 
        keys = pygame.key.get_pressed()
        self.rotate(dt, keys)
        self.translate(dt, keys)