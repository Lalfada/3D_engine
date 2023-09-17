from vector import Vec3
import pygame

class Camera():
    """A camera holding informations usefull to the view
    transform and which can be controlled by the user"""

    def __init__(self, pos, look_dir, up):
        self.pos = pos # Vec3
        self.look_dir = look_dir # Vec3
        self.up = up # Vec3
        self.speed = 5.0

    def update(self, dt):
        # that input polling method comes from chat gpt 
        keys = pygame.key.get_pressed()
        speed = Vec3(0, 0, 0)
        
        if keys[pygame.K_UP] or keys[pygame.K_SPACE]:
            speed.y -= 1
        if keys[pygame.K_DOWN] or keys[pygame.K_LALT]:
            speed.y += 1

        if keys[pygame.K_d]:
            speed.x += 1
        if keys[pygame.K_a]:
            speed.x -= 1

        if keys[pygame.K_w]:
            speed.z += 1
        if keys[pygame.K_s]:
            speed.z -= 1

        self.pos += speed.normalized() * self.speed * dt