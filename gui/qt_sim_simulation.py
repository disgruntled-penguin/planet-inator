import pygame
import numpy as np
import math

class QtSimSimulation:
    def __init__(self, width=600, height=600):
        self.width = width
        self.height = height
        pygame.init()
        self.surface = pygame.Surface((self.width, self.height))
        self.clock = pygame.time.Clock()
        # Default Doofs Planet parameters
        self.doofs_params = {
            'mass': 400,
            'size': 0.07,
            'color': (128, 0, 128),
            'a': 3.3,
            'e': 0.3,
            'inc': 15,
            'Omega': 80,
            'omega': 60,
            'f': 10
        }
        self.t = 0
        self.dt = 0.01

    def update_doofs(self, params):
        self.doofs_params.update(params)

    def step(self):
        self.t += self.dt

    def draw(self):
        self.surface.fill((0, 0, 0))
        # Draw a simple orbit and planet for Doofs Planet
        cx, cy = self.width // 2, self.height // 2
        a = self.doofs_params['a'] * 60
        e = self.doofs_params['e']
        angle = math.radians(self.doofs_params['f']) + self.t
        x = cx + a * math.cos(angle)
        y = cy + a * math.sin(angle) * math.sqrt(1 - e**2)
        # Orbit
        pygame.draw.ellipse(self.surface, (100, 100, 100), (cx - a, cy - a * math.sqrt(1 - e**2), 2*a, 2*a*math.sqrt(1 - e**2)), 1)
        # Planet
        size = int(self.doofs_params['size'] * 30)
        color = self.doofs_params['color']
        pygame.draw.circle(self.surface, color, (int(x), int(y)), size)

    def get_surface_array(self):
        # Returns a numpy array (height, width, 3) for QImage
        return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2)) 