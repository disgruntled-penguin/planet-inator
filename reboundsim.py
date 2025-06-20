#switching to python
from dataclasses import dataclass

import numpy as np
import rebound

@dataclass
class Body:
    name: str
    size: float
    color: str
    particle: rebound.Particle

    @property
    def x(self):
        return self.particle.x

    @property
    def y(self):
        return self.particle.y
    
class Simulation:
    def __init__(self):
        self.sim = rebound.Simulation()
        self.sim.integrator = "whfast"
        self.sim.dt = 0.1

        self.bodies = []

        self.add(name='Star', size=0.1, color='yellow', m=1.)
    
    def add(self, name='noname', size=10, color='black', **kwargs):
        particle = rebound.Particle(simulation=self.sim, **kwargs)
        self.sim.add(particle)
        particle = self.sim.particles[-1]
        body = Body(name=name, size=size, color=color, particle=particle)
        self.bodies.append(body)