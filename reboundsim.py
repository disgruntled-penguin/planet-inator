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

        
        
        

        self.add(name='Sun', size=0.1, color='yellow', m=1)

        m_earth = 1 / 333000  # Earth's mass in solar masses


        self.add(name='Mercury', size=0.015, color='gray', m=0.0553 * m_earth, a=0.387, e=0.2056, omega=0.0)
        self.add(name='Venus', size=0.02, color='orange', m=0.815 * m_earth, a=0.723, e=0.0067, omega=0.0)
        self.add(name='Earth', size=0.025, color='blue', m=1.0 * m_earth, a=1.0, e=0.0167, omega=0.0)
        self.add(name='Mars', size=0.018, color='red', m=0.107 * m_earth, a=1.524, e=0.0934, omega=0.0)
        self.add(name='Jupiter', size=0.05, color='orange', m=317.8 * m_earth, a=5.203, e=0.0484, omega=0.0)
        self.add(name='Saturn', size=0.045, color='goldenrod', m=95.2 * m_earth, a=9.537, e=0.0541, omega=0.0)
        self.add(name='Uranus', size=0.035, color='lightblue', m=14.5 * m_earth, a=19.191, e=0.0472, omega=0.0)
        self.add(name='Neptune', size=0.035, color='blue', m=17.1 * m_earth, a=30.07, e=0.0086, omega=0.0)
        #have to add omega and inclination, true anomaly 

        self.t = 0  # (40000*np.pi)
        self.delta_t = (16 * np.pi) / 1000
    
    def add(self, name='noname', size=10, color='black', **kwargs):
        particle = rebound.Particle(simulation=self.sim, **kwargs)
        self.sim.add(particle)
        particle = self.sim.particles[-1]
        body = Body(name=name, size=size, color=color, particle=particle)
        self.bodies.append(body)

    def iterate(self):
        self.sim.integrate(self.t)
        self.t += self.delta_t
