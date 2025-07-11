#sclone for ref
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

        m_earth = 1 / 332946.0487 # Earth's mass in solar masses

        
        self.add(name='Mercury', size=0.015, color='gray',
          m=0.0553 * m_earth, a=0.387098, e=0.205630,
          inc=np.radians(7.005), Omega=np.radians(48.331), omega=np.radians(29.124), f=np.radians(174))


        self.add(name='Venus', size=0.02, color='orange',
          m=0.815 * m_earth, a=0.723332, e=0.006772,
          inc=np.radians(3.39458), Omega=np.radians(76.680), omega=np.radians(54.884), f=np.radians(50))


        self.add(name='Earth', size=0.025, color='blue',
           m=1.0 * m_earth, a=1.000000, e=0.016710,
           inc=np.radians(0.000), Omega=np.radians(-11.26064), omega=np.radians(114.207), f=np.radians(100))


        self.add(name='Mars', size=0.018, color='red',
          m=0.107 * m_earth, a=1.523679, e=0.093400,
          inc=np.radians(1.850), Omega=np.radians(49.558), omega=np.radians(286.502), f=np.radians(150))


        self.add(name='Jupiter', size=0.05, color='orange',
         m=317.8 * m_earth, a=5.203, e=0.0484,
         inc=np.radians(1.3), Omega=np.radians(100.5), omega=np.radians(273.9), f=np.radians(20))


        self.add(name='Saturn', size=0.045, color='goldenrod',
         m=95.2 * m_earth, a=9.537, e=0.0541,
         inc=np.radians(2.5), Omega=np.radians(113.7), omega=np.radians(339.4), f=np.radians(317))


        self.add(name='Uranus', size=0.035, color='lightblue',
         m=14.5 * m_earth, a=19.191, e=0.0472,
         inc=np.radians(0.77), Omega=np.radians(74.0), omega=np.radians(96.7), f=np.radians(142)) #rem: make it 3d


        self.add(name='Neptune', size=0.035, color='blue',
         m=17.1 * m_earth, a=30.07, e=0.0086,
         inc=np.radians(1.77), Omega=np.radians(131.8), omega=np.radians(273.2), f=np.radians(256)) 
        
        '''self.add(name="doofs-planet", size=0.07, color="purple",
            m=400 * m_earth,  
            a=3.3,             
            e=0.3,             
            inc=np.radians(15),  # Inclined orbit — orbit not in same plane
            Omega=np.radians(80), omega=np.radians(60), f=np.radians(10))'''
        
        '''self.doof_params = {
               "name": "doofs-planet",
               "size": 0.07,
               "color": "purple",
               "m": 400 * m_earth,
               "a": 3.3,
               "e": 0.3,
               "inc": np.radians(15),
               "Omega": np.radians(80),
               "omega": np.radians(60),
               "f": np.radians(10)
              }'''
        self.add(**self.doof_params)
        print("[debug] bodies added:", [b.name for b in self.bodies])



        self.sim.move_to_com()
        self.t = 0  # (40000*np.pi)
        self.delta_t = (14 * np.pi) / 100# 7 * 2pi  - 7 revolutions of eath/s - 7 * fps years/s = 28e1 yrs/s
        print(self.delta_t)

    '''def update_doof_params(self, new_params):
      
      for key, val in new_params.items():
        if key in self.doof_params:
            self.doof_params[key] = val
    # Remove and re-add doof's planet
      self.bodies = [b for b in self.bodies if b.name != "doofs-planet"]
      self.sim = rebound.Simulation()
      self.sim.units = ('AU', 'yr', 'Msun')
      self._add_initial_bodies()  # Write a method to re-add other planets here if needed
      self.add(**self.doof_params)
      self.sim.move_to_com()
'''





    
    def add(self, name='noname', size=10, color='black', **kwargs):
        particle = rebound.Particle(simulation=self.sim, **kwargs)
        self.sim.add(particle)
        particle = self.sim.particles[-1]
        body = Body(name=name, size=size, color=color, particle=particle)
        self.bodies.append(body)



    def iterate(self):
        self.sim.integrate(self.t)
        self.t += self.delta_t

    def update_doof_params(self, new_params):
     try:
        print("[debug] Updating doof with:", new_params)
        for key, val in new_params.items():
            if key in self.doof_params:
                self.doof_params[key] = val

        self.sim = rebound.Simulation()
        self.sim.units = ('AU', 'yr', 'Msun')
        self.bodies.clear()
        self.__init__()
        print("[debug] Doof update complete.")
     except Exception as e:
        print("[CRASH IN update_doof_params]:", e)



if __name__ == '__main__':
    simulation = Simulation()
   
