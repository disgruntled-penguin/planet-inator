import numpy as np
import rebound
from dataclasses import dataclass
from neo_loader import NEALoader  # Import your NEA loader

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
        
        m_earth = 1 / 332946.0487  # Earth's mass in solar masses
        self.doof_params = {
            "name":  "doofs-planet",
            "size":  0.07,
            "color": "purple",
            "m":     400 * m_earth,
            "a":     3.3,
            "e":     0.3,
            "inc":   np.radians(15),
            "Omega": np.radians(80),
            "omega": np.radians(60),
            "f":     np.radians(10)
        }

        # visibility
        self.doof_planet_created = False
        
        # Asteroid loading
        self.nea_loader = NEALoader('nea_extended.json')
        self.asteroids_loaded = False
        self.asteroid_count = 400  # Default number of asteroids to load
        
        self.sim = rebound.Simulation()
        self.sim.integrator = "whfast"
        self.sim.dt = 0.1

        self.bodies = []
        self._init_bodies()
        self.asteroids=[]


        self.sim.move_to_com()
        self.t = 0
        self.delta_t = (14 * np.pi) / (40*7) # 1 yr/sec # speed, 2*pi would mean 1 yr/sec but fps is 40 so *40 yrs

    def _init_bodies(self):
        # Reset rebound sim and bodies list
        self.sim = rebound.Simulation()
        self.sim.integrator = "whfast"
        self.sim.dt = 0.1
        self.bodies = []
        self.asteroids=[]

        m_earth = 1 / 332946.0487

        # Add Sun and planets
        self.add(name='Sun', size=0.1, color='yellow', m=1)
        self.add(name='Mercury', size=0.015, color='gray', m=0.0553 * m_earth,
                 a=0.387098, e=0.205630, inc=np.radians(7.005),
                 Omega=np.radians(48.331), omega=np.radians(29.124), f=np.radians(174))
        self.add(name='Venus', size=0.022, color='orange', m=0.815 * m_earth,
                 a=0.723332, e=0.006772, inc=np.radians(3.39458),
                 Omega=np.radians(76.680), omega=np.radians(54.884), f=np.radians(50))
        self.add(name='Earth', size=0.025, color='blue', m=1.0 * m_earth,
                 a=1.000000, e=0.016710, inc=np.radians(0.000),
                 Omega=np.radians(-11.26064), omega=np.radians(114.207), f=np.radians(100))
        self.add(name='Mars', size=0.018, color='red', m=0.107 * m_earth,
                 a=1.523679, e=0.093400, inc=np.radians(1.850),
                 Omega=np.radians(49.558), omega=np.radians(286.502), f=np.radians(150))
        self.add(name='Jupiter', size=0.05, color='orange', m=317.8 * m_earth,
                 a=5.203, e=0.0484, inc=np.radians(1.3),
                 Omega=np.radians(100.5), omega=np.radians(273.9), f=np.radians(20))
        self.add(name='Saturn', size=0.045, color='goldenrod', m=95.2 * m_earth,
                 a=9.537, e=0.0541, inc=np.radians(2.5),
                 Omega=np.radians(113.7), omega=np.radians(339.4), f=np.radians(317))
        self.add(name='Uranus', size=0.035, color='lightblue', m=14.5 * m_earth,
                 a=19.191, e=0.0472, inc=np.radians(0.77),
                 Omega=np.radians(74.0), omega=np.radians(96.7), f=np.radians(142))
        self.add(name='Neptune', size=0.035, color='blue', m=17.1 * m_earth,
                 a=30.07, e=0.0086, inc=np.radians(1.77),
                 Omega=np.radians(131.8), omega=np.radians(273.2), f=np.radians(256))

        # Only add Doof's planet if it has been created
        if self.doof_planet_created:
            self.add(**self.doof_params)

        # Add asteroids if they have been loaded
        if self.asteroids_loaded:
            self._add_asteroids()

        # Debug output
        print("[debug] Bodies in sim:", [b.name for b in self.bodies])

    def load_asteroids(self, count=200):
        """Load asteroids from JSON file and add them to simulation"""
        try:
            print(f"Loading {count} asteroids...")
            self.nea_loader.load_asteroids(limit=5000)  # Load more than we need for filtering
            
            # Get interesting asteroids first, then filter for suitable ones
            loaded_asteroids = self.nea_loader.get_asteroids(count)
            '''filtered_asteroids = self.nea_loader.filter_asteroids(
                max_count=count,
                min_size=1.0,  # H magnitude - lower means bigger 
                max_size=100.0,  
                orbit_types=['Amor', 'Apollo', 'Aten', 'Atira'],  # NEA orbit types
                min_period=0.1,  # orbital period in years
                max_period=10.0  
            )'''
            
            # Combine interesting and filtered asteroids
            #combined_asteroids = interesting_asteroids 
            # Remove duplicates by name
            seen_names = set()
            unique_asteroids = []
            for asteroid in loaded_asteroids:
                name = asteroid.get('Name', 'Unknown')
                if name not in seen_names:
                    seen_names.add(name)
                    unique_asteroids.append(asteroid)
            
            # Take only the requested count
            self.asteroid_data = unique_asteroids[:count]
            self.asteroid_count = len(self.asteroid_data)
            self.asteroids_loaded = True
            
            print(f"Successfully loaded {self.asteroid_count} asteroids")
            
            # If simulation is already running, reinitialize to include asteroids
            if len(self.bodies) > 0:
                print("Reinitializing simulation to include asteroids...")
                self._init_bodies()
                self.sim.move_to_com()
                self.t = 0
                
        except Exception as e:
            print(f"Error loading asteroids: {e}")
            self.asteroids_loaded = False

    def _add_asteroids(self):
        """Add loaded asteroids to the simulation"""
        if not self.asteroids_loaded:
            return
            
        print(f"Adding {len(self.asteroid_data)} asteroids to simulation...")
        for asteroid_data in self.asteroid_data:
            try:
                # Convert asteroid data to rebound parameters
                params = self.nea_loader.convert_to_rebound_params(asteroid_data)
                
                # Scale up the visual size to make asteroids visible
                # Make them bigger than the original calculation
                visual_size = max(0.008, params['size'] * 5)  # At least 0.008, scaled up 5x
                
                # Add asteroid to simulation
                self.add_neo(
                    name=params['name'],
                    size=visual_size,
                    color= params['color'], #'aqua',
                    m=params['m'],
                    a=params['a'],
                    e=params['e'],
                    inc=params['inc'],
                    Omega=params['Omega'],
                    omega=params['omega'],
                    f=params['f']
                )
                
            except Exception as e:
                print(f"Error adding asteroid {asteroid_data.get('Name', 'Unknown')}: {e}")
                continue

    def add(self, name='noname', size=10, color='black', **kwargs):
        # Add a particle to the simulation and wrap it
        self.sim.add(**kwargs)
        # rebound.Simulation.add returns None; retrieve the newly added particle
        particle = self.sim.particles[-1]
        body = Body(name=name, size=size, color=color, particle=particle)
        self.bodies.append(body)
    
    def add_neo(self, name='noname', size=10, color='black', **kwargs):
        # Add a particle to the simulation and wrap it
        self.sim.add(**kwargs)
        # rebound.Simulation.add returns None; retrieve the newly added particle
        particle = self.sim.particles[-1]
        asteroid = Body(name=name, size=size, color=color, particle=particle)
        self.asteroids.append(asteroid)
    

    def create_doof_planet(self):
     
        if not self.doof_planet_created:
            self.doof_planet_created = True
            # Add the planet to the current simulation
            self.add(**self.doof_params)
            self.sim.move_to_com()
            print("[debug] Doof's planet created for the first time!")
        else:
            print("[debug] Doof's planet already exists!")

    def update_doof_params(self, new_params):
        #trigger spock prediction
        # Update stored parameters
        for key, val in new_params.items():
            if key in self.doof_params:
                self.doof_params[key] = val

        # If Doof's planet hasn't been created yet, create it
        if not self.doof_planet_created:
            self.create_doof_planet()
        else:
            # Rebuild simulation to apply new parameters
            self._init_bodies()
            self.sim.move_to_com()
            # Reset time
            self.t = 0
            print("[debug] Doof's planet updated with new parameters:", self.doof_params)

    def get_spock_ready_simulation(self):
        
        sim_copy = self.sim.copy()
        
        sim_copy.move_to_com()
        
       
        sim_copy.integrator = "whfast"
        sim_copy.dt = 0.1
        
        print(f"spock simulation with {sim_copy.N} particles")
        return sim_copy

    def iterate(self):
        self.sim.integrate(self.t) #sim one step at a time
        self.t += self.delta_t

if __name__ == '__main__':
    sim = Simulation()
   
    print("Testing SPOCK integration...")
    
    # Test asteroid loading
    sim.load_asteroids(count=500)
    
    sim.create_doof_planet()
   
    test_params = {
        "a": 2.5,
        "e": 0.1,
        "inc": np.radians(10),
        "m": 200 * (1 / 332946.0487)
    }
    
    sim.update_doof_params(test_params)
    
    
    spock_sim = sim.get_spock_ready_simulation()
    
    print(f"Simulation ready for SPOCK with {spock_sim.N} bodies")
    print("Integration test...")
    
    # Run a few iterations
    for i in range(10):
        sim.iterate()
        if i % 3 == 0:
            print(f"Iteration {i}: t = {sim.t:.3f}")
    
    print("Simulation test complete!")