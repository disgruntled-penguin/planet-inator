import numpy as np
import rebound
from dataclasses import dataclass
from nea import NEALoader  # Import our NEA loader
from typing import Dict, Optional

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
        self.asteroids_added = False
        
        self.sim = rebound.Simulation()
        self.sim.integrator = "whfast"
        self.sim.dt = 0.1

        self.bodies = []
        self.nea_loader = None  # Will hold our NEA loader
        self._init_bodies()

        self.sim.move_to_com()
        self.t = 0
        self.delta_t = (14 * np.pi) / (40*7) # 1 yr/sec

    def _init_bodies(self):
        # Reset rebound sim and bodies list
        self.sim = rebound.Simulation()
        self.sim.integrator = "whfast"
        self.sim.dt = 0.1
        self.bodies = []

        m_earth = 1 / 332946.0487

        # Add Sun and planets
        self.add(name='Sun', size=0.1, color='yellow', m=1)
        self.add(name='Mercury', size=0.015, color='gray', m=0.0553 * m_earth,
                 a=0.387098, e=0.205630, inc=np.radians(7.005),
                 Omega=np.radians(48.331), omega=np.radians(29.124), f=np.radians(174))
        self.add(name='Venus', size=0.02, color='orange', m=0.815 * m_earth,
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

        # Debug output
        print("[debug] Bodies in sim:", [b.name for b in self.bodies])

    def add(self, name='noname', size=10, color='black', **kwargs):
        # Add a particle to the simulation and wrap it
        self.sim.add(**kwargs)
        # rebound.Simulation.add returns None; retrieve the newly added particle
        particle = self.sim.particles[-1]
        body = Body(name=name, size=size, color=color, particle=particle)
        self.bodies.append(body)

    def load_nea_data(self, json_file_path: str, limit: Optional[int] = 1000):
        """Load NEA data from JSON file"""
        self.nea_loader = NEALoader(json_file_path)
        return self.nea_loader.load_asteroids(limit=limit)

    def add_asteroids(self, max_count: int = 50, filter_params: Optional[Dict] = None, interesting_only: bool = False):
        """Add asteroids to the simulation"""
        
        if not self.nea_loader or not self.nea_loader.loaded:
            print("Error: NEA data not loaded. Call load_nea_data() first.")
            return
        
        # Get filtered asteroids
        if interesting_only:
            selected_asteroids = self.nea_loader.get_interesting_asteroids(max_count)
        else:
            filter_params = filter_params or {}
            selected_asteroids = self.nea_loader.filter_asteroids(max_count=max_count, **filter_params)
        
        # Convert and add each asteroid
        added_count = 0
        for asteroid_data in selected_asteroids:
            try:
                asteroid_params = self.nea_loader.convert_to_rebound_params(asteroid_data)
                self.add(**asteroid_params)
                added_count += 1
            except Exception as e:
                print(f"Error adding asteroid {asteroid_data.get('Name', 'Unknown')}: {e}")
                continue
        
        print(f"Successfully added {added_count} asteroids to simulation")
        self.asteroids_added = True
        self.sim.move_to_com()

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

    def get_simulation_stats(self):
        """Get statistics about the current simulation"""
        stats = {
            'total_bodies': len(self.bodies),
            'current_time': self.t,
            'has_doof_planet': self.doof_planet_created,
            'has_asteroids': self.asteroids_added
        }
        return stats

if __name__ == '__main__':
    sim = Simulation()
    
    # Example usage with NEA data
    print("Testing NEA integration...")
    
    # Load NEA data (replace with your actual file path)
    nea_file = "nea_extended.json"  # Replace with your JSON file path
    
    try:
        loaded_count = sim.load_nea_data(nea_file, limit=500)  # Load first 500 asteroids
        print(f"Loaded {loaded_count} asteroids")
        
        # Add some interesting asteroids
        sim.add_asteroids(max_count=20, interesting_only=True)
        
        # Create Doof's planet
        sim.create_doof_planet()
        
        # Print stats
        stats = sim.get_simulation_stats()
        print("Simulation stats:", stats)
        
        # Run simulation
        for i in range(10):
            sim.iterate()
            if i % 3 == 0:
                print(f"Iteration {i}: t = {sim.t:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to update 'nea_file' with your actual JSON file path")
    
    print("Simulation test complete!")