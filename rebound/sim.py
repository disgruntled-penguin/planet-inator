import numpy as np
import rebound
import math
from dataclasses import dataclass
from rebound.asteroid_loader import NEALoader

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
        
        self.nea_loader = NEALoader('minor_planets/nea_extended.json')
        self.distant_loader = NEALoader('minor_planets/distant_extended.json')  
        
        self.asteroids_loaded = {'nea': False, 'distant': False}  
        self.asteroid_counts = {'nea': 200, 'distant': 100}  
        self.asteroid_data = {'nea': [], 'distant': []}  
        
        
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

    def load_asteroids(self, nea_count=200, distant_count=0):
     """Load asteroids from JSON files and add them to simulation"""
     try:
        total_loaded = 0
        all_asteroids = []
        
        # Load NEAs if requested
        if nea_count > 0:
            print(f"Loading {nea_count} NEA asteroids...")
            self.nea_loader.load_asteroids(limit=5000)
            nea_asteroids = self.nea_loader.get_asteroids(nea_count)
            
            # Handle both string and dict 
            for asteroid in nea_asteroids:
                if isinstance(asteroid, dict):
                    asteroid['_type'] = 'nea'
                    all_asteroids.append(asteroid)
                else:
                    # If string, create a simple dict wrapper
                    all_asteroids.append({'Name': asteroid, '_type': 'nea', '_raw': asteroid})
            total_loaded += len(nea_asteroids)
        
        # Load distant asteroids if requested
        if distant_count > 0:
            print(f"Loading {distant_count} distant asteroids...")
            self.distant_loader.load_asteroids(limit=5000)
            distant_asteroids = self.distant_loader.get_asteroids(distant_count)
            
            # Handle both string and dict formats
            for asteroid in distant_asteroids:
                if isinstance(asteroid, dict):
                    asteroid['_type'] = 'distant'
                    all_asteroids.append(asteroid)
                else:
                    # If it's a string, create a simple dict wrapper
                    all_asteroids.append({'Name': asteroid, '_type': 'distant', '_raw': asteroid})
            total_loaded += len(distant_asteroids)
        
        # Remove duplicates by name across both types
        seen_keys = set()
        unique_asteroids = []
        for asteroid in all_asteroids:
         name = asteroid.get('Name', 'Unknown')
    # Add orbital parameter to make key unique
         a = asteroid.get('a', 0)  # semi-major axis
         key = f"{name}_{a}"
         if key not in seen_keys:
          seen_keys.add(key)
          unique_asteroids.append(asteroid)
        '''seen_names = set()
        for asteroid in all_asteroids:
            name = asteroid.get('Name', 'Unknown')
            if name not in seen_names:
                seen_names.add(name)'''
        
        # Store the combined data
        self.asteroid_data = unique_asteroids
        self.asteroid_count = len(self.asteroid_data)
        self.asteroids_loaded = True
        
        print(f"Successfully loaded {self.asteroid_count} total asteroids")
        
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
        # Set default values outside try block
        asteroid_name = 'Unknown'
        asteroid_type = 'nea'
        
        try:
            # Handle both string and dict formats
            if isinstance(asteroid_data, str):
                asteroid_name = asteroid_data
                asteroid_type = 'nea'
                data_to_convert = asteroid_data
            else:
                # It's a dictionary
                asteroid_name = asteroid_data.get('Name', 'Unknown')
                asteroid_type = asteroid_data.get('_type', 'nea')
                data_to_convert = asteroid_data.get('_raw', asteroid_data)
            
            loader = self.distant_loader if asteroid_type == 'distant' else self.nea_loader
            
            # Convert asteroid data to rebound parameters
            params = loader.convert_to_rebound_params(data_to_convert)
            
            # Adjust visual properties based on type
            if asteroid_type == 'distant':
                visual_size = max(0.01, params['size'] * 3)  # Smaller for distant objects
                color = params.get('color', 'darkblue')  
            else:  # nea
                visual_size = max(0.008, params['size'] * 5)
                color = params.get('color', 'aqua')
            
            # Add asteroid to simulation
            self.add_neo(
                name=f"{asteroid_type}_{params['name']}",
                size=visual_size,
                color=color,
                m=params['m'],
                a=params['a'],
                e=params['e'],
                inc=params['inc'],
                Omega=params['Omega'],
                omega=params['omega'],
                f=params['f']
            )
            
        except Exception as e:
            print(f"Error adding asteroid {asteroid_name}: {e}")
            continue
    
    def get_body_info(self, body_index):
     if body_index >= len(self.bodies):
        return None
    
     body = self.bodies[body_index]
     info = {
        'name': body.name,
        'size': body.size,
        'position': (body.x, body.y),
        'mass': f"{(body.particle.m * 332946.0487):.3f} Earth Masses"
     }
    
    # planetary data dictionary
     planetary_data = {
        'Sun': {
            'type': 'star',
            'radius': '696,340 km',
            'surface_temperature': '5,778 K',
            'spectral_class': 'G2V',
            'luminosity': '3.828 × 10²⁶ W',
            'composition': 'Hydrogen (~73%), Helium (~25%)',
            'rotation_period': '25.05 days (equatorial)'
        },
        'Mercury': {
            'radius': '2,439.7 km',
            'surface_temperature': '167°C (average)',
            'atmosphere': 'Extremely thin (oxygen, sodium, hydrogen)',
            'rotation_period': '58.65 days',
            'surface_gravity': '3.7 m/s²',
            'moons': 0,
            'composition': 'Iron core, silicate mantle'
        },
        'Venus': {
            'radius': '6,051.8 km',
            'surface_temperature': '464°C',
            'atmosphere': 'Dense CO₂ (96.5%), N₂ (3.5%)',
            'rotation_period': '243.02 days (retrograde)',
            'surface_gravity': '8.87 m/s²',
            'moons': 0,
            'composition': 'Iron core, silicate mantle'
        },
        'Earth': {
            'radius': '6,371 km',
            'surface_temperature': '15°C (average)',
            'atmosphere': 'N₂ (78%), O₂ (21%), Ar (0.93%)',
            'rotation_period': '23.93 hours',
            'surface_gravity': '9.81 m/s²',
            'moons': 1,
            'composition': 'Iron core, silicate mantle, water oceans'
        },
        'Mars': {
            'radius': '3,389.5 km',
            'surface_temperature': '-65°C (average)',
            'atmosphere': 'Thin CO₂ (95.3%), N₂ (2.7%), Ar (1.6%)',
            'rotation_period': '24.62 hours',
            'surface_gravity': '3.71 m/s²',
            'moons': 2,
            'composition': 'Iron core, basaltic crust'
        },
        'Jupiter': {
            'radius': '69,911 km',
            'surface_temperature': '-110°C (cloud tops)',
            'atmosphere': 'H₂ (89%), He (10%), traces of methane',
            'rotation_period': '9.93 hours',
            'surface_gravity': '24.79 m/s²',
            'moons': 95,
            'composition': 'Gas giant - mostly hydrogen and helium'
        },
        'Saturn': {
            'radius': '58,232 km',
            'surface_temperature': '-140°C (cloud tops)',
            'atmosphere': 'H₂ (96%), He (3%), traces of methane',
            'rotation_period': '10.66 hours',
            'surface_gravity': '10.44 m/s²',
            'moons': 146,
            'composition': 'Gas giant with prominent ring system'
        },
        'Uranus': {
            'radius': '25,362 km',
            'surface_temperature': '-195°C',
            'atmosphere': 'H₂ (83%), He (15%), methane (2%)',
            'rotation_period': '17.24 hours (retrograde)',
            'surface_gravity': '8.69 m/s²',
            'moons': 28,
            'composition': 'Ice giant with tilted axis (98°)'
        },
        'Neptune': {
            'radius': '24,622 km',
            'surface_temperature': '-200°C',
            'atmosphere': 'H₂ (80%), He (19%), methane (1%)',
            'rotation_period': '16.11 hours',
            'surface_gravity': '11.15 m/s²',
            'moons': 16,
            'composition': 'Ice giant with strong winds'
        }
    }
    

     if body.name in planetary_data:
        info.update(planetary_data[body.name])
    
    # Handle Sun separately (no orbital parameters)
     if body_index == 0:
        info.update({'type': 'star'})
        return info
    
    # For planets and other orbiting bodies, add orbital mechanics
     if hasattr(body.particle, 'a'):
        # Basic orbital elements
        semi_major_axis = body.particle.a
        eccentricity = body.particle.e
        inclination_rad = body.particle.inc
        inclination_deg = np.degrees(inclination_rad)
        
        # Derived orbital parameters
        orbital_period_years = semi_major_axis**1.5
        orbital_period_days = orbital_period_years * 365.25
        
        # Perihelion and aphelion distances
        perihelion = semi_major_axis * (1 - eccentricity)
        aphelion = semi_major_axis * (1 + eccentricity)
        
        # Orbital velocity calculations (approximate)
        # Mean orbital velocity: v = 2π * a / T
        mean_orbital_velocity = (2 * np.pi * semi_major_axis * 149597870.7) / (orbital_period_days * 24 * 3600)  # km/s
        
        # Perihelion and aphelion velocities using vis-viva equation
        # v = sqrt(GM(2/r - 1/a)) where GM_sun ≈ 1.327124400e11 km³/s²
        GM_sun = 1.327124400e11  # km³/s²
        v_perihelion = math.sqrt(GM_sun * (2/(perihelion * 149597870.7) - 1/(semi_major_axis * 149597870.7)))
        v_aphelion = math.sqrt(GM_sun * (2/(aphelion * 149597870.7) - 1/(semi_major_axis * 149597870.7)))
        
        # Mean motion (degrees per day)
        mean_motion = 360.0 / orbital_period_days
        
        # Orbital circumference
        # Approximation for elliptical orbit: C ≈ π * [3(a+b) - sqrt((3a+b)(a+3b))]
        # where b = a * sqrt(1 - e²)
        b = semi_major_axis * math.sqrt(1 - eccentricity**2)  # semi-minor axis
        orbital_circumference = np.pi * (3*(semi_major_axis + b) - 
                                       math.sqrt((3*semi_major_axis + b) * (semi_major_axis + 3*b)))
        orbital_circumference_km = orbital_circumference * 149597870.7
        
        # Angular momentum per unit mass (specific angular momentum)
        # h = sqrt(GM * a * (1 - e²))
        specific_angular_momentum = math.sqrt(GM_sun * semi_major_axis * 149597870.7 * (1 - eccentricity**2))
        
        # Escape velocity at perihelion
        escape_velocity_perihelion = math.sqrt(2 * GM_sun / (perihelion * 149597870.7))
        
        # Hill sphere radius (approximate, assumes circular orbit)
        # R_Hill ≈ a * (m_planet / 3*m_sun)^(1/3)
        if body.particle.m > 0:
            hill_sphere_radius = semi_major_axis * ((body.particle.m / 3.0)**(1/3))
        else:
            hill_sphere_radius = 0
        
        # info with orbital mechanics
        info.update({
            'type': 'planet',
            'semi_major_axis': f"{semi_major_axis:.6f} AU",
            'eccentricity': f"{eccentricity:.6f}",
            'inclination': f"{inclination_deg:.3f}°",
            'inclination_rad': f"{inclination_rad:.6f} rad",
            'orbital_period': f"{orbital_period_years:.2f} years",
            'orbital_period_days': f"{orbital_period_days:.1f} days",
            'perihelion_distance': f"{perihelion:.6f} AU ({perihelion * 149.6:.1f} million km)",
            'aphelion_distance': f"{aphelion:.6f} AU ({aphelion * 149.6:.1f} million km)",
            'orbital_eccentricity_type': 'circular' if eccentricity < 0.05 else 'elliptical' if eccentricity < 0.9 else 'highly elliptical',
            'mean_orbital_velocity': f"{mean_orbital_velocity:.2f} km/s",
            'perihelion_velocity': f"{v_perihelion:.2f} km/s",
            'aphelion_velocity': f"{v_aphelion:.2f} km/s",
            'velocity_variation': f"{((v_perihelion - v_aphelion) / mean_orbital_velocity * 100):.1f}%",
            'mean_motion': f"{mean_motion:.6f} °/day",
            'orbital_circumference': f"{orbital_circumference:.3f} AU ({orbital_circumference_km/1e9:.1f} billion km)",
            'specific_angular_momentum': f"{specific_angular_momentum:.2e} km²/s",
            'escape_velocity_at_perihelion': f"{escape_velocity_perihelion:.2f} km/s",
            'hill_sphere_radius': f"{hill_sphere_radius:.6f} AU" if hill_sphere_radius > 0 else "N/A"
        })
        
        # derived facts
        facts = []
        
        if eccentricity > 0.1:
            speed_diff = v_perihelion - v_aphelion
            facts.append(f"Travels {speed_diff:.1f} km/s faster at closest approach to Sun")
        
        if inclination_deg > 5:
            facts.append(f"Orbit tilted {inclination_deg:.1f}° from Earth's orbital plane")
        
        if orbital_period_years < 1:
            facts.append(f"Completes orbit in {orbital_period_days:.0f} days")
        elif orbital_period_years > 10:
            facts.append(f"Takes {orbital_period_years:.0f} years to complete one orbit")
        
        # Compare to Earth
        if body.name != 'Earth':
            earth_distance = 1.0  # AU
            distance_comparison = semi_major_axis / earth_distance
            if distance_comparison > 1:
                facts.append(f"{distance_comparison:.1f}x farther from Sun than Earth")
            else:
                facts.append(f"{1/distance_comparison:.1f}x closer to Sun than Earth")
        
        if facts:
            info['interesting_facts'] = facts
    
    
     if body.name == "doofs-planet":
        info.update({
            'type': 'planet',
            'origin': "Dr. Doofenshmirtz's creation",
            'description': 'Purple coloration (unknown composition)',
        })
    
     return info


    def get_asteroid_info(self, asteroid_index):
        if asteroid_index >= len(self.asteroids):
            return None
        
        asteroid = self.asteroids[asteroid_index]
        original_data = self.asteroid_data[asteroid_index] if asteroid_index < len(self.asteroid_data) else None
        
        info = {
            'name': asteroid.name,
            'type': 'asteroid',
            'position': (asteroid.x, asteroid.y),
        }
        
        if hasattr(asteroid.particle, 'a'):
            info.update({
                'semi_major_axis': f"{asteroid.particle.a:.4f} AU",
                'eccentricity': f"{asteroid.particle.e:.4f}",
                'inclination': f"{np.degrees(asteroid.particle.inc):.2f}°",
                'orbital_period': f"{(asteroid.particle.a**1.5):.2f} years",
            })
        
        if original_data and isinstance(original_data, dict):
            data_map = {
                'Orbit_type': 'orbit_type',
                'Orbital_period': ('orbital_period_precise', lambda x: f"{x:.4f} years"),
                'Synodic_period': ('synodic_period', lambda x: f"{x:.4f} years"),
                'Perihelion_dist': ('perihelion_distance', lambda x: f"{x:.4f} AU"),
                'Aphelion_dist': ('aphelion_distance', lambda x: f"{x:.4f} AU"),
                'H': ('absolute_magnitude', lambda x: f"{x:.2f}"),
                'G': ('slope_parameter', lambda x: f"{x:.2f}"),
                'Number': 'catalog_number',
                'Principal_desig': 'principal_designation',
                'Last_obs': 'last_observation',
                'Arc_years': 'observation_arc',
                'Num_obs': 'total_observations',
                'Num_opps': 'observation_oppositions',
                'rms': ('rms_residual', lambda x: f"{x:.2f} arcsec"),
                'M': ('mean_anomaly', lambda x: f"{x:.5f}°"),
                'Peri': ('argument_of_periapsis', lambda x: f"{x:.5f}°"),
                'Node': ('longitude_ascending_node', lambda x: f"{x:.5f}°"),
                'n': ('mean_motion', lambda x: f"{x:.8f} °/day"),
                'Epoch': ('epoch', lambda x: f"JD {x}"),
                'Tp': ('time_of_periapsis', lambda x: f"JD {x:.5f}"),
                'Semilatus_rectum': ('semilatus_rectum', lambda x: f"{x:.4f} AU"),
            }
            
            for key, mapping in data_map.items():
                if key in original_data:
                    if isinstance(mapping, tuple):
                        info[mapping[0]] = mapping[1](original_data[key])
                    else:
                        info[mapping] = original_data[key]
            
            if 'H' in original_data:
                diameter_km = 1329.0 / math.sqrt(0.25) * 10**(-0.2 * original_data['H'])
                info['estimated_diameter'] = f"{diameter_km:.2f} km"
            
            if 'U' in original_data:
                uncertainty_map = {'0': 'Very well known', '1': 'Well known', '2': 'Good', 
                                 '3': 'Fair', '4': 'Poor'}
                info['orbit_uncertainty'] = uncertainty_map.get(original_data['U'], f"Quality code: {original_data['U']}")
            
            if original_data.get('NEO_flag'):
                info['near_earth_object'] = "Yes"
            
            if original_data.get('One_km_NEO_flag'):
                info['potentially_hazardous'] = "Yes (>1km diameter)"
        
        return info

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
          
            self.add(**self.doof_params)
            self.sim.move_to_com()
            print("[debug] Doof's planet created for the first time!")
        else:
            print("[debug] Doof's planet already exists!")

    def update_doof_params(self, new_params):
        #trigger spock prediction
      
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

    
   
    ''' for i in range(10):
        sim.iterate()
        if i % 3 == 0:
            print(f"Iteration {i}: t = {sim.t:.3f}")'''
    
  