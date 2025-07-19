import json
import math
import numpy as np
from typing import List, Dict, Optional

class NEALoader:
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.asteroids = []
        self.loaded = False
    
    def load_asteroids(self, limit: Optional[int] = None):
        print(f"Loading asteroids from {self.json_file_path}...")
        
        try:
            with open(self.json_file_path, 'r') as f:
                data = json.load(f)
                
            if limit:
                self.asteroids = data[:limit]
                print(f"Loaded {len(self.asteroids)} asteroids (limited)")
            else:
                self.asteroids = data
                print(f"Loaded {len(self.asteroids)} asteroids")
            
            # Print sample data automatically after loading
            print(f"\nFirst 3 asteroids:")
            for i, asteroid in enumerate(self.asteroids[:3]):
                name = asteroid.get('Name', 'Unknown')
                number = asteroid.get('Number', 'N/A')
                h_mag = asteroid.get('H', 'N/A')
                orbit_type = asteroid.get('Orbit_type', 'Unknown')
                print(f"  {i+1}. {number} {name} (H={h_mag}, Type={orbit_type})")
                
            self.loaded = True
            return len(self.asteroids)
            
        except FileNotFoundError:
            print(f"Error: Could not find file {self.json_file_path}")
            return 0
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {self.json_file_path}")
            return 0
    
    def print_sample_data(self, num_samples: int = 150):
        if not self.loaded:
            print("Error: Asteroids not loaded yet. Call load_asteroids() first.")
            return
        if not self.asteroids:
            print("No asteroids found in the loaded data.")
            return
        
        print(f"\n--- Sample Data ({min(num_samples, len(self.asteroids))} asteroids) ---")
        
        for i, asteroid in enumerate(self.asteroids[:num_samples]):
            print(f"\nAsteroid {i+1}:")
            print(f"  Name: {asteroid.get('Name', 'Unknown')}")
            print(f"  Number: {asteroid.get('Number', 'N/A')}")
            print(f"  H magnitude: {asteroid.get('H', 'N/A')}")
            print(f"  Semi-major axis (a): {asteroid.get('a', 'N/A')} AU")
            print(f"  Eccentricity (e): {asteroid.get('e', 'N/A')}")
            print(f"  Inclination (i): {asteroid.get('i', 'N/A')}°")
            print(f"  Orbit type: {asteroid.get('Orbit_type', 'Unknown')}")
            print(f"  Orbital period: {asteroid.get('Orbital_period', 'N/A')} years")
            print(f"  Potentially hazardous: {asteroid.get('One_km_NEO_flag', 'N/A')}")
            
            # Show all available keys for the first asteroid
            if i == 0:
                print(f"  Available keys: {list(asteroid.keys())}")
        
        print(f"\nTotal asteroids loaded: {len(self.asteroids)}")
    
    def filter_asteroids(self, 
                        max_count: int = 10000,
                        min_size: Optional[float] = None,
                        max_size: Optional[float] = None,
                        orbit_types: Optional[List[str]] = None,
                        min_period: Optional[float] = None,
                        max_period: Optional[float] = None,
                        
                        potentially_hazardous_only: bool = False):
        """Filter asteroids based on various criteria"""
        
        if not self.loaded:
            print("Error: Asteroids not loaded yet. Call load_asteroids() first.")
            return []
        
        filtered = []
        
        for asteroid in self.asteroids:
            # Size filtering (using H magnitude - lower H means bigger asteroid)
            if min_size is not None and asteroid.get('H', 999) > min_size:
                continue
            if max_size is not None and asteroid.get('H', 0) < max_size:
                continue
            
            # Orbit type filtering
            if orbit_types and asteroid.get('Orbit_type') not in orbit_types:
                continue
            
            # Period filtering
            period = asteroid.get('Orbital_period', 0)
            if min_period is not None and period < min_period:
                continue
            if max_period is not None and period > max_period:
                continue
            
            # Potentially Hazardous Object filtering
            if potentially_hazardous_only and not asteroid.get('One_km_NEO_flag', 0):
                continue
            
            filtered.append(asteroid)
            
            # Stop if we've reached the limit
            if len(filtered) >= max_count:
                break
        
        print(f"Filtered to {len(filtered)} asteroids")
        return filtered
    
    def get_interesting_asteroids(self, count: int = 500):
        if not self.loaded:
            self.load_asteroids(limit=10000)  # Load a subset first
        

        regular = []
        
        for asteroid in self.asteroids:
           
                regular.append(asteroid)
        
          
        return regular[:count]
    
    def convert_to_rebound_params(self, asteroid_data: Dict):
        """Convert asteroid JSON data to rebound parameters"""
        
        # Extract orbital elements
        a = asteroid_data.get('a')  # Semi-major axis in AU
        e = asteroid_data.get('e')  # Eccentricity
        inc = np.radians(asteroid_data.get('i'))  # Inclination
        Omega = np.radians(asteroid_data.get('Node'))  # Longitude of ascending node
        omega = np.radians(asteroid_data.get('Peri'))  # Argument of periapsis
        M = np.radians(asteroid_data.get('M'))  # Mean anomaly
        
        # Convert mean anomaly to true anomaly (approximate)
        f = M  # Simple approximation - for more accuracy, solve Kepler's equation
        
        # Estimate mass from H magnitude (very rough approximation)
        H = asteroid_data.get('H')
        estimated_diameter_km = 1329.0 / math.sqrt(0.25) * 10**(-0.2 * H)  # removed albedo
        estimated_mass_kg = (4/3) * math.pi * (estimated_diameter_km * 500)**3 * 2000  # Assume 2 g/cc density
        m_earth_kg = 5.972e24
        mass_earth_units = estimated_mass_kg / m_earth_kg
        mass_solar_units =  mass_earth_units / 332946.0487 #wrt sun mass
        
        # Visual properties
        name = asteroid_data.get('Name', 'Unknown')
        if asteroid_data.get('Number'):
            name = f"{asteroid_data['Number']} {name}"
        
        # Size for visualization (scale with H magnitude)
        visual_size = 0.1 * estimated_diameter_km/1391400# Smaller asteroids are tiny #wrt sun diameter
        
        # Color based on orbit type
        orbit_type = asteroid_data.get('Orbit_type', 'Unknown')
        color_map = {
            'Amor': 'aqua',
            'Apollo': 'aquamarine', 
            'Aten': 'aquamarine3',
            'Atira': 'aquamarine2',
            'Centaur': 'chocolate',
            'Trojan': 'chocolate1'
        }
        color = color_map.get(orbit_type, 'gray')
        
        return {
            'name': name,
            'size': visual_size,
            'color':  color ,   #'gray', # doesnt override aqua
            'm': mass_solar_units,
            'a': a,
            'e': e,
            'inc': inc,
            'Omega': Omega,
            'omega': omega,
            'f': f
        }

if __name__ == "__main__":
    
    loader = NEALoader('nea_extended.json')
    loader.load_asteroids()
    loader.print_sample_data()