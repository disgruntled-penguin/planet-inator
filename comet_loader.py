import json
import math
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta

class CometLoader:
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.comets = []
        self.loaded = False
    
    def load_comets(self, limit: Optional[int] = None):
        print(f"Loading comets from {self.json_file_path}...")
        
        try:
            with open(self.json_file_path, 'r') as f:
                data = json.load(f)
                
            if limit:
                self.comets = data[:limit]
                print(f"Loaded {len(self.comets)} comets (limited)")
            else:
                self.comets = data
                print(f"Loaded {len(self.comets)} comets")
            
            # Print sample data automatically after loading
            print(f"\nFirst 3 comets:")
            for i, comet in enumerate(self.comets[:3]):
                name = comet.get('Designation_and_name', 'Unknown')
                orbit_type = comet.get('Orbit_type', 'Unknown')
                perihelion_dist = comet.get('Perihelion_dist', 'N/A')
                eccentricity = comet.get('e', 'N/A')
                print(f"  {i+1}. {name} (Type={orbit_type}, q={perihelion_dist} AU, e={eccentricity})")
                
            self.loaded = True
            return len(self.comets)
            
        except FileNotFoundError:
            print(f"Error: Could not find file {self.json_file_path}")
            return 0
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {self.json_file_path}")
            return 0
    
    def print_sample_data(self, num_samples: int = 50):
        if not self.loaded:
            print("Error: Comets not loaded yet. Call load_comets() first.")
            return
        if not self.comets:
            print("No comets found in the loaded data.")
            return
        
        print(f"\n--- Sample Data ({min(num_samples, len(self.comets))} comets) ---")
        
        for i, comet in enumerate(self.comets[:num_samples]):
            print(f"\nComet {i+1}:")
            print(f"  Designation: {comet.get('Designation_and_name', 'Unknown')}")
            print(f"  Orbit type: {comet.get('Orbit_type', 'Unknown')}")
            print(f"  Perihelion distance (q): {comet.get('Perihelion_dist', 'N/A')} AU")
            print(f"  Eccentricity (e): {comet.get('e', 'N/A')}")
            print(f"  Inclination (i): {comet.get('i', 'N/A')}°")
            print(f"  Argument of periapsis (ω): {comet.get('Peri', 'N/A')}°")
            print(f"  Longitude of ascending node (Ω): {comet.get('Node', 'N/A')}°")
            print(f"  H magnitude: {comet.get('H', 'N/A')}")
            print(f"  Year of perihelion: {comet.get('Year_of_perihelion', 'N/A')}")
            print(f"  Month of perihelion: {comet.get('Month_of_perihelion', 'N/A')}")
            print(f"  Day of perihelion: {comet.get('Day_of_perihelion', 'N/A')}")
            
            # Show all available keys for the first comet
            if i == 0:
                print(f"  Available keys: {list(comet.keys())}")
        
        print(f"\nTotal comets loaded: {len(self.comets)}")
    
    def get_comets(self, count: int = 100):
        if not self.loaded:
            self.load_comets(limit=5000)  # Load a subset first
        
        regular = []
        
        for comet in self.comets:
            regular.append(comet)
        
        return regular[:count]
    
    def calculate_semi_major_axis(self, q: float, e: float) -> Optional[float]:
        """Calculate semi-major axis from perihelion distance and eccentricity"""
        if e >= 1.0:
            # Parabolic or hyperbolic orbit - no meaningful semi-major axis
            return None
        return q / (1 - e)
    
    def estimate_orbital_period(self, a: Optional[float]) -> Optional[float]:
        """Estimate orbital period in years using Kepler's third law"""
        if a is None or a <= 0:
            return None
        return math.sqrt(a**3)
    
    def convert_to_rebound_params(self, comet_data: Dict):
        """Convert comet JSON data to rebound parameters"""
        
        # Extract orbital elements
        q = comet_data.get('Perihelion_dist')  # Perihelion distance in AU
        e = comet_data.get('e')  # Eccentricity
        inc = np.radians(comet_data.get('i', 0))  # Inclination
        Omega = np.radians(comet_data.get('Node', 0))  # Longitude of ascending node
        omega = np.radians(comet_data.get('Peri', 0))  # Argument of periapsis
        
        # Calculate semi-major axis
        if e < 1.0:
            a = self.calculate_semi_major_axis(q, e)
        else:
            # For parabolic/hyperbolic orbits, use perihelion distance as reference
            a = q
        
        # Estimate mean anomaly based on time since perihelion
        # This is a simplified approach - for accurate positions, 
        # you'd need to calculate from the exact perihelion time
        M = 0.0  # Start at perihelion for simplicity
        f = M  # Approximate true anomaly
        
        # Estimate mass from H magnitude (very rough approximation for comets)
        # Comets are typically less dense than asteroids
        H = comet_data.get('H', 15)  # Default to faint if not specified
        if H is not None:
            estimated_diameter_km = 1329.0 / math.sqrt(0.04) * 10**(-0.2 * H)  # Lower albedo for comets
            estimated_mass_kg = (4/3) * math.pi * (estimated_diameter_km * 500)**3 * 500  # Lower density ~0.5 g/cc
            m_earth_kg = 5.972e24
            mass_earth_units = estimated_mass_kg / m_earth_kg
            mass_solar_units = mass_earth_units / 332946.0487
        else:
            mass_solar_units = 1e-15  # Very small default mass
        
        # Visual properties
        name = comet_data.get('Designation_and_name', 'Unknown Comet')
        
        # Size for visualization (comets are typically smaller than asteroids)
        if H is not None:
            visual_size = 0.05 * estimated_diameter_km / 1391400  # Smaller than asteroids
        else:
            visual_size = 1e-6  # Very small default
        
        # Color based on orbit type and eccentricity
        orbit_type = comet_data.get('Orbit_type', 'Unknown')
        if e >= 1.0:
            color = 'yellow'  # Hyperbolic/parabolic comets
        elif e > 0.9:
            color = 'orange'  # Highly eccentric
        else:
            color = 'lightblue'  # Short-period comets
        
        # Orbital period estimation
        period = self.estimate_orbital_period(a) if e < 1.0 else None
        
        result = {
            'name': name,
            'size': visual_size,
            'color': color,
            'm': mass_solar_units,
            'a': a if a is not None else q,  # Use perihelion distance for hyperbolic orbits
            'e': e,
            'inc': inc,
            'Omega': Omega,
            'omega': omega,
            'f': f,
            'perihelion_dist': q,
            'orbital_period': period,
            'orbit_type': orbit_type
        }
        
        return result

    
    def get_orbit_statistics(self):
        """Print statistics about the loaded comets"""
        if not self.loaded:
            print("Error: Comets not loaded yet. Call load_comets() first.")
            return
        
        if not self.comets:
            print("No comets loaded.")
            return
        
        # Count orbit types
        orbit_types = {}
        elliptical_count = 0
        hyperbolic_count = 0
        
        perihelion_distances = []
        eccentricities = []
        
        for comet in self.comets:
            orbit_type = comet.get('Orbit_type', 'Unknown')
            orbit_types[orbit_type] = orbit_types.get(orbit_type, 0) + 1
            
            e = comet.get('e')
            if e is not None:
                eccentricities.append(e)
                if e < 1.0:
                    elliptical_count += 1
                else:
                    hyperbolic_count += 1
            
            q = comet.get('Perihelion_dist')
            if q is not None:
                perihelion_distances.append(q)
        
        print(f"\n--- Comet Statistics ---")
        print(f"Total comets: {len(self.comets)}")
        print(f"Elliptical orbits (e < 1.0): {elliptical_count}")
        print(f"Hyperbolic/Parabolic orbits (e >= 1.0): {hyperbolic_count}")
        
        print(f"\nOrbit types:")
        for orbit_type, count in sorted(orbit_types.items()):
            print(f"  {orbit_type}: {count}")
        
        if perihelion_distances:
            print(f"\nPerihelion distance range: {min(perihelion_distances):.3f} - {max(perihelion_distances):.3f} AU")
            print(f"Average perihelion distance: {sum(perihelion_distances)/len(perihelion_distances):.3f} AU")
        
        if eccentricities:
            print(f"\nEccentricity range: {min(eccentricities):.3f} - {max(eccentricities):.3f}")
            print(f"Average eccentricity: {sum(eccentricities)/len(eccentricities):.3f}")

if __name__ == "__main__":
    # Example usage
    loader = CometLoader('allcometels.json')  # Adjust filename as needed
    loader.load_comets()
    loader.print_sample_data()
    loader.get_orbit_statistics()
    
    # Example filtering
    ''' print("\n--- Filtering Examples ---")
    short_period = loader.filter_comets(elliptical_only=True, max_perihelion=5.0)
    long_period = loader.filter_comets(elliptical_only=True, min_perihelion=5.0)
    hyperbolic = loader.filter_comets(hyperbolic_only=True)'''