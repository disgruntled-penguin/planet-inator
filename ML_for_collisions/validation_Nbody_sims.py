import numpy as np
import rebound as rb
import pickle
import time
import os

# function to draw planet masses from a non-negative Gaussian
def get_mass(loc, scale, min_mass=0.0):
    mass = min_mass - 1.0
    while mass < min_mass:
        mass = np.random.normal(loc=loc, scale=scale)
    return mass

# function to get planet radii from their masses according to Wolfgang+2016
def get_rad(m):
    rad = (m/(2.7*3.0e-6))**(1/1.3)
    return rad*4.26e-5 # units of AU

if __name__ == "__main__":
    top_dir = "/scratch/gpfs/cl5968/ML_data/"
    run_num = int(os.environ['SLURM_PROCID']) # get run number on cluster
    
    P1 = 0.0316 # corresponds to a1=0.1AU
    mean_mass = 3*3.0e-6 # 3 Earth masses
    std_mass = 3*3.0e-6 # 3 Earth masses

    # draw random masses
    m0 = get_mass(mean_mass, std_mass)
    m1 = get_mass(mean_mass, std_mass)
    m2 = get_mass(mean_mass, std_mass)
    m3 = get_mass(mean_mass, std_mass)
    m4 = get_mass(mean_mass, std_mass)
    m5 = get_mass(mean_mass, std_mass)
    m6 = get_mass(mean_mass, std_mass)
    m7 = get_mass(mean_mass, std_mass)
    m8 = get_mass(mean_mass, std_mass)
    m9 = get_mass(mean_mass, std_mass)

    # get corresponding radii
    r0 = get_rad(m0)
    r1 = get_rad(m1)
    r2 = get_rad(m2)
    r3 = get_rad(m3)
    r4 = get_rad(m4)
    r5 = get_rad(m5)
    r6 = get_rad(m6)
    r7 = get_rad(m7)
    r8 = get_rad(m8)
    r9 = get_rad(m9)
    
    # setup REBOUND simulation
    sim = rb.Simulation()
    sim.integrator = 'mercurius'
    sim.units = ('yr', 'AU', 'Msun')
    sim.dt = P1/20
    sim.collision = 'direct'
    sim.collision_resolve = 'merge'
    
    # save state of system at 1000 equally-spaced points in time
    sim.save_to_file(top_dir + "Nbody_giant_impact/sa" + str(run_num) + '.bin', interval=1e6*P1, delete_file=True)

    # add particles
    sim.add(m=1.0, hash='star')
    P = P1
    sim.add(m=m0, P=P, e=np.random.rayleigh(0.01), inc=(180/np.pi)*np.random.rayleigh(0.5), omega=np.random.uniform(0.0, 2*np.pi), Omega=np.random.uniform(0.0, 2*np.pi), l=np.random.uniform(0.0, 2*np.pi), r=r0)
    P = P*np.random.uniform(1.10, 1.75)
    sim.add(m=m1, P=P, e=np.random.rayleigh(0.01), inc=(180/np.pi)*np.random.rayleigh(0.5), omega=np.random.uniform(0.0, 2*np.pi), Omega=np.random.uniform(0.0, 2*np.pi), l=np.random.uniform(0.0, 2*np.pi), r=r1)
    P = P*np.random.uniform(1.10, 1.75)
    sim.add(m=m2, P=P, e=np.random.rayleigh(0.01), inc=(180/np.pi)*np.random.rayleigh(0.5), omega=np.random.uniform(0.0, 2*np.pi), Omega=np.random.uniform(0.0, 2*np.pi), l=np.random.uniform(0.0, 2*np.pi), r=r2)
    P = P*np.random.uniform(1.10, 1.75)
    sim.add(m=m3, P=P, e=np.random.rayleigh(0.01), inc=(180/np.pi)*np.random.rayleigh(0.5), omega=np.random.uniform(0.0, 2*np.pi), Omega=np.random.uniform(0.0, 2*np.pi), l=np.random.uniform(0.0, 2*np.pi), r=r3)
    P = P*np.random.uniform(1.10, 1.75)
    sim.add(m=m4, P=P, e=np.random.rayleigh(0.01), inc=(180/np.pi)*np.random.rayleigh(0.5), omega=np.random.uniform(0.0, 2*np.pi), Omega=np.random.uniform(0.0, 2*np.pi), l=np.random.uniform(0.0, 2*np.pi), r=r4)
    P = P*np.random.uniform(1.10, 1.75)
    sim.add(m=m5, P=P, e=np.random.rayleigh(0.01), inc=(180/np.pi)*np.random.rayleigh(0.5), omega=np.random.uniform(0.0, 2*np.pi), Omega=np.random.uniform(0.0, 2*np.pi), l=np.random.uniform(0.0, 2*np.pi), r=r5)
    P = P*np.random.uniform(1.10, 1.75)
    sim.add(m=m6, P=P, e=np.random.rayleigh(0.01), inc=(180/np.pi)*np.random.rayleigh(0.5), omega=np.random.uniform(0.0, 2*np.pi), Omega=np.random.uniform(0.0, 2*np.pi), l=np.random.uniform(0.0, 2*np.pi), r=r6)
    P = P*np.random.uniform(1.10, 1.75)
    sim.add(m=m7, P=P, e=np.random.rayleigh(0.01), inc=(180/np.pi)*np.random.rayleigh(0.5), omega=np.random.uniform(0.0, 2*np.pi), Omega=np.random.uniform(0.0, 2*np.pi), l=np.random.uniform(0.0, 2*np.pi), r=r7)
    P = P*np.random.uniform(1.10, 1.75)
    sim.add(m=m8, P=P, e=np.random.rayleigh(0.01), inc=(180/np.pi)*np.random.rayleigh(0.5), omega=np.random.uniform(0.0, 2*np.pi), Omega=np.random.uniform(0.0, 2*np.pi), l=np.random.uniform(0.0, 2*np.pi), r=r8)
    P = P*np.random.uniform(1.10, 1.75)
    sim.add(m=m9, P=P, e=np.random.rayleigh(0.01), inc=(180/np.pi)*np.random.rayleigh(0.5), omega=np.random.uniform(0.0, 2*np.pi), Omega=np.random.uniform(0.0, 2*np.pi), l=np.random.uniform(0.0, 2*np.pi), r=r9)
    sim.move_to_com()
    
    # integrate and record time elapsed
    start = time.time()
    sim.integrate(1e9*P1)
    end = time.time()

    # save runtime
    with open(top_dir + "/Nbody_giant_impact_runtimes/run" + str(run_num) + ".pkl", "wb") as f:
        pickle.dump(end - start, f)
