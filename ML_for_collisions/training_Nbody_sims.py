import numpy as np
import rebound as rb # version 4.0.1
import os

# function to get planet radii from their masses according to Wolfgang+2016
def get_rad(m):
    rad = (m/(2.7*3.0e-6))**(1/1.3)
    return rad*4.26e-4 # units of innermost a (assumed to be ~0.1AU)

# perfect inelastic merger function
def perfect_merge(sim_pointer, collided_particles_index):
    sim = sim_pointer.contents
    ps = sim.particles

    # note that p1 < p2 is not guaranteed 
    i = collided_particles_index.p1
    j = collided_particles_index.p2

    total_mass = ps[i].m + ps[j].m
    merged_planet = (ps[i]*ps[i].m + ps[j]*ps[j].m)/total_mass # conservation of momentum
    merged_radius = (ps[i].r**3 + ps[j].r**3)**(1/3) # merge radius assuming a uniform density

    ps[i] = merged_planet   # update p1's state vector (mass and radius will need to be changed)
    ps[i].m = total_mass    # update to total mass
    ps[i].r = merged_radius # update to joined radius

    sim.stop() # stop sim
    return 2 # remove particle with index j

# celmech helper function
def _compute_transformation_angles(sim):
    Gtot_vec = sim.angular_momentum()
    Gtot_vec = np.array(Gtot_vec)
    Gtot = np.sqrt(Gtot_vec @ Gtot_vec)
    Ghat = Gtot_vec / Gtot
    Ghat_z = Ghat[-1]
    Ghat_perp = np.sqrt(1 - Ghat_z**2)
    theta1 = np.pi/2 - np.arctan2(Ghat[1],Ghat[0])
    theta2 = np.pi/2 - np.arctan2(Ghat_z,Ghat_perp)
    return theta1,theta2

# celmech helper function
def npEulerAnglesTransform(xyz,Omega,I,omega):
    x,y,z = xyz
    s1,c1 = np.sin(omega),np.cos(omega)
    x1 = c1 * x - s1 * y
    y1 = s1 * x + c1 * y
    z1 = z

    s2,c2 = np.sin(I),np.cos(I)
    x2 = x1
    y2 = c2 * y1 - s2 * z1
    z2 = s2 * y1 + c2 * z1

    s3,c3 = np.sin(Omega),np.cos(Omega)
    x3 = c3 * x2 - s3 * y2
    y3 = s3 * x2 + c3 * y2
    z3 = z2

    return np.array([x3,y3,z3])

# change positions and velocities so that angular momentum lies along z-axis
def align_simulation(sim):
    theta1,theta2 = _compute_transformation_angles(sim)
    for p in sim.particles[:sim.N_real]:
        p.x,p.y,p.z = npEulerAnglesTransform(p.xyz,0,theta2,theta1)
        p.vx,p.vy,p.vz = npEulerAnglesTransform(p.vxyz,0,theta2,theta1)

# function to create REBOUND sim object
def initialize_sim():
    sim = rb.Simulation()
    sim.G = 4*np.pi**2 #change to units in which P1=1.0 and a1=1.0
    sim.add(m=1.00)

    i1 = 10**np.random.uniform(-3.0, np.log10(0.3))
    i2 = 10**np.random.uniform(-3.0, np.log10(0.3))
    i3 = 10**np.random.uniform(-3.0, np.log10(0.3))
    m1 = 10**np.random.uniform(-7.0, -4.0)
    m2 = 10**np.random.uniform(-7.0, -4.0)
    m3 = 10**np.random.uniform(-7.0, -4.0)
    r1 = get_rad(m1)
    r2 = get_rad(m2)
    r3 = get_rad(m3)

    dyn_spacing12 = np.random.uniform(0.0, 7.5)
    a_ratio12 = 1 + dyn_spacing12*((m1 + m2)**(1/4))
    P_ratio12 = a_ratio12**(3/2)
    e_cross12 = 1 - 1/P_ratio12
    e1 = 10**np.random.uniform(-3.0, np.log10(e_cross12))

    dyn_spacing23 = np.random.uniform(0.0, 7.5)
    a_ratio23 = 1 + dyn_spacing23*((m2 + m3)**(1/4))
    P_ratio23 = a_ratio23**(3/2)
    e_cross23 = 1 - 1/P_ratio23
    e3 = 10**np.random.uniform(-3.0, np.log10(e_cross23))

    if e_cross12 < e_cross23:
        e2 = 10**np.random.uniform(-3.0, np.log10(e_cross12))
    else:
        e2 = 10**np.random.uniform(-3.0, np.log10(e_cross23))
 
    sim.add(m=m1, P=1.00, e=e1, inc=i1, pomega=np.random.uniform(0.0, 2*np.pi), l=np.random.uniform(0.0, 2*np.pi), Omega=np.random.uniform(0.0, 2*np.pi), r=r1)
    sim.add(m=m2, P=P_ratio12, e=e2, inc=i2, pomega=np.random.uniform(0.0, 2*np.pi), l=np.random.uniform(0.0, 2*np.pi), Omega=np.random.uniform(0.0, 2*np.pi), r=r2)
    sim.add(m=m3, P=P_ratio12*P_ratio23, e=e3, inc=i3, pomega=np.random.uniform(0.0, 2*np.pi), l=np.random.uniform(0.0, 2*np.pi), Omega=np.random.uniform(0.0, 2*np.pi), r=r3)
    sim.move_to_com()
    align_simulation(sim)

    # set integrator
    sim.integrator = 'mercurius'
    sim.collision = 'direct'
    sim.collision_resolve = perfect_merge

    # set timestep
    Ps = np.array([p.P for p in sim.particles[1:len(sim.particles)]])
    es = np.array([p.e for p in sim.particles[1:len(sim.particles)]])
    minTperi = np.min(Ps*(1 - es)**1.5/np.sqrt(1 + es))
    sim.dt = 0.05*minTperi

    return sim

if __name__ == "__main__":
    top_dir = '/scratch/gpfs/cl5968/ML_data/training_sims/'
    run_num = int(os.environ['SLURM_PROCID']) # get run number on cluster
    
    for loop_num in range(1000):
        try:
            filename = top_dir + 'sa' + str(run_num) + '_' + str(loop_num) + '.bin'
            if not os.path.exists(filename):
                sim = initialize_sim() # get sim setup

                # record 100 snapshots over 10^4 orbit integration
                num_saves = 0
                short_times = np.linspace(0.0, 1e4, 100)
                for t in short_times:
                    if len(sim.particles) == 4:
                        sim.integrate(t, exact_finish_time=0)
                        sim.save_to_file(filename)
                        num_saves += 1

                # if there are still three planets, run full integration
                if len(sim.particles) == 4:
                    sim.integrate(1e7, exact_finish_time=0)
                    if sim.t < 9999999.9:
                        sim.save_to_file(filename)
                        num_saves += 1

                # delete things that destabilize before 10^4 or after 10^7 retroactively
                if num_saves < 101:
                    os.remove(filename)
        except Exception as e:
            print(e) # print if something goes wrong
