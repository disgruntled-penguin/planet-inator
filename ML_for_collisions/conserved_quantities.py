import numpy as np
import rebound as rb
import pickle
import os
from tqdm import tqdm

if __name__ == "__main__":
    initial_Es = []
    initial_Ls = []
    final_Es = []
    final_Ls = []
    top_dir = '/scratch/gpfs/cl5968/ML_data/'
    for run_num in tqdm(range(1537)):
        for loop_num in range(1000):
            filename = top_dir + 'training_sims/sa' + str(run_num) + '_' + str(loop_num) + '.bin'
            if os.path.exists(filename):
                sa = rb.Simulationarchive(filename)

                if len(sa) == 101 and sa[100].t < 9999999.9: # remove systems that collide before 1e4 or not at all
                    ps = sa[100].particles
                    if 0.0 < ps[1].a < 50.0 and 0.0 < ps[2].a < 50.0 and 0.0 < ps[1].e < 1.0 and 0.0 < ps[2].e < 1.0: # remove systems with ejected planets
                        # calculate initial energy and angular momentum
                        initial_E = sa[0].energy()
                        initial_Lx, initial_Ly, initial_Lz = sa[0].angular_momentum()
                        initial_Es.append(initial_E)
                        initial_Ls.append([initial_Lx, initial_Ly, initial_Lz])

                        # calculate final energy and angular momentum
                        final_E = sa[100].energy()
                        final_Lx, final_Ly, final_Lz = sa[100].angular_momentum()
                        final_Es.append(final_E)
                        final_Ls.append([final_Lx, final_Ly, final_Lz])

    initial_Es = np.array(initial_Es)
    initial_Ls = np.array(initial_Ls)
    final_Es = np.array(final_Es)
    final_Ls = np.array(final_Ls)

    # save data
    with open(top_dir + "conserved_quantities.pkl", "wb") as f:
        pickle.dump(initial_Es, f)
        pickle.dump(initial_Ls, f)
        pickle.dump(final_Es, f)
        pickle.dump(final_Ls, f)
