import numpy as np
import rebound as rb
import pickle
import os
from tqdm import tqdm

if __name__ == "__main__":
    train_inputs = []
    train_outputs = []
    top_dir = '/scratch/gpfs/cl5968/ML_data/'
    for run_num in tqdm(range(1537)):
        for loop_num in range(1000):
            filename = top_dir + 'training_sims/sa' + str(run_num) + '_' + str(loop_num) + '.bin'
            if os.path.exists(filename):
                sa = rb.Simulationarchive(filename)
                if len(sa) == 101 and sa[-1].t < 9999999.9: # remove systems that collide before 1e4 or not at all
                    ps = sa[100].particles
                    if 0.0 < ps[1].a < 50.0 and 0.0 < ps[2].a < 50.0 and 0.0 < ps[1].e < 1.0 and 0.0 < ps[2].e < 1.0: # remove systems with ejected planets
                        # get input quantities
                        ps = sa[0].particles
                        input_lst = [np.log10(ps[1].m), np.log10(ps[2].m), np.log10(ps[3].m)]
                        for i in range(100):
                            ps = sa[i].particles
                            input_lst.extend([ps[1].a, ps[2].a, ps[3].a,
                                              np.log10(ps[1].e), np.log10(ps[2].e), np.log10(ps[3].e),
                                              np.log10(ps[1].inc), np.log10(ps[2].inc), np.log10(ps[3].inc),
                                              np.sin(ps[1].pomega), np.sin(ps[2].pomega), np.sin(ps[3].pomega),
                                              np.cos(ps[1].pomega), np.cos(ps[2].pomega), np.cos(ps[3].pomega),
                                              np.sin(ps[1].Omega), np.sin(ps[2].Omega), np.sin(ps[3].Omega),
                                              np.cos(ps[1].Omega), np.cos(ps[2].Omega), np.cos(ps[3].Omega)])                            
                        train_inputs.append(input_lst)

                        # get output quantities (easiest to determine which two planets collided using their masses)
                        ps = sa[0].particles
                        ps_after = sa[100].particles
                        if ps_after[1].m == ps[3].m or ps_after[2].m == ps[3].m:
                            train_outputs.append([1.0, 0.0, 0.0])
                        elif ps_after[1].m == ps[1].m or ps_after[2].m == ps[1].m:
                            train_outputs.append([0.0, 1.0, 0.0])
                        elif ps_after[1].m == ps[2].m or ps_after[2].m == ps[2].m:
                            train_outputs.append([0.0, 0.0, 1.0])
                        else:
                            print('Something went wrong!')
                            print('Before:', ps[1].m, ps[2].m, ps[3].m)
                            print('After:', ps_after[1].m, ps_after[2].m)
            
    train_inputs = np.array(train_inputs)
    train_outputs = np.array(train_outputs)
            
    print('Num inputs:', train_inputs.shape)
    print('Num outputs:', train_outputs.shape)
    
    #s save training data
    with open(top_dir + "classification_train_data.pkl", "wb") as f:
        pickle.dump(train_inputs, f)
        pickle.dump(train_outputs, f)
