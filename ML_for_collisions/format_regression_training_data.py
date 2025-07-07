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

                if len(sa) == 101 and sa[100].t < 9999999.9: # remove systems that collide before 1e4 or not at all
                    ps = sa[100].particles
                    if 0.0 < ps[1].a < 50.0 and 0.0 < ps[2].a < 50.0 and 0.0 < ps[1].e < 1.0 and 0.0 < ps[2].e < 1.0: # remove systems with ejected planets
                        # determine indices of input and output particles
                        ps = sa[0].particles
                        ps_after = sa[100].particles
                        if ps_after[1].m == ps[3].m:
                            i1, i2, i3 = 1, 2, 3
                            j1, j2 = 2, 1
                        elif ps_after[2].m == ps[3].m:
                            i1, i2, i3 = 1, 2, 3
                            j1, j2 = 1, 2
                        elif ps_after[1].m == ps[1].m:
                            i1, i2, i3 = 2, 3, 1
                            j1, j2 = 2, 1
                        elif ps_after[2].m == ps[1].m:
                            i1, i2, i3 = 2, 3, 1
                            j1, j2 = 1, 2
                        elif ps_after[1].m == ps[2].m:
                            i1, i2, i3 = 1, 3, 2
                            j1, j2 = 2, 1
                        elif ps_after[2].m == ps[2].m:
                            i1, i2, i3 = 1, 3, 2
                            j1, j2 = 1, 2
                        else:
                            print('Error: planets did not collide!')
                        
                        # get input quantities 
                        input_lst = [np.log10(ps[i1].m), np.log10(ps[i2].m), np.log10(ps[i3].m)]
                        for i in range(100):
                            ps = sa[i].particles
                            input_lst.extend([ps[i1].a, ps[i2].a, ps[i3].a,
                                              np.log10(ps[i1].e), np.log10(ps[i2].e), np.log10(ps[i3].e),
                                              np.log10(ps[i1].inc), np.log10(ps[i2].inc), np.log10(ps[i3].inc),
                                              np.sin(ps[i1].pomega), np.sin(ps[i2].pomega), np.sin(ps[i3].pomega),
                                              np.cos(ps[i1].pomega), np.cos(ps[i2].pomega), np.cos(ps[i3].pomega),
                                              np.sin(ps[i1].Omega), np.sin(ps[i2].Omega), np.sin(ps[i3].Omega),
                                              np.cos(ps[i1].Omega), np.cos(ps[i2].Omega), np.cos(ps[i3].Omega)])
                        train_inputs.append(input_lst)

                        # get output quantities
                        ps = sa[100].particles
                        train_outputs.append([ps[j1].a, ps[j2].a,
                                              np.log10(ps[j1].e), np.log10(ps[j2].e),
                                              np.log10(ps[j1].inc), np.log10(ps[j2].inc)])
            
    train_inputs = np.array(train_inputs)
    train_outputs = np.array(train_outputs)
    
    print('Num inputs:', train_inputs.shape)
    print('Num outputs:', train_outputs.shape)
            
    # save training data
    with open(top_dir + "regression_train_data.pkl", "wb") as f:
        pickle.dump(train_inputs, f)
        pickle.dump(train_outputs, f)
