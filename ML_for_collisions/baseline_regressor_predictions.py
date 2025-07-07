import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle

# function to predict post-collision orbital elements with baseline model
def make_baseline_pred(inputs, G=4*np.pi**2):
    masses = 10**np.array(inputs[:3])
    orb_elements = np.array(inputs[3:]).reshape((100, 21))
    
    init_a1, init_a2, init_a3 = orb_elements[0][0], orb_elements[0][1], orb_elements[0][2]
    init_e1, init_e2, init_e3 = 10**orb_elements[0][3], 10**orb_elements[0][4], 10**orb_elements[0][5]
    init_inc1, init_inc2, init_inc3 = 10**orb_elements[0][6], 10**orb_elements[0][7], 10**orb_elements[0][8]
    
    E_before = -G*masses[0]/(2*init_a1) - G*masses[1]/(2*init_a2) - G*masses[2]/(2*init_a3)
    Lz_before = masses[0]*np.cos(init_inc1)*np.sqrt(G*init_a1)*np.sqrt(1 - init_e1**2) +\
                masses[1]*np.cos(init_inc2)*np.sqrt(G*init_a2)*np.sqrt(1 - init_e2**2) +\
                masses[2]*np.cos(init_inc3)*np.sqrt(G*init_a3)*np.sqrt(1 - init_e3**2)
    
    # determine a1, e1, and inc1 with mass-weighted mean over short integration
    a1s = orb_elements[:,0]
    a2s = orb_elements[:,1]
    e1s = 10**orb_elements[:,3]
    e2s = 10**orb_elements[:,4]
    e3s = 10**orb_elements[:,5]
    inc1s = 10**orb_elements[:,6]
    inc2s = 10**orb_elements[:,7]
    inc3s = 10**orb_elements[:,8]
    
    new_a1 = np.mean(a1s*masses[0] + a2s*masses[1])/(masses[0] + masses[1])
    new_e1 = np.mean(e1s*masses[0] + e2s*masses[1])/(masses[0] + masses[1])
    new_inc1 = np.mean(inc1s*masses[0] + inc2s*masses[1])/(masses[0] + masses[1])
    
    # determine a2 assuming energy conservation
    new_masses = [masses[0] + masses[1], masses[2]]
    new_a2 = -0.5*G*new_masses[1]/(E_before + G*new_masses[0]/(2*new_a1))
    
    # take inc2 to be mean inc2 over short integration
    new_inc2 = np.mean(inc3s)
    
    # determine e2 using L_z conservation
    Lz_2 = Lz_before - new_masses[0]*np.cos(new_inc1)*np.sqrt(G*new_a1)*np.sqrt(1 - new_e1**2)
    new_e2 = np.sqrt(1 - (Lz_2/(new_masses[1]*np.cos(new_inc2)*np.sqrt(G*new_a2)))**2)
    
    # if Lz_2 < Lz_before, just use mean e3
    if np.isnan(new_e2):
        new_e2 = np.mean(e3s)
    
    return np.array([new_a1, new_a2, np.log10(new_e1), np.log10(new_e2), np.log10(new_inc1), np.log10(new_inc2)])

if __name__ == "__main__":
    # load training set
    top_dir = '/scratch/gpfs/cl5968/ML_data/'
    filename = top_dir + 'regression_train_data.pkl'
    f = open(filename, "rb")
    training_inputs = pickle.load(f)
    training_outputs = pickle.load(f)
    f.close()
    
    # split training set in the same way
    _, eval_inputs, _, eval_outputs = train_test_split(training_inputs, training_outputs, test_size=0.2, shuffle=False)
    
    # make predictions with baseline model
    eval_preds = []
    for inputs in tqdm(eval_inputs):
        eval_preds.append(make_baseline_pred(inputs))
    eval_preds = np.array(eval_preds)

    # save results
    with open(top_dir + 'baseline_regressor_validation_preds.pkl', 'wb') as f:
        pickle.dump(eval_inputs, f)
        pickle.dump(eval_preds, f)
        pickle.dump(eval_outputs, f)
