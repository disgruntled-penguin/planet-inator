import numpy as np
import random
import os
import torch
import time
import warnings
from tqdm import tqdm
import rebound as rb
from spock import DeepRegressor
from spock.tseries_feature_functions import get_collision_tseries
from spock.simsetup import scale_sim, align_simulation, get_rad, npEulerAnglesTransform, revert_sim_units, sim_subset, remove_ejected_ps, replace_trio

# baseline model that resembles ML-based emulator except instability times and outcomes are predicted differently
class GiantImpactPhaseBaseline():
    # initialize function
    def __init__(self, seed=None):
        # set random seed
        if not seed is None:
            os.environ["PL_GLOBAL_SEED"] = str(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
    def predict(self, sims, tmaxs=None, verbose=False):
        """
        Predict outcome of giant impact simulations up to a user-specified maximum time.

        Parameters:

        sims (rebound.Simulation or list): Initial condition(s) for giant impact simulations.
        tmaxs (float or list): Maximum time for simulation predictions in the same units as sims.t. The default is 10^9 P1, the maximum possible time.
        verbose (bool): Whether or not to provide outputs during the iterative prediction process.

        Returns:
        
        rebound.Simulation or list: Predicted post-giant impact states for each provided initial condition.

        """
        single_sim = False
        if isinstance(sims, rb.Simulation): # passed a single sim
            sims = [sims]
            single_sim = True
        
        # main loop
        sims, tmaxs = self._make_lists(sims, tmaxs)
        while np.any([sim.t < tmaxs[i] for i, sim in enumerate(sims)]): # take another step if any sims are still at t < tmax
            sims = self.step(sims, tmaxs, verbose=verbose) # keep dimensionless units until the end
               
        if single_sim:
            sims = sims[0]
                
        return sims

    def step(self, sims, tmaxs, verbose=False):
        """
        Perform another step in the iterative prediction process, merging any planets that go unstable in t < tmax.

        Parameters:

        sims (rebound.Simulation or list): Current state of the giant impact simulations.
        tmaxs (float or list): Maximum time for simulation predictions in the same units as sims.t. The default is 10^9 P1, the maximum possible time.
        verbose (bool): Whether or not to provide outputs during the iterative prediction process.
        

        Returns:
        
        rebound.Simulation or list: Predicted states after one step of predicting instability times and merging planets.

        """
        single_sim = False
        if isinstance(sims, rb.Simulation): # passed a single sim
            sims = [sims]
            single_sim = True
        
        sims, tmaxs = self._make_lists(sims, tmaxs)
        for i, sim in enumerate(sims): # assume all 2 planet systems (N=3) are stable (could use Hill stability criterion)
            if sim.N < 4:
                sim.t = tmaxs[i]
        
        sims_to_update = [sim for i, sim in enumerate(sims) if sim.t < tmaxs[i]]
        # estimate instability times for the subset of systems
        if len(sims_to_update) == 0:
            return sims

        if verbose:
            print('Predicting trio instability times')
            start = time.time()
        
        t_insts, trio_inds = self._get_unstable_trios(sims_to_update)
        
        if verbose:
            end = time.time()
            print('Done:', end - start, 's' + '\n')
        
        # get list of sims for which planets need to be merged
        sims_to_merge = []
        trios_to_merge = []
        for i, sim in enumerate(sims_to_update):
            idx = sims.index(sim)           # get index in original list
            if t_insts[i] > tmaxs[idx]:     # won't merge before max time, so just update to that time
                sims[idx].t = tmaxs[idx]
            else:                           # need to merge
                sim.t = t_insts[i]         # update time
                sims_to_merge.append(sim)
                trios_to_merge.append(trio_inds[i])
        
        if verbose:
            print('Predicting instability outcomes')
            start = time.time()
        
        # get new sims with planets merged
        sims = self._handle_mergers(sims, sims_to_merge, trios_to_merge)
        
        if verbose:
            end = time.time()
            print('Done:', end - start, 's' + '\n')
        
        if single_sim:
            sims = sims[0]
        
        return sims

    # internal function for baseline t_inst predictions
    def _predict_instability_times(self, trio_sims, P1=0.0316):
        ps = trio_sims.particles
        #P1 = ps[1].P
        mutual_rad12 = ((ps[1].m + ps[2].m)/(3*ps[0].m))**(1/3)
        sep12 = (ps[2].a - ps[1].a)/(0.5*(ps[1].a + ps[2].a)) - ps[2].e - ps[1].e
        spacing12 = sep12/mutual_rad12
        
        mutual_rad23 = ((ps[2].m + ps[3].m)/(3*ps[0].m))**(1/3)
        sep23 = (ps[3].a - ps[2].a)/(0.5*(ps[2].a + ps[3].a)) - ps[3].e - ps[2].e
        spacing23 = sep23/mutual_rad23
        
        min_delta = np.min([spacing12, spacing23])
        
        return (10**(min_delta*9/10))*P1 # trick to make system stable once sep > 10 mutual Hill radii
    
    # get unstable trios for list of sims
    def _get_unstable_trios(self, sims):
        trio_sims = []
        trio_inds = []
        Npls = [sim.N - 1 for sim in sims]
        
        # break system up into three-planet sub-systems
        for i in range(len(sims)):
            for j in range(Npls[i] - 2):
                trio_inds.append([j+1, j+2, j+3])
                trio_sims.append(sim_subset(sims[i], [j+1, j+2, j+3]))
        
        # predict instability times for sub-trios
        t_insts = []
        for i, trio_sim in enumerate(trio_sims):
            t_inst = self._predict_instability_times(trio_sim)
            t_insts.append(t_inst)
        t_insts = np.array(t_insts)

        # get the minimum sub-trio instability time for each system
        min_t_insts = []
        min_trio_inds = []
        for i in range(len(sims)):
            temp_t_insts = []
            temp_trio_inds = []
            for j in range(Npls[i] - 2):
                temp_t_insts.append(t_insts[int(np.sum(Npls[:i]) - 2*i + j)])
                temp_trio_inds.append(trio_inds[int(np.sum(Npls[:i]) - 2*i + j)])
            min_ind = np.argmin(temp_t_insts)
            min_t_insts.append(temp_t_insts[min_ind])
            min_trio_inds.append(temp_trio_inds[min_ind])
            
        return min_t_insts, min_trio_inds

    # internal function for baseline collision pair probabilities
    def _predict_collision_probs(self, sim, trio_inds):
        ind1, ind2, ind3 = int(trio_inds[0]), int(trio_inds[1]), int(trio_inds[2])
        m1, m2, m3 = sim.particles[ind1].m, sim.particles[ind2].m, sim.particles[ind3].m

        c1, c2, c3 = 2.354e+00, 2.119e+00, 1.570e+00
        logit1 = c1*np.log10(np.max([m2/m1, m1/m2]))
        logit2 = c2*np.log10(np.max([m3/m2, m2/m3]))
        logit3 = c3*np.log10(np.max([m3/m1, m1/m3]))
        probs = np.exp([logit1, logit2, logit3])/sum(np.exp([logit1, logit2, logit3]))

        return probs
    
    # internal function for predicting collision indices
    def _predict_collision_pair(self, sims, trio_inds):
        collision_inds = np.zeros((len(sims), 2))
        for i, sim in enumerate(sims):
            probs = self._predict_collision_probs(sim, trio_inds[i])
            
            rand_num = np.random.rand()
            if rand_num < probs[0]:
                collision_inds[i] = [1, 2]
            elif rand_num < probs[0] + probs[1]:
                collision_inds[i] = [2, 3]
            else:
                collision_inds[i] = [1, 3]
        
        return collision_inds
    
    # internal function for predicting post-collision orbital elements for a single sim
    def _baseline_reg_pred(self, inputs, G=4*np.pi**2):
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
    
    # internal function for baseline collision outcome handling
    def _predict_collision_outcome(self, sims, collision_inds, trio_inds):
        sims = [scale_sim(sim, np.arange(1, sim.N)) for sim in sims] # re-scale input sims and convert units
        done_sims = []
        trio_sims = []
        mlp_inputs = []
        done_inds = []
        for i in tqdm(range(len(sims))):
            out, trio_sim, _ = get_collision_tseries(sims[i], trio_inds[i])
            
            if len(trio_sim.particles) == 4:
                # no merger (or ejection)
                mlp_inputs.append(out)
                trio_sims.append(trio_sim)
            else: 
                # if merger/ejection occurred, save sim
                done_sims.append(replace_trio(sims[i], trio_inds[i], trio_sim))
                done_inds.append(i)

        # get collision_inds for sims that did not experience a merger
        if 0 < len(done_inds):
            mask = np.ones(len(collision_inds), dtype=bool)
            mask[np.array(done_inds)] = False
            subset_collision_inds = list(np.array(collision_inds)[mask])
        else:
            subset_collision_inds = collision_inds
        
        if 0 < len(mlp_inputs):
            # re-order input array based on input collision_inds
            reg_inputs = []
            for i, col_ind in enumerate(subset_collision_inds):
                masses = mlp_inputs[i][:3]
                orb_elements = mlp_inputs[i][3:]

                if (col_ind[0] == 1 and col_ind[1] == 2) or (col_ind[0] == 2 and col_ind[1] == 1): # merge planets 1 and 2
                    ordered_masses = masses
                    ordered_orb_elements = orb_elements
                elif (col_ind[0] == 2 and col_ind[1] == 3) or (col_ind[0] == 3 and col_ind[1] == 2): # merge planets 2 and 3
                    ordered_masses = np.array([masses[1], masses[2], masses[0]])
                    ordered_orb_elements = np.column_stack((orb_elements[1::3], orb_elements[2::3], orb_elements[0::3])).flatten()
                elif (col_ind[0] == 1 and col_ind[1] == 3) or (col_ind[0] == 3 and col_ind[1] == 1): # merge planets 1 and 3
                    ordered_masses = np.array([masses[0], masses[2], masses[1]])
                    ordered_orb_elements = np.column_stack((orb_elements[0::3], orb_elements[2::3], orb_elements[1::3])).flatten()
                else:
                    warnings.warn('Invalid collision_inds')

                reg_inputs.append(np.concatenate((ordered_masses, ordered_orb_elements)))

            # predict orbital elements with baseline model
            reg_outputs = []
            reg_inputs = np.array(reg_inputs)
            for reg_input in reg_inputs:
                reg_outputs.append(self._baseline_reg_pred(reg_input))
            reg_outputs = np.array(reg_outputs)

            m1s = 10**reg_inputs[:,0] + 10**reg_inputs[:,1] # new planet
            m2s = 10**reg_inputs[:,2] # surviving planet
            a1s = reg_outputs[:,0]
            a2s = reg_outputs[:,1]
            e1s = 10**reg_outputs[:,2]
            e2s = 10**reg_outputs[:,3]
            inc1s = 10**reg_outputs[:,4]
            inc2s = 10**reg_outputs[:,5]
            
        new_sims = []
        j = 0 # index for new sims array
        k = 0 # index for mlp prediction arrays
        for i in range(len(sims)):
            if i in done_inds:
                new_sims.append(done_sims[j])
                j += 1
            else:                
                # create sim that contains state of two predicted planets
                new_state_sim = rb.Simulation()
                new_state_sim.G = 4*np.pi**2 # units in which a1=1.0 and P1=1.0
                new_state_sim.add(m=1.00)
               
                try:
                    if (0.0 < a1s[k] < 50.0) and (0.0 <= e1s[k] < 1.0):
                        new_state_sim.add(m=m1s[k], a=a1s[k], e=e1s[k], inc=inc1s[k], pomega=np.random.uniform(0.0, 2*np.pi), Omega=np.random.uniform(0.0, 2*np.pi), l=np.random.uniform(0.0, 2*np.pi))
                    else:
                        warnings.warn('Removing ejected planet')
                except Exception as e:
                    warnings.warn('Removing planet with unphysical orbital elements')
                try:
                    if (0.0 < a2s[k] < 50.0) and (0.0 <= e2s[k] < 1.0):
                        new_state_sim.add(m=m2s[k], a=a2s[k], e=e2s[k], inc=inc2s[k], pomega=np.random.uniform(0.0, 2*np.pi), Omega=np.random.uniform(0.0, 2*np.pi), l=np.random.uniform(0.0, 2*np.pi))
                    else:
                        warnings.warn('Removing ejected planet')
                except Exception as e:
                    warnings.warn('Removing planet with unphysical orbital elements')
                new_state_sim.move_to_com()
               # print(f'new system after ejection/stabilising object at {hex(sims)}, N = {a2s} t = {delta.t}')
                
                for p in new_state_sim.particles[:new_state_sim.N]:
                    p.x, p.y, p.z = npEulerAnglesTransform(p.xyz, -trio_sims[k].theta1, -trio_sims[k].theta2, 0)
                    p.vx, p.vy, p.vz = npEulerAnglesTransform(p.vxyz, -trio_sims[k].theta1, -trio_sims[k].theta2, 0)
                # replace trio with predicted duo (or single/zero if planets have unphysical orbital elements)
                new_sims.append(replace_trio(sims[i], trio_inds[i], new_state_sim))
                k += 1
        
        # convert sims back to original units
        new_sims = revert_sim_units(new_sims)
            
        return new_sims
    
    # internal function for handling mergers with class_model and reg_model
    def _handle_mergers(self, sims, sims_to_merge, trio_inds):
        # predict which planets will collide in each trio_sim
        collision_inds = self._predict_collision_pair(sims_to_merge, trio_inds)
            
        # predict post-collision orbital states with regression model
        new_sims = self._predict_collision_outcome(sims_to_merge, collision_inds, trio_inds)
       
        # update sims
        for i, sim in enumerate(sims_to_merge):
            idx = sims.index(sim) # find index in original list
            sims[idx] = new_sims[i]
        return sims

    # internal function with logic for initializing orbsmax as an array and checking for warnings
    def _make_lists(self, sims, tmaxs):
        sims = remove_ejected_ps(sims) # remove ejected/hyperbolic particles
        
        # use passed value
        if tmaxs:
            try:
                len(tmaxs) == len(sims)
            except:
                tmaxs = tmaxs*np.ones(len(sims)) # convert from float to array
        else:       # default = 1e9 orbits
            tmaxs = [1e9*sim.particles[1].P for sim in sims]

        for i, t in enumerate(tmaxs):
            orbsmax = t/sims[i].particles[1].P
            if orbsmax > 1.5e9:
                warnings.warn('Giant impact phase emulator not trained to predict beyond 10^9 orbits, check results carefully (tmax for sim {0} = {1} = {2} orbits)'.format(i, t, orbsmax))
        return sims, tmaxs
        