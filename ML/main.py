import rebound
from spock import FeatureClassifier
from reboundsim import Simulation

sim = Simulation()
feature_model = FeatureClassifier()


sim.sim.move_to_com()

print(feature_model.predict_stable(sim.sim))
# >>> 0.06591137

import numpy as np
from spock import DeepRegressor
deep_model = DeepRegressor()

median, lower, upper, samples = deep_model.predict_instability_time(
    sim.sim, samples=10000, return_samples=True, seed=0
)
print(10**np.average(np.log10(samples)))  # Expectation of log-normal
# >>> 414208.4307974086

print(median)
# >>> 223792.38826507595 -- in mercury orbit unots

print((median*88) / 365.25) # in human years


'''from spock import AnalyticalClassifier
analytical_model = AnalyticalClassifier()

print(analytical_model.predict_stable(sim.sim))
# >>> 0.0'''

from spock import GiantImpactPhaseEmulator
emulator = GiantImpactPhaseEmulator()

new_sim= emulator.predict(sim.sim)

print(new_sim)
# >>> <rebound.simulation.Simulation object at 0x303f05c50, N=3, t=999999999.9999993>