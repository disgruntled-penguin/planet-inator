import rebound
from spock import FeatureClassifier
from spock import DeepRegressor
from spock import GiantImpactPhaseEmulator
from spock import CollisionOrbitalOutcomeRegressor

import numpy as np

sim = rebound.Simulation()

sim.add(["Sun", "Earth", "Venus", "Mercury", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"])

print("^^solar system without doofs planet")

'''mport rebound
from spock import FeatureClassifier
feature_model = FeatureClassifier()

sim.move_to_com()

print(feature_model.predict_stable(sim))
# >>> 0.06591137'''




reg_model = CollisionOrbitalOutcomeRegressor()



feature_model = FeatureClassifier()


print("stability of the system:", feature_model.predict_stable(sim))  # predicts stabilty - 99 for perfect solar syatem


deep_model = DeepRegressor()

median, lower, upper, samples = deep_model.predict_instability_time(
    sim, samples=10000, return_samples=True, seed=0
)
print(10**np.average(np.log10(samples)))  # Expectation of log-normal 
# >>> 414208.4307974086

print(median) #expectation of number years it remains
# >>> 223792.38826507595  
print("Total particles:", len(sim.particles))

print("system will be unstable in:", int(median*88/365))

'''from spock import AnalyticalClassifier
analytical_model = AnalyticalClassifier()

print(analytical_model.predict_stable(sim.sim))'''
# >>> 0.0 # perfectly chaotic



emulator = GiantImpactPhaseEmulator()

new_sim = emulator.predict(sim)

print("new system after ejection/stabilising", new_sim) #planets ejected?
# >>> <rebound.simulation.Simulation object at 0x303f05c50, N=3, t=999999999.9999993>