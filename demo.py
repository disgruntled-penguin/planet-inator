#spock demo 
import rebound 

from spock import FeatureClassifier
feature_model = FeatureClassifier()


sim = rebound.Simulation()
sim.add(m=1.)
sim.add(m=1.e-5, P=1., e=0.03, pomega=2., l=0.5)
sim.add(m=1.e-5, P=1.2, e=0.03, pomega=3., l=3.)
sim.add(m=1.e-5, P=1.5, e=0.03, pomega=1.5, l=2.)
sim.move_to_com()

print(feature_model.predict_stable(sim))
# >>> 0.06591137

import numpy as np
from spock import DeepRegressor
deep_model = DeepRegressor()

median, lower, upper, samples = deep_model.predict_instability_time(
    sim, samples=10000, return_samples=True, seed=0
)
print(10**np.average(np.log10(samples)))  # Expectation of log-normal
# >>> 414208.4307974086 

print(median)
# >>> 223792.38826507595. #i think,The returned time is expressed in the time units used in setting up the REBOUND Simulation above. Since we set the innermost planet orbit to unity, this corresponds to 242570 innermost planet orbits.

from spock import AnalyticalClassifier
analytical_model = AnalyticalClassifier()

print(analytical_model.predict_stable(sim))
# >>> 0.0 #confidently chaotic

from spock import CollisionMergerClassifier
class_model = CollisionMergerClassifier()

prob_12, prob_23, prob_13 = class_model.predict_collision_probs(sim)

print(prob_12, prob_23, prob_13)
# >>> 0.2738345 0.49277353 0.23339202