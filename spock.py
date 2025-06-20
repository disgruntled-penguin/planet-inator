# spock demo 
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

