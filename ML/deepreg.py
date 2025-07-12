from reboundsim import Simulation
sim = Simulation()
sim.sim.move_to_com()

from spock import DeepRegressor
deep_model = DeepRegressor()

median, lower, upper, samples = deep_model.predict_instability_time(
    sim.sim, samples=10000, return_samples=True, seed=0
)
print((median*88) / 365.25) # in human years #5600