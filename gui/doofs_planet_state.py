from threading import Lock

class DoofsPlanetState:
    def __init__(self):
        self.lock = Lock()
        self.mass = 400
        self.size = 0.07
        self.color = (128, 0, 128, 255)
        self.a = 3.3
        self.e = 0.3
        self.inc = 15
        self.Omega = 80
        self.omega = 60
        self.f = 10

doofs_planet_state = DoofsPlanetState() 