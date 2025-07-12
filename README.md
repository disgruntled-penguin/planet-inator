<img src = moments/year_t_42161_0.png width="400" height="400">


# 🪐 Planet-Inator: Simulate Orbital Chaos

<!-->> *"Curse you, Doofenshmirtz! You broke the Solar System!"*-->

Planet-Inator is a real-time, interactive orbital simulator that lets users introduce a new planet — **Doof’s Planet** — into the early Solar System and visualize how it destabilizes or survives across billions of years.


---

## Core Idea



---

## 🚀 Features

- 🌍 **Accurate Planetary Initialization**  
  Real orbital parameters for all 8 planets, derived from NASA JPL and Horizons.

- 🧠 **Fully Interactive Pygame GUI**  
  Customize "Doof's Planet" (mass, eccentricity, inclination, etc.) with live sliders and inputs.

- ⏱️ **Fast-forward Through Time**  
  Adjustable simulation speed: simulate hundreds of years per second.

- 🧮 **N-body Physics with REBOUND**  
  Uses `whfast`, a symplectic integrator optimized for long-term simulations of planetary orbits.

- 🌌 **Dynamic Visualization**  
  Real-time rendering of orbits with zoom, pause, and starfield toggle.

- 💥 **Scientific Legitimacy**  
  Grounded in long-term stability studies of our Solar System. This isn’t just a space toy — it’s a physics lab in disguise.

---

## 🧑‍🔬 Scientific Background

- Based on chaos theory in celestial mechanics: even small perturbations can destroy the Solar System over millions of years.
- Doof’s Planet is designed to **maximize orbital instability**:
  - 400 Earth masses (bigger than Jupiter)
  - Semi-major axis: 3.3 AU (between Mars and Jupiter)
  - High eccentricity and inclination to promote orbital crossings and resonance disruptions

- Professional simulations (Laskar 2008, Levison et al. 2011, Batygin & Laughlin 2015) show that introducing new planets — even small ones — can result in planetary ejections, collisions, or chaotic long-term drift.

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `sim.py` | Initializes the simulation using REBOUND. Includes all 8 planets + Doof’s Planet |
| `controllers.py` | Interactive GUI built with `pygame_gui`, sliders for Doof's parameters |
| `render.py` | Visualization logic: drawing orbits, handling frame updates |
| `gui/theme.json` | Custom UI theme for the GUI controls |

---

## 🛠 Setup Instructions

1. **Clone the repo**  
   ```bash
   git clone https://github.com/disgruntled-penguin/planet-inator
   cd planet-inator



