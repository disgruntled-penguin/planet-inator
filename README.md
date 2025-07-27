<!--<img src = moments/year_t_42161_0.png width="400" height="400">-->
# Welcome to Planet-inator!

A simplified version of [super earth simulations run by stephen kane](https://manyworlds.space/2023/03/13/what-would-happen-if-our-solar-system-had-a-super-earth-like-many-others-chaos/), you don't have to learn astrophysics to inseert a planet wherever and with whatever dimensions you want



Heres the ui: <br>
<img src = moments/closeup-stable.png width="650" height="425"><br>
close up<br>
<img src = moments/far-calculating.png width="650" height="425"><br>
distant asteroids in green<br>
<img src = moments/sisyphus.png width="650" height="425"><br>
celestial profile bubbles<br>
<img src = moments/ui.png width="650" height="425"><br>
a little choas by the rouge planet<br>
<img src = moments/finish.png width="650" height="375"><br>

try it yourself:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements
python3 render.py
```
change number of asteroids being loaded: (default is 115 combined)
```bash
python3 render.py --nea_asteroids 1000 --distant_Asteroids 450
```
(requires higher GPU performance)




thanks to [rebound](https://rebound.readthedocs.io/en/latest/) for making n-body simulations simpler


all resources: <br>
[Minor Planet Database](https://www.minorplanetcenter.net/mpcops/documentation/) <br>
[dissolution of planetary systems](https://arxiv.org/abs/2101.04117) <br>
[spock](https://arxiv.org/abs/2007.06521) <br>
[Regresser ML](https://arxiv.org/abs/2408.08873) <br>
[Planet 9 predictions](https://arxiv.org/abs/1902.10103) <br>
[pygame gui](https://github.com/MyreMylar/pygame_gui) <br>
