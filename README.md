# Goals
- Create a reinforcement learning agent that can play Pokemon Red without any external knowledge
- Utilize a reward purely based on exploration

Inspired by Peter Whidden's YouTube video: 
https://www.youtube.com/watch?v=DcYLT37ImBY&ab_channel=PeterWhidden

# Requirements
- Python3
- Pokemon Red ROM
- Linux (or WSL on Windows)

# Usage
It is recommended to use a Python virtual environment. Official documentation is [here](https://docs.python.org/3/library/venv.html).
After installing Python, the venv module can be used to create a virtual environment:
```
python -m venv venv
```
To activate the virtual environment on Linux:
```
source venv/bin/activate
```
To install requirements:
```
pip install -U -r requirements.txt
``` 
To run the program:
```
python main.py
```

# Roadmap
- [x] Create [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) environment for reinforcement learning
- [x] Implement [PyBoy](https://github.com/Baekalfen/PyBoy) for simulation
- [x] Track data of runs for evaluation
- [x] Add persistent, count-based exploration reward (shared across parallel training workers)
- [ ] Tune exploration-reward hash granularity (downsample resolution, quantization levels) based on observed training behavior (now per-run configurable and observable via TensorBoard distinct-state metrics; picking final values needs a real training run, not yet performed)
- [x] Register Gymnasium environment to be able to use ```gym.make()``` ([here](https://gymnasium.farama.org/introduction/create_custom_env/#registering-and-making-the-environment))
- [x] Add map stitching of current area (every ~10 rollouts, discovered-state screenshots from all workers are merged into a persistent `images/master_map.png` and logged to TensorBoard; observational only, no reward/termination/observation impact)
- [ ] Add identification of new areas
  - [ ] Test template matching method
- [ ] Adjust reward based on exploring map with map stitching and new area identification
- [ ] Save video of agent runs ([here](https://gymnasium.farama.org/introduction/record_agent/))
- [ ] Optimize training (consider cnn policy [here](https://stable-baselines.readthedocs.io/en/master/modules/policies.html#stable_baselines.common.policies.CnnPolicy) for gpu acceleration)
