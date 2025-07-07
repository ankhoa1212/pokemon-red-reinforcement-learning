# Goals
- Create a reinforcement learning agent that can play Pokemon Red without any external knowledge
- Utilize a reward purely based on exploration

Inspired by Peter Whidden's YouTube video: 
https://www.youtube.com/watch?v=DcYLT37ImBY&ab_channel=PeterWhidden

# Requirements
- Python3
- Pokemon Red ROM

# Usage
It is recommended to use a Python virtual environment. Official documentation is [here](https://docs.python.org/3/library/venv.html).

Once the Python virtual environment has been activated, run the following commands:

To install requirements: ```pip install -r requirements.txt``` 

To run the program: ```python3 main.py```

# Roadmap
- [x] Create gymnasium environment for reinforcement learning
- [x] Implement PyBoy for simulation
- [x] Track data of runs for evaluation
- [ ] Save video of agent runs
- [x] Add simple exploration reward
- [x] Add map stitching of current area
- [ ] Add identification of new areas
- [ ] Adjust reward based on exploring map with map stitching
- [ ] Optimize training (consider cnn for gpu acceleration)
