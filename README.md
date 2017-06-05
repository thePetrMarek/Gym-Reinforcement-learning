# Gym Reinforcement learning
Set of reinforcement learning algorithms for solving OpenAI Gym.
 
## Quick start
You need Python3 with Gym. Some agents need more, like Tensorflow.

Open ``main.py`` and select learning agent and hyperparameters.

Run by: 
    
    py -3 main.py
      
## Structure
There are several reinforcement learning agents implemented in folders.

### Random agent
Random agents select random action each step.

### DeepQ agent
DeepQ agent has Q-learning algorithm. Values of Q(s,a) are approximated 
by neural neural network.
