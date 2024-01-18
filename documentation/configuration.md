# Configuration
This document details configuration setting.
## Misc

### experiment_name
Set the name of an experiment used in logging

## Environment

### speed_in_kmh
The walking speed of an agent in kmh.

### gamma
discount factor

### semi_markov
using temporal (semi-markov) discounting
### Common Parameters
#### batch_size
The size of a batch used for learning
#### train steps
The number of training iterations whenever an agent trains.
#### start_learning
Number of steps when an agent starts learning.

### Exploration
#### epsilon_initial
Number of steps until the epsilon decay starts.
#### epsilon_min
Minimum value of epsilon.
#### epsilon_decay_start
Number of steps until the epsilon decay starts.
#### steps_till_min_epsilon
Steps until the minimum epsilon value should be reached.
#### epsilon_decay
Exponential or linear decay. Possible values: exp,linear



## Agents
* ACO
* Greedy
* DGAT
* PTR
* SDDQN
* SATOP
* Tianshou for RL-based agents


## Logging
### JSON
### WANDB

## Misc
### Areas
* Docklands
* Queensberry
* Downtown