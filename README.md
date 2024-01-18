# Spatial Aware Deep Reinforcement Learning for the Traveling Officer Problem
This repo contains a simulation environment for the TOP that replays real-world parking events.
Additionally, we implement our approach (SATOP) and several baselines:
* ACO
* Greedy
* DGAT
* PTR
* SDDQN
* SATOP (our method)



## Simulation Environment
Our framework contains a flexible simulation environment for the Traveling Officer Problem. It is written in C++ and has python bindings using pybind11. We also provide an OpenAI gym compatible interface to use in RL frameworks. 
### Dataset
Download the parking restriction and sensor data and bay locations from https://data.melbourne.vic.gov.au/browse?tags=parking.
Then use create_dataset.py and process_data_for_cpp.py to preprocess these files into the correct format for the simulation.

## Configuration
Everything is configured using hydra. Details regarding parameters can be found in 'configuration.md'

## How to run

Note that the environment uses C++. Build everything using 'DEBUG=0 python setup.py install'

Run using:
python main_tian.py -m +experiment=name_of_experiment