import logging
import os
import random
import sys
from functools import partialmethod

import hydra
import numpy as np
from omegaconf.omegaconf import OmegaConf
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm
from agents.no_train_agent import NoTrainAgent
from agents.tianshou_agents import TianshouPPOAgent, TianshouDQNAgent, TianshouSACAgent
from agents.dgat import DGATAgent
from agents.ptr_net import PtrAgent
from envs.top_env_gym import TopEnvGym

from datasets.datasets import DataSplit
from datasets.event_log_loader import EventLogLoader
from envs.utils import load_graph_from_file, precompute_shortest_paths
from graph.graph_helpers import create_graph





def simulate(config: DictConfig) -> float:
    """Performs a simulation of the environment."""
    logging.info('Starting simulation.')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False

    if config.agent == "tian_ddqn":
        agent = TianshouDQNAgent(config)
    elif config.agent == "tian_ppo":
        agent = TianshouPPOAgent(config)
    elif config.agent == "tian_sac":
        agent = TianshouSACAgent(config)
    elif config.agent == "aco" or config.agent == "greedy_single":
        agent = NoTrainAgent(config)
    elif config.agent == "dgat":
        agent = DGATAgent(config)
    elif config.agent == "ptr":
        agent = PtrAgent(config)
    else:
        raise RuntimeError(f"Invalid agent {config.agent}")
    
    agent.run()

    return 0.0


def load_data_for_env(config: DictConfig):
    graph_file_name = os.path.join(to_absolute_path(config.path_to_graphs), f"{config.area_name}.gpickle")

    if not os.path.exists(graph_file_name):
        # Creates a new graph
        os.makedirs(os.path.dirname(graph_file_name), exist_ok=True)
        create_graph(config, filename=graph_file_name)
    graph = load_graph_from_file(graph_file_name)
    # Lookup table for the shortest path between two nodes.
    shortest_path_lookup = precompute_shortest_paths(graph)

    event_log = EventLogLoader(graph, config).load()
    return event_log, graph, shortest_path_lookup

def _set_seeds(seed: int):
    logging.info('Setting seed: "%s"', seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)



def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.error("Uncaught exception", exc_info=(
        exc_type, exc_value, exc_traceback))


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def run(config: DictConfig):
    print(f"pid:{os.getpid()}")
    # logging is configured by hydra

    # exceptions will be loggegd to hydra log file
    sys.excepthook = handle_exception

    logging.debug(torch.cuda.is_available())

    # tdqm should not spam error logs
    tqdm.__init__ = partialmethod(tqdm.__init__, file=sys.stdout)

    _set_seeds(config.seed)

    return simulate(config)


if __name__ == '__main__':
    run()
