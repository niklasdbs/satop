from typing import Iterable, Iterator, Union
from pettingzoo import AECEnv
from top_env import TOPEnv
from top_env import DataSplit as CDataSplit
from gym import spaces
from datasets.datasets import DataSplit
import logging


class CAECIterator(Iterator):
    def __init__(self, cenv, max_iter):
        self.cenv = cenv
        self.iters_til_term = max_iter

    def __next__(self):
        agent_selection = self.cenv.current_agent_selection()
        if agent_selection < 0 or self.iters_til_term <= 0:
            raise StopIteration
        self.iters_til_term -= 1
        return agent_selection

    def __iter__(self):
        return self

class CAECIterable(Iterable):
    def __init__(self, cenv, max_iter):
        self.cenv = cenv
        self.max_iter = max_iter

    def __iter__(self):
        return CAECIterator(self.cenv, self.max_iter)



class TopEnvPy(AECEnv):
    def __init__(self, graph, split, config):
        csplit = None
        if split == DataSplit.TRAINING:
            csplit = CDataSplit.TRAINING
        elif split == DataSplit.VALIDATION:
            csplit = CDataSplit.VALIDATION
        elif split == DataSplit.TEST:
            csplit = CDataSplit.TEST
        self.split = split
        self.cenv = TOPEnv(csplit, config)
        self.graph = graph #TODO check/be carefull that graphs, actions, ... match!!
        self.possible_agents = {i:i for i in range(self.cenv.number_of_agents())}
        logging.debug('Action space: %s', self.action_space(0))
        logging.debug('Observation space: %s', self.observation_space(0))
        
    def observation_space(self, agent):
        """
        Takes in agent and returns the observation space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the observation_spaces dict
        """
        return spaces.Dict(spaces={
            key: spaces.Box(low=-1, high=1, shape=value)
            for key, value in self.cenv.observation_shape().items()
        })

    def action_space(self, agent):
        """
        Takes in agent and returns the action space for that agent.

        MUST return the same value for the same agent name

        Default implementation is to return the action_spaces dict
        """
        return spaces.Discrete(self.cenv.number_of_actions())

    def step(self, action):
        return self.cenv.step(action)
    
    def reset(self, reset_days=False, only_do_single_episode=False) -> Union[bool, dict]:
        return self.cenv.reset(reset_days, only_do_single_episode)

    def last(self, agent=None, observe=True):
        if agent is None:
            agent = -1
        
        return self.cenv.last(agent, observe)
    
    def agent_iter(self, max_iter: int = 2 ** 63) -> CAECIterable:
        return CAECIterable(self.cenv, max_iter)
    
    @property
    def agents(self):
        return self.cenv.active_agents()
    
    @property
    def final_advanced_metrics(self):
        return self.cenv.get_final_advanced_metrics()
