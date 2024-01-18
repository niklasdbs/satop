import gymnasium
from top_env import TOPEnv
from top_env import DataSplit as CDataSplit
from datasets.datasets import DataSplit
import logging
from gymnasium.core import ObsType
from gymnasium.spaces import Box, Dict, Discrete
from typing import Any, Optional, Tuple, Union

class TopEnvGym(gymnasium.Env):
    def __init__(self, graph, split, config) -> None:
        super().__init__()
        csplit = None
        if split == DataSplit.TRAINING:
            csplit = CDataSplit.TRAINING
        elif split == DataSplit.VALIDATION:
            csplit = CDataSplit.VALIDATION
        elif split == DataSplit.TEST:
            csplit = CDataSplit.TEST

        self.cenv = TOPEnv(csplit, config)

        self.observation_space = Dict(spaces={
            key: Box(low=-1, high=1, shape=value)
            for key, value in self.cenv.observation_shape().items()
        })
        
        self.action_space = Discrete(self.cenv.number_of_actions())

        logging.debug('Action space: %s', self.action_space)
        logging.debug('Observation space: %s', self.observation_space)

    def step(self, action: int) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        self.cenv.step(action)
        
        observation, reward, discounted_reward, terminated, info =  self.cenv.last(-1, True)
        
        return observation, discounted_reward, terminated, False,  info
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        if options:
            reset_days = options["reset_days"]
            only_do_single_episode = options["only_do_single_episode"]
            
            obs = self.cenv.reset(reset_days, only_do_single_episode)
            
            if obs:
                return obs[0], {}
            else:
                return obs, {}
        else:
            obs = self.cenv.reset(False, False)
            return obs[0], {}

    @property
    def final_advanced_metrics(self):
        return self.cenv.get_final_advanced_metrics()

    
    def get_resource_id_to_edge_id_mapping(self):
        return self.cenv.get_resource_id_to_edge_id_mapping()
    
    def get_edge_id_to_action_mapping(self):
        return self.cenv.get_edge_id_to_action_mapping()