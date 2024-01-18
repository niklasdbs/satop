import os
from omegaconf import DictConfig
from agents.aco import ACO
from agents.greedy_single import GreedySingle
from agents.tianshou_agents import _initialize_environment, load_data_for_env
from datasets.datasets import DataSplit

from utils.logging.logger import JSONOutput, Logger, WANDBLogger
import logging
import numpy as np
from tqdm import tqdm

class NoTrainEvaluator:
    def __init__(self, logger: Logger) -> None:
        self.writer = logger
        self.eval_episodes = 1 #TODO do not hardcode
        self.render = False #TODO do not hardcode
    
    def log_advanced_metrics(self, epoch, global_step, env, mode):
        advanced_metrics = env.final_advanced_metrics
        for key, values in advanced_metrics.items():
            self.writer.add_scalar(f"{mode}_advanced_metrics/{key}", values, global_step=global_step, epoch=epoch, current_step=global_step)

        return advanced_metrics

    def evaluate(self, epoch, global_step, env, agent, mode="validation"):
        logging.info(f"evaluate using mode {mode}")
        self.writer.write()
        result = 0.0

        render = self.render and mode == "test"

        for episode in tqdm(range(self.eval_episodes)):
            first_reset_in_episode = True  # this is used to set the flag of only doing a

            while True:
                env_reset_result, _ = env.reset(options={"reset_days": first_reset_in_episode, "only_do_single_episode":True})
                if hasattr(agent, 'current_path'):
                    agent.current_path = None
                
                first_reset_in_episode = False

                if env_reset_result == False:
                    break

                if render:
                    video = self._create_cv_writer(env, episode)

                
                observation = env_reset_result
                while True:
                    action = agent.act(observation)
                    
                    
                    observation, reward, terminated, truncated, info = env.step(action)
                    
                    if terminated or truncated:
                        break
                    
                    

                if render:
                    #cv2.destroyAllWindows()
                    video.release()

            self.log_advanced_metrics(epoch, global_step, env, mode)

        self.writer.write()

        return result / self.eval_episodes

class NoTrainAgent:
    def __init__(self, config: DictConfig) -> None:
        _, graph, _ = load_data_for_env(config, for_cpp=True) #TODO do not hardcode cppp flag

        self.validation_env = _initialize_environment(
            DataSplit.VALIDATION, None, None, None, config)
        self.test_env = _initialize_environment(
            DataSplit.TEST, None, None, None, config)

        if config.agent == "aco":
            self.agent = ACO(config, graph, self.test_env)
        elif config.agent == "greedy_single":
            self.agent = GreedySingle(config, graph, self.test_env)
        else:
            raise RuntimeError(f"unkown agent {config.agent}")
        
        output_loggers = [
            #TensorboardOutput(log_dir=".", comment=f""),
            JSONOutput(log_dir=os.getcwd())
        ]

        if not config.experiment_name == "debug":
            output_loggers.append(WANDBLogger(config=config))

        self.writer = Logger(output_loggers)
        self.evaluator = NoTrainEvaluator(self.writer)

    def run(self):
        self.evaluator.evaluate(0, 0, self.validation_env,
                                self.agent, mode="validation")
        self.evaluator.evaluate(0, 0, self.test_env, self.agent, mode="test")

        logging.info("Simulation finished.")
        self.writer.close()
