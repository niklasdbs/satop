from utils.logging.logger import Logger
import logging
import torch
from tianshou.data.batch import Batch
import numpy as np
from tqdm import tqdm

class TianshouEvaluator:
    def __init__(self, logger: Logger) -> None:
        self.writer = logger
        self.eval_episodes = 1 #TODO
        self.render = False #TODO
    
    def log_advanced_metrics(self, epoch, global_step, env, mode):
        advanced_metrics = env.final_advanced_metrics
        for key, values in advanced_metrics.items():
            self.writer.add_scalar(f"{mode}_advanced_metrics/{key}", values, global_step=global_step, epoch=epoch, current_step=global_step)

        return advanced_metrics

    def evaluate(self, epoch, global_step, env, policy, mode="validation"):
        logging.info(f"evaluate using mode {mode}")
        self.writer.write()
        result = 0.0

        render = self.render and mode == "test"

        with torch.no_grad():
            for episode in tqdm(range(self.eval_episodes)):
                first_reset_in_episode = True  # this is used to set the flag of only doing a

                while True:
                    env_reset_result, _ = env.reset(options={"reset_days": first_reset_in_episode, "only_do_single_episode":True})
                    first_reset_in_episode = False

                    if env_reset_result == False:
                        break

                    if render:
                        video = self._create_cv_writer(env, episode)

                    
                    observation = env_reset_result
                    while True:
                        batch = Batch({"obs": np.expand_dims(observation, 0), "state": {}, "info": {}})
                        with torch.no_grad():
                            action_batch = policy.forward(batch)

                        action =  action_batch["act"].item() #, action_batch["state"]

                        observation, reward, terminated, truncated, info = env.step(action)
                        
                        if terminated or truncated:
                            break
                        
                        

                    if render:
                        #cv2.destroyAllWindows()
                        video.release()

                self.log_advanced_metrics(epoch, global_step, env, mode)

        self.writer.write()

        return result / self.eval_episodes
