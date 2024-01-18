import itertools

import glob
import sys
from pathlib import Path
from typing import Any, Optional, Union

import wandb
from omegaconf import OmegaConf
from agents.tianshou_agents import TianshouAgent, TianshouDQNAgent
from main_tian import _set_seeds

from datasets.datasets import DataSplit
from agents.ptr_net import PtrAgent
from utils.logging.logger import JSONOutput, Logger

relevant_metric = "validation_advanced_metrics/fined_resources"  # violation_catched_quota fined_resources

areas = ["queensberry", "downtown", "docklands"] #"queensberry", "downtown", "docklands"
number_of_agents = [1]
kinds = ["vanilla"]
use_summary = False

api = wandb.Api(timeout=120)


class WANDBUpdateLogger:
    def __init__(self, run):
        self.run = run
        wandb.init(project="potop", resume="must", id=run.id)

    def add_scalar(self, tag: str, scalar_value: Any, global_step: int, epoch: int = -1, current_step: int = -1):
        wandb.log({tag: scalar_value, "step": global_step, "epoch": epoch, "current_step": current_step}, commit=False)

    def add_video(self,
                  tag: str,
                  vid_tensor: Any,
                  global_step: int,
                  fps: Optional[Union[int, float]] = 4):
        pass

    def add_weight_histogram(self, model, global_step: int, prefix: str = ""):
        pass

    def write(self):
        wandb.log({"save": "save"}, commit=True)

    def close(self):
        wandb.log({"save": "final"}, commit=True)
        wandb.finish()


def find_path_for_run(run):
    search = f"multirun/**/*{run.id}/"
    runs = sorted(glob.glob(search, recursive=True))
    if len(runs) == 0:
        print(f"WARNING: could not find run {run.name} with id {run.id}", file=sys.stderr)
        return None

    run = runs[-1]

    file_path = Path(run) / "../../"
    if not file_path.exists():
        print(f"WARNING: could not find run {run.name} with id {run.id}", file=sys.stderr)
        return None

    return file_path


def evaluate_run(run):
    if "test_eval_final" in run.config:
        print(f"WARNING: already evaluated skip run {run.name} with id {run.id}", file=sys.stderr)
        return

    if run.state == 'running':
        print(f"WARNING: still running skip run {run.name} with id {run.id}", file=sys.stderr)
        return

    path = find_path_for_run(run)

    if path is None:
        return

   # _, full_row_best_run = scan_history_for_metric(run, relevant_metric, only_fetch_relevant_metric=False)

    #best_step = full_row_best_run["step"]
    #best_step = full_row_best_run["epoch"]

    if best_step <= 0:
        all_saved_models = [(int(file.name.split("_")[-2]), int(file.name.split("_")[-1].replace(".pth", ""))) for file
                            in Path(path / "models").glob('*.pth')]
        last_saved_step = max(all_saved_models)[0]
        best_step = last_saved_step

    best_step = int(best_step)

    cfg = OmegaConf.load(path / ".hydra/config.yaml")
    
    # cfg = OmegaConf.create(run.config)
    # #only for create (because some floats 1.0 will be 1 and now int)
    # cfg["distance_normalization"] = float(cfg["distance_normalization"])
    # cfg["speed_in_kmh"] = float(cfg["speed_in_kmh"])
    
    cfg["path_to_model"] = str(Path(path / "models").absolute())
    cfg["load_step"] = best_step
    cfg["load_agent_model"] = True
    cfg["number_of_parallel_envs"] = 1
    _set_seeds(cfg.seed)

    wandb_logger = WANDBUpdateLogger(run)
    json_logger = JSONOutput(log_dir=path)
    output_loggers = [
            json_logger,
            wandb_logger
    ]

    writer = Logger(output_loggers)
    
    if cfg.agent == "tian_ddqn":
        agent = TianshouDQNAgent(cfg, update_writer=writer)
        agent.policy.eval()
        agent.policy.set_eps(0.0)
        agent.evaluator.evaluate(0, 0, agent.test_env, agent.policy, mode="test_final")
    elif cfg.agent =="ptr":
        agent = PtrAgent(cfg, update_writer=writer)
        agent.current_path = None
        agent.evaluator.evaluate(epoch=0, global_step=0,env=agent.test_env, agent=agent, mode="test_final")

    else:
        raise RuntimeError(f"Invalid agent {cfg.agent}")
    
    
    wandb_logger.close()
    json_logger.close()
    #test_env.close()
    
    agent.test_env.close()
    agent.validation_env.close()
    if cfg.agent == "tian_ddqn":
        agent.test_envs.close()

    
    run_for_update = api.run("/".join(run.path))
    run_for_update.config["test_eval_final"] = True
    run_for_update.update()


def execute_single(run_path : str):
    run = api.run(run_path)
    evaluate_run(run)



def execute_list():

    run_ids = [
        "id_of_runs"
    ]


    for run_id in run_ids:
        try:
            execute_single("lmudbs/satop/"+run_id)
        except Exception as e:
            print(f"Exception run with id {run_id}: {e}", file=sys.stderr)


if __name__ == '__main__':
    #execute_single("lmudbs/sato/run_id")
    execute_list()