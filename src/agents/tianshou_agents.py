from functools import partial
import logging
import os
import pathlib
from hydra.utils import to_absolute_path
import numpy as np
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf
from tianshou.data.buffer.vecbuf import VectorReplayBuffer
from tianshou.data.collector import AsyncCollector, Collector
from tianshou.env.venvs import DummyVectorEnv, ShmemVectorEnv, SubprocVectorEnv
from tianshou.trainer.offpolicy import OffpolicyTrainer
from tianshou.trainer.onpolicy import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor, Critic
import torch
from datasets.datasets import DataSplit
from datasets.event_log_loader import EventLogLoader
from envs.top_env_gym import TopEnvGym
from envs.utils import edge_to_edge_matrix, get_adjacency_matrix, get_complex_matrix, get_distance_matrix, get_distance_matrix_resources, load_graph_from_file, precompute_shortest_paths
from graph.graph_helpers import create_graph
from modules.tian.test_net import TestNet

from modules.tian.tian_satop import TianTimeAT, TimeATActor, TimeATCritic, SATOPModule
from modules.tian.tian_ssdqngrcn import TianGRCN
from tianshou_helpers.evaluator import TianshouEvaluator
from tianshou_helpers.logger_adapter import TianshouLoggerAdapter
from tianshou_helpers.semi_markov import SemiDQNPolicy, SemiDiscreteSACPolicy, SemiPPOPolicy
from utils.logging.logger import JSONOutput, Logger, WANDBLogger
from utils.python.unmatched_kwargs import ignore_unmatched_kwargs


def load_data_for_env(config: DictConfig, for_cpp=False):    
    graph_file_name = os.path.join(to_absolute_path(
        config.path_to_graphs), f"{config.area_name}.gpickle") #TODO

    if not os.path.exists(graph_file_name):
        # Creates a new graph
        os.makedirs(os.path.dirname(graph_file_name), exist_ok=True)
        create_graph(config, filename=graph_file_name)
    graph = load_graph_from_file(graph_file_name)
    
    if for_cpp:
        return None, graph, None
    else:
        # Lookup table for the shortest path between two nodes.
        shortest_path_lookup = precompute_shortest_paths(graph)

        event_log = EventLogLoader(graph, config).load()
        return event_log, graph, shortest_path_lookup


def _initialize_environment(datasplit: DataSplit, event_log, graph, shortest_path_lookup, config):
    logging.info('Initializing environment.')

    env = TopEnvGym(graph, datasplit, OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True))

    return env


class TianshouAgent:
    def __init__(self, config: DictConfig, update_writer=None):
        self.config = config
        self.device = config.device if torch.cuda.is_available() else 'cpu'
        _, graph, _ = load_data_for_env(config, for_cpp=True)

        self.validation_env = _initialize_environment(
            DataSplit.VALIDATION, None, None, None, config)
        self.test_env = _initialize_environment(
            DataSplit.TEST, None, None, None, config)

        self.train_num = config.number_of_parallel_envs
        train_envs_list = [partial(_initialize_environment, DataSplit.TRAINING,
                                   None, None, None, config) for _ in range(self.train_num)]
        if self.train_num == 1 or config.force_dummy_vec_env:
            self.train_envs = DummyVectorEnv(train_envs_list)
        else:
            #self.train_envs = ShmemVectorEnv(train_envs_list)
            self.train_envs = SubprocVectorEnv(train_envs_list) #do not use the shared memory version because it creates a dummy env just to init all the buffers...
            

        self.test_envs = DummyVectorEnv([partial(_initialize_environment, DataSplit.TEST,
                                                 None, None, None, config) for _ in range(1)])

        # TODO
        if config.model.name == "MLP":
            self.net = TestNet(self.test_env.observation_space,
                               device=self.device).to(self.device)
        elif config.model.name == "GRCN":
            self.net = TianGRCN(self.test_env.observation_space, self.device, torch.Tensor(
                get_distance_matrix(graph)), config.model.grcn).to(self.device)
        elif config.model.name == "TimeAT":
            self.net = TianTimeAT(torch.Tensor(get_distance_matrix(graph)), torch.Tensor(
                get_adjacency_matrix(graph)), torch.Tensor(get_distance_matrix_resources(graph)), self.test_env.observation_space, self.device, config.model)
        elif config.model.name == "SATOP":
            self.net = SATOPModule(torch.Tensor(edge_to_edge_matrix(graph)).to(self.device), 
                                 torch.Tensor(get_adjacency_matrix(graph)).to(self.device), 
                                 torch.Tensor(get_distance_matrix(graph)).to(self.device), 
                                 torch.from_numpy(get_complex_matrix(graph)).to_sparse().to(self.device),
                                 self.test_env.observation_space, 
                                 self.device, 
                                 config.model,
                                 config.area_name).to(self.device)
        else:
            raise RuntimeError(f"Unknown model: {config.model.name}")

        if update_writer is None:
            output_loggers = [
                #TensorboardOutput(log_dir=".", comment=f""),
                JSONOutput(log_dir=os.getcwd())
            ]

            if not config.experiment_name == "debug":
                output_loggers.append(WANDBLogger(config=config))
            self.writer = Logger(output_loggers)
        else:
            self.writer = update_writer
            
        save_model_dir = pathlib.Path(config.save_model_dir).expanduser()
        save_model_dir.mkdir(parents=True, exist_ok=True)
        self.save_model_path = save_model_dir.absolute()

        self.logger = TianshouLoggerAdapter(self.writer, update_interval=100)
        self.evaluator = TianshouEvaluator(self.writer)
        
    def run(self):
        pass


class TianshouDQNAgent(TianshouAgent):
    def __init__(self, config: DictConfig, update_writer=None):
        super().__init__(config, update_writer)

        self.optim = ignore_unmatched_kwargs(getattr(torch.optim, config.model_optimizer.optimizer))(
            self.net.parameters(), **config.model_optimizer)
        self.policy = SemiDQNPolicy(self.net,
                                    self.optim,
                                    discount_factor=config.gamma,
                                    estimation_step=1,  #set to 1 because of semi-markov
                                    target_update_freq=config.update_target_every,
                                    clip_loss_grad=False)
        if config.load_agent_model:
            model = torch.load(f"{config.path_to_model}/checkpoint_{config.load_step}.pth", map_location=self.device)
            self.policy.load_state_dict(model["model"])

        self.train_collector = AsyncCollector(self.policy, self.train_envs,
                                              VectorReplayBuffer(
                                                  config.replay_size, 
                                                  len(self.train_envs)
                                                  ), 
                                              exploration_noise=True)
        self.test_collector = Collector(
            self.policy, self.test_envs, exploration_noise=False)

        epsilon_initial = config.epsilon_initial
        epsilon_min = config.epsilon_min
        epsilon_decay_start = config.epsilon_decay_start
        steps_till_min_epsilon = config.steps_till_min_epsilon
       
        eps_func = None
        if config.epsilon_decay == "exp":
            exp_scaling = (-1) * steps_till_min_epsilon / \
                np.log(epsilon_min) if epsilon_min > 0 else 1

            eps_func = lambda env_step: min(epsilon_initial, max(epsilon_min,np.exp(- (env_step - epsilon_decay_start) / exp_scaling)))
        elif config.epsilon_decay == "linear":
            per_step_decay = (epsilon_initial- epsilon_min) / steps_till_min_epsilon
            eps_func = lambda env_step: min(epsilon_initial, max(epsilon_min, epsilon_initial - (per_step_decay * (env_step - epsilon_decay_start))))
        else:
            raise RuntimeError(f"unkown eps decay func {config.epsilon_decay}")
        
        def train_fn(epoch, env_step):
            eps = eps_func(env_step)

            self.policy.set_eps(eps)
            if env_step % 1000 == 0:
                self.logger.write("train/env_step", env_step,
                                  {"train/eps": eps})


        def save_checkpoint_fn(epoch, env_step, gradient_step):
            # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
            ckpt_path = os.path.join(
                self.save_model_path, f"checkpoint_{epoch}.pth")
            torch.save({"model": self.policy.state_dict()}, ckpt_path)
            return ckpt_path
        
        def test_fn(epoch, env_step):
            self.policy.set_eps(0.0)
            self.evaluator.evaluate(
                epoch, env_step, self.validation_env, self.policy, mode="validation")
            
            save_checkpoint_fn(epoch, env_step, None)

        
        
        self.trainer = OffpolicyTrainer(
            self.policy,
            self.train_collector,
            self.test_collector,
            max_epoch=config.max_epoch,
            step_per_epoch=config.step_per_epoch,
            step_per_collect=config.step_per_collect,
            episode_per_test=1,
            batch_size=config.batch_size,
            update_per_step=1.0/config.step_per_collect,
            train_fn=train_fn,
            test_fn=test_fn,
            logger=self.logger,
            test_in_train=False,
            save_checkpoint_fn=save_checkpoint_fn
        )

    def run(self):
        self.train_collector.collect(
            n_step=self.config.batch_size * self.train_num)

        self.trainer.run()

        self.policy.eval()
        self.policy.set_eps(0.0)
        ckpt_path = os.path.join(
                self.save_model_path, f"checkpoint_final.pth")
        torch.save({"model": self.policy.state_dict()}, ckpt_path)

        self.evaluator.evaluate(0, 0, self.validation_env,
                                self.policy, mode="validation")
        self.evaluator.evaluate(0, 0, self.test_env, self.policy, mode="test")

        logging.info("Simulation finished.")
        self.writer.close()


class TianshouPPOAgent(TianshouAgent):
    def __init__(self, config: DictConfig):
        super().__init__(config)

        if config.model.name == "TimeAT":
            actor = TimeATActor(
                self.net, self.test_env.action_space.n, config.model).to(self.device)
            critic = TimeATCritic(self.net).to(self.device)
        else:
            actor = Actor(self.net, self.test_env.action_space.n,
                          device=self.device).to(self.device)
            critic = Critic(self.net, device=self.device).to(self.device)

        actor_critic = ActorCritic(actor, critic)

        # optimizer of the actor and the critic
        optim = ignore_unmatched_kwargs(getattr(torch.optim, config.model_optimizer.optimizer))(
            actor_critic.parameters(), **config.model_optimizer)
        dist = torch.distributions.Categorical
        self.policy = SemiPPOPolicy(actor,
                                    critic,
                                    optim,
                                    dist,
                                    action_space=self.test_env.action_space,
                                    observation_space=self.test_env.observation_space,
                                    deterministic_eval=config.deterministic_eval,
                                    reward_normalization=config.reward_normalization,
                                    gae_lambda=config.gae_lambda,
                                    advantage_normalization=config.advantage_normalization,
                                    discount_factor=config.gamma,
                                    vf_coef=config.vf_coef,
                                    ent_coef=config.ent_coef,
                                    max_batchsize=config.max_batchsize,
                                    )

        self.train_collector = AsyncCollector(
            self.policy, self.train_envs, VectorReplayBuffer(config.buffer_size, len(self.train_envs)))
        self.test_collector = Collector(self.policy, self.test_envs)

        def save_checkpoint_fn(epoch, env_step, gradient_step):
            # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
            ckpt_path = os.path.join(
                self.save_model_path, f"checkpoint_{epoch}.pth")
            torch.save({"model": self.policy.state_dict()}, ckpt_path)
            return ckpt_path

        self.trainer = OnpolicyTrainer(
            self.policy,
            self.train_collector,
            self.test_collector,
            max_epoch=config.max_epoch,
            step_per_epoch=config.step_per_epoch,
            repeat_per_collect=config.repeat_per_collect,
            episode_per_test=1,
            batch_size=config.batch_size,
            step_per_collect=config.step_per_collect,
            test_fn=lambda epoch, global_step: self.evaluator.evaluate(
                epoch, global_step, self.validation_env, self.policy, mode="validation"),  # TODO
            logger=self.logger,
            # stop_fn=lambda mean_reward: mean_reward >= 195,
            test_in_train=False,
            save_checkpoint_fn=save_checkpoint_fn
        )

    def run(self):
        self.trainer.run()

        self.policy.eval()
        self.evaluator.evaluate(0, 0, self.validation_env,
                                self.policy, mode="validation")
        self.evaluator.evaluate(0, 0, self.test_env, self.policy, mode="test")

        logging.info("Simulation finished.")
        self.writer.close()


class TianshouSACAgent(TianshouAgent):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        if config.model.name == "TimeAT":
            actor = TimeATActor(self.net, self.test_env.action_space.n,
                                config.model, softmax_output=False).to(self.device)
            critic1 = TimeATCritic(
                self.net, last_size=self.test_env.action_space.n).to(self.device)
            critic2 = TimeATCritic(
                self.net, last_size=self.test_env.action_space.n).to(self.device)
        else:
            actor = Actor(self.net, self.test_env.action_space.n,
                          device=self.device, softmax_output=False).to(self.device)
            critic1 = Critic(
                self.net, last_size=self.test_env.action_space.n, device=self.device).to(self.device)
            critic2 = Critic(
                self.net, last_size=self.test_env.action_space.n, device=self.device).to(self.device)

        actor_optim = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)
        critic1_optim = torch.optim.Adam(
            critic1.parameters(), lr=config.critic_lr)
        critic2_optim = torch.optim.Adam(
            critic2.parameters(), lr=config.critic_lr)

        auto_alpha = config.auto_alpha
        alpha = config.alpha
        if auto_alpha:
            target_entropy = 0.98 * np.log(self.test_env.action_space.n)
            log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=config.alpha_lr)
            alpha = (target_entropy, log_alpha, alpha_optim)

        # self.optim = ignore_unmatched_kwargs(getattr(torch.optim, config.model_optimizer.optimizer))(
        #     self.net.parameters(), **config.model_optimizer)
        self.policy = SemiDiscreteSACPolicy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau=config.tau,
            gamma=config.gamma,
            alpha=alpha,
            estimation_step=1,
            reward_normalization=config.reward_normalization).to(self.device)

        self.train_collector = AsyncCollector(self.policy, self.train_envs, VectorReplayBuffer(
            config.replay_size, len(self.train_envs)), exploration_noise=True)
        self.test_collector = Collector(
            self.policy, self.test_envs, exploration_noise=False)

        def test_fn(epoch, env_step):
            self.evaluator.evaluate(
                epoch, env_step, self.validation_env, self.policy, mode="validation")

        def save_checkpoint_fn(epoch, env_step, gradient_step):
            # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
            ckpt_path = os.path.join(
                self.save_model_path, f"checkpoint_{epoch}.pth")
            torch.save({"model": self.policy.state_dict()}, ckpt_path)
            return ckpt_path

        self.trainer = OffpolicyTrainer(
            self.policy,
            self.train_collector,
            self.test_collector,
            max_epoch=config.max_epoch,
            step_per_epoch=config.step_per_epoch,
            step_per_collect=config.step_per_collect,
            episode_per_test=1,
            batch_size=config.batch_size,
            update_per_step=1.0/config.step_per_collect,
            test_fn=test_fn,
            logger=self.logger,
            test_in_train=False,
            save_checkpoint_fn=save_checkpoint_fn
        )

    def run(self):
        self.train_collector.collect(
            n_step=self.config.batch_size * self.train_num)

        self.trainer.run()

        self.policy.eval()
        self.evaluator.evaluate(0, 0, self.validation_env,
                                self.policy, mode="validation")
        self.evaluator.evaluate(0, 0, self.test_env, self.policy, mode="test")

        logging.info("Simulation finished.")
        self.writer.close()
