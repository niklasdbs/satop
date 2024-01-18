import os
import pathlib
from typing import Iterator
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import IterableDataset
from hydra.utils import to_absolute_path
from tqdm import tqdm
from agents.no_train_agent import NoTrainEvaluator
from agents.tianshou_agents import _initialize_environment, load_data_for_env

from datasets.datasets import DataSplit
from agents.dgat import _is_in_data_split
from envs.utils import get_distance_matrix, get_distance_matrix_resources
from utils.logging.logger import JSONOutput, Logger, WANDBLogger
# adapted from https://github.com/Rintarooo/TSP_DRL_PtrNet

class PtrDataSet(IterableDataset):
    def __init__(self, graph, config, distance_matrix_resources, per_day_events, distance_matrix, data_split = DataSplit.TRAINING) -> None:
        
        self.area_name = config.area_name
        self.distance_matrix_resources = distance_matrix_resources
        self.per_day_events = per_day_events

        self.days = [day for day in range(1, 366) if _is_in_data_split(day, data_split)] 
        self.start = config.start_hour * 60 * 60
        self.end = config.end_hour * 60 * 60
        
        self.speed_in_ms = config.speed_in_kmh / 3.6
        
        
        
        resources = pd.read_csv(to_absolute_path(f"../data/{config.year}/{self.area_name}_resources.csv"))
        resources["x"] = (resources["x"] - resources["x"].min()) / (resources["x"].max() - resources["x"].min())
        resources["y"] = (resources["y"] - resources["y"].min()) / (resources["y"].max() - resources["y"].min())

        self.resource_observation = resources[["x", "y"]].to_numpy()
        

    def __iter__(self) -> Iterator:
        while True:
            yield self.get_problem_instance()

    def get_problem_instance(self):
        #current_time = np.random.randint(low=self.start, high=self.end, size=1)
        #current_day = np.random.choice(self.days, size=1, replace=True)
        current_day = np.random.choice(self.days)
        current_time = np.random.randint(low=self.start, high=self.end)
        current_pos = np.random.randint(low=0, high=self.resource_observation.shape[0])
        start_point = np.random.randint(self.distance_matrix_resources.shape[0])
        
        events_per_resource = self.per_day_events[current_day]
        
        violation_status = []        
        for r_id, events in enumerate(events_per_resource):
            relevant_events = events[(events[...,0] < current_time)]
            
            if len(relevant_events) == 0:
                in_violation = False
            else:
                last_event = relevant_events[-1]
                    
                in_violation = last_event[3] == 1

            
            if in_violation:
                violation_status.append(1)
            else:
                violation_status.append(0)
                
                
        # new_distance_matrix = np.zeros((len(resources_in_violation), len(resources_in_violation)))
        
        # for viol_r_id, real_r_id in violation_resource_id_to_real_resource_id_mapping.items():
        #     for viol_id_target, real_r_id_target in violation_resource_id_to_real_resource_id_mapping.items():
        #         new_distance_matrix[viol_r_id, viol_id_target] = self.distance_matrix_resources[real_r_id, real_r_id_target]
        
        return current_time, current_day, current_pos, np.hstack((self.resource_observation, np.array(violation_status)[:, None])), self.resource_observation[start_point]
        
        
    def evaluate_solution(self, solution_path, current_time, current_day, start_pos):
        current_time = current_time
        prev_r_id = 0
        
        fined_resources = 0
        
        start = True
        
        for viol_r_id in solution_path:
            if start:
                travel_time = self.distance_matrix_resources[start_pos, viol_r_id] / self.speed_in_ms
                start = False
            else:
                travel_time = self.distance_matrix_resources[prev_r_id, viol_r_id] / self.speed_in_ms
            
            current_time += int(travel_time)
            
            if current_time > self.end:
                break
            
            events = self.per_day_events[current_day][viol_r_id]
            
            relevant_events = events[(events[...,0] < current_time)]

            if len(relevant_events) == 0:
                pass #not in violation nothing to do here
            else:
                last_event = relevant_events[-1]
                in_violation = last_event[3] == 1
                
                if in_violation:
                    fined_resources += 1
                    
            prev_r_id = viol_r_id
                    
                    
        return 1.0 - fined_resources/len(solution_path)

    
    def interactive_solution(self, action, current_time, current_day, current_pos):        
        travel_time = self.distance_matrix_resources[current_pos, action] / self.speed_in_ms
        current_time += travel_time
        
        events_per_resource = self.per_day_events[current_day]

        violation_status = []        
        for r_id, events in enumerate(events_per_resource):
            relevant_events = events[(events[...,0] < current_time)]
            
            if len(relevant_events) == 0:
                in_violation = False
            else:
                last_event = relevant_events[-1]
                    
                in_violation = last_event[3] == 1

            
            if in_violation:
                violation_status.append(1)
            else:
                violation_status.append(0)
                
                
        # new_distance_matrix = np.zeros((len(resources_in_violation), len(resources_in_violation)))
        
        # for viol_r_id, real_r_id in violation_resource_id_to_real_resource_id_mapping.items():
        #     for viol_id_target, real_r_id_target in violation_resource_id_to_real_resource_id_mapping.items():
        #         new_distance_matrix[viol_r_id, viol_id_target] = self.distance_matrix_resources[real_r_id, real_r_id_target]
        
        return current_time, np.hstack((self.resource_observation, np.array(violation_status)[:, None]))

        
        


class PtrNet1(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.Embedding = nn.Linear(3, cfg.embed, bias=False)
        # self.Encoder = nn.LSTM(input_size=cfg.embed,
        #                       hidden_size=cfg.hidden, batch_first=True)
        self.Decoder = nn.LSTM(input_size=cfg.embed,
                               hidden_size=cfg.hidden, batch_first=True)
        self.Vec = nn.Parameter(torch.FloatTensor(cfg.embed))
        self.Vec2 = nn.Parameter(torch.FloatTensor(cfg.embed))
        self.W_q = nn.Linear(cfg.hidden, cfg.hidden, bias=True)
        self.W_ref = nn.Conv1d(cfg.hidden, cfg.hidden, 1, 1)
        self.W_q2 = nn.Linear(cfg.hidden, cfg.hidden, bias=True)
        self.W_ref2 = nn.Conv1d(cfg.hidden, cfg.hidden, 1, 1)
        self.dec_input = nn.Parameter(torch.FloatTensor(cfg.embed))
        self._initialize_weights(cfg.init_min, cfg.init_max)
        self.clip_logits = cfg.clip_logits
        self.softmax_T = cfg.softmax_T
        self.n_glimpse = cfg.n_glimpse
        
        
        self.resource_id_to_edge_id_mapping = None #self.validation_env.get_resource_id_to_edge_id_mapping()
        self.edge_id_to_action_mapping = None #self.validation_env.get_edge_id_to_action_mapping()

    def _initialize_weights(self, init_min=-0.08, init_max=0.08):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)

    def forward(self, current_time, current_day, current_pos, resource_observations, data_set, device, start_point=None, greedy=False, encode_future=True, use_current_mask=False):
        '''	x: (batch, city_t, 2)
            enc_h: (batch, city_t, embed)
            dec_input: (batch, 1, embed)
            h: (1, batch, embed)
            return: pi: (batch, city_t), ll: (batch)
        '''
        x = resource_observations
        
        batch, city_t, _ = x.size()
        embed_enc_inputs = self.Embedding(x)
        embed = embed_enc_inputs.size(2)
        mask = torch.zeros((batch, city_t), device=x.device)
        #enc_h, (h, c) = self.Encoder(embed_enc_inputs, None)
        hidden_state = None
        ref = embed_enc_inputs
        pi_list, log_ps = [], []
        
        if use_current_mask:
            mask = (x[..., 2] - 1) * -1 #we want to mask out violations and not the rest...
        
        
        if start_point is None:
            dec_input = self.dec_input.unsqueeze(0).repeat(batch, 1).unsqueeze(1)
        else:
            tmp = torch.zeros((batch, city_t, 3), dtype=torch.float32, device=x.device)
            tmp[..., :2]  = start_point
            dec_input = self.Embedding(tmp)
        for i in range(city_t):
            _, (h, c) = self.Decoder(dec_input, hidden_state)
            hidden_state = (h, c)
            query = h.squeeze(0)
            
            for i in range(self.n_glimpse):
                query = self.glimpse(query, ref, mask)
            logits = self.pointer(query, ref, mask)
            log_p = torch.log_softmax(logits, dim = -1)

            if greedy:
                next_node = torch.argmax(log_p, dim = 1).long()
            else:
                next_node = torch.distributions.Categorical(logits=log_p).sample().long()
            
                              
            pi_list.append(next_node)
            log_ps.append(log_p)
            #mask += torch.zeros((batch,city_t), device = x.device).scatter_(dim = 1, index = next_node.unsqueeze(1), value = 1)
            mask = mask.scatter(1, next_node.unsqueeze(-1), 1)

            if encode_future:
                current_time, new_obs =  data_set.interactive_solution(next_node.item(), current_time, current_day, current_pos)
                                
                new_obs = torch.tensor(new_obs, dtype=torch.float32, device=device).unsqueeze(0)
                
                if (new_obs[...,2]* (mask - 1)).sum() == 0: #no violations left
                    break
                
                if use_current_mask:
                    mask = mask + ((new_obs[..., 2] - 1) * -1)
                    mask = torch.clamp_max(mask, 1)
                    
                    
                    if (mask - 1).sum() == 0: #no violations left
                        break
                
                ref = self.Embedding(new_obs)
                embed_enc_inputs = ref
                current_pos = next_node.item()
            else:
                if use_current_mask:
                    if (mask - 1).sum() == 0: #no violations left
                        break

            
            dec_input = torch.gather(input = embed_enc_inputs, dim = 1, index = next_node.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, embed))

            current_pos = next_node.item()
            
            
        pi = torch.stack(pi_list, dim = 1)
        ll = self.get_log_likelihood(torch.stack(log_ps, 1), pi)
        return pi, ll 
    
    def glimpse(self, query, ref, mask, inf = 1e8):
        """	-ref about torch.bmm, torch.matmul and so on
            https://qiita.com/tand826/items/9e1b6a4de785097fe6a5
            https://qiita.com/shinochin/items/aa420e50d847453cc296
            
                Args: 
            query: the hidden state of the decoder at the current
            (batch, 128)
            ref: the set of hidden states from the encoder. 
            (batch, city_t, 128)
            mask: model only points at cities that have yet to be visited, so prevent them from being reselected
            (batch, city_t)
        """
        u1 = self.W_q(query).unsqueeze(-1).repeat(1,1,ref.size(1))# u1: (batch, 128, city_t)
        u2 = self.W_ref(ref.permute(0,2,1))# u2: (batch, 128, city_t)
        V = self.Vec.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
        # V: (batch, 1, 128) * u1+u2: (batch, 128, city_t) => u: (batch, 1, city_t) => (batch, city_t)
        u = u - inf * mask
        a = F.softmax(u / self.softmax_T, dim = 1)
        d = torch.bmm(u2, a.unsqueeze(2)).squeeze(2)
        # u2: (batch, 128, city_t) * a: (batch, city_t, 1) => d: (batch, 128)
        return d

    def pointer(self, query, ref, mask, inf = 1e8):
        """	Args: 
            query: the hidden state of the decoder at the current
            (batch, 128)
            ref: the set of hidden states from the encoder. 
            (batch, city_t, 128)
            mask: model only points at cities that have yet to be visited, so prevent them from being reselected
            (batch, city_t)
        """
        u1 = self.W_q2(query).unsqueeze(-1).repeat(1,1,ref.size(1))# u1: (batch, 128, city_t)
        u2 = self.W_ref2(ref.permute(0,2,1))# u2: (batch, 128, city_t)
        V = self.Vec2.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = torch.bmm(V, self.clip_logits * torch.tanh(u1 + u2)).squeeze(1)
        # V: (batch, 1, 128) * u1+u2: (batch, 128, city_t) => u: (batch, 1, city_t) => (batch, city_t)
        u = u - inf * mask
        return u

    def get_log_likelihood(self, _log_p, pi):
        """	args:
            _log_p: (batch, city_t, city_t)
            pi: (batch, city_t), predicted tour
            return: (batch)
        """
        log_p = torch.gather(input = _log_p, dim = 2, index = pi[:,:,None])
        return torch.sum(log_p.squeeze(-1), 1)



class PtrNet2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.Embedding = nn.Linear(3, cfg.embed, bias = False)
        self.Encoder = nn.LSTM(input_size = cfg.embed, hidden_size = cfg.hidden, batch_first = True)
        self.Decoder = nn.LSTM(input_size = cfg.embed, hidden_size = cfg.hidden, batch_first = True)
        if torch.cuda.is_available():
            self.Vec = nn.Parameter(torch.cuda.FloatTensor(cfg.embed))
        else:
            self.Vec = nn.Parameter(torch.FloatTensor(cfg.embed))
        self.W_q = nn.Linear(cfg.hidden, cfg.hidden, bias = True)
        self.W_ref = nn.Conv1d(cfg.hidden, cfg.hidden, 1, 1)
        # self.dec_input = nn.Parameter(torch.FloatTensor(cfg.embed))
        self.final2FC = nn.Sequential(
                    nn.Linear(cfg.hidden, cfg.hidden, bias = False),
                    nn.ReLU(inplace = False),
                    nn.Linear(cfg.hidden, 1, bias = False))
        self._initialize_weights(cfg.init_min, cfg.init_max)
        self.n_glimpse = cfg.n_glimpse
        self.n_process = cfg.n_process
    
    def _initialize_weights(self, init_min = -0.08, init_max = 0.08):
        for param in self.parameters():
            nn.init.uniform_(param.data, init_min, init_max)
            
    def forward(self, current_time, current_day, current_pos, resource_observations, device, encode_future=False):
        '''	x: (batch, city_t, 2)
            enc_h: (batch, city_t, embed)
            query(Decoder input): (batch, 1, embed)
            h: (1, batch, embed)
            return: pred_l: (batch)
        '''
        
        
        
        x = resource_observations
        batch, city_t, xy = x.size()
        embed_enc_inputs = self.Embedding(x)
        embed = embed_enc_inputs.size(2)
        enc_h, (h, c) = self.Encoder(embed_enc_inputs, None)
        ref = enc_h
        ref = embed_enc_inputs
        # ~ query = h.permute(1,0,2).to(device)# query = self.dec_input.unsqueeze(0).repeat(batch,1).unsqueeze(1).to(device)
        query = h[-1]
        
        
        # ~ process_h, process_c = [torch.zeros((1, batch, embed), device = device) for _ in range(2)]
        for i in range(self.n_process):
            # ~ _, (process_h, process_c) = self.Decoder(query, (process_h, process_c))
            # ~ _, (h, c) = self.Decoder(query, (h, c))
            # ~ query = query.squeeze(1)
            for i in range(self.n_glimpse):
                query = self.glimpse(query, ref)
                # ~ query = query.unsqueeze(1)
        '''	
        - page 5/15 in paper
        critic model architecture detail is out there, "Critic’s architecture for TSP"
        - page 14/15 in paper
        glimpsing more than once with the same parameters 
        made the model less likely to learn and barely improved the results 
        
        query(batch,hidden)*FC(hidden,hidden)*FC(hidden,1) -> pred_l(batch,1) ->pred_l(batch)
        '''
        pred_l = self.final2FC(query).squeeze(-1).squeeze(-1)
        return pred_l 
    
    def glimpse(self, query, ref, infinity = 1e8):
        """	Args: 
            query: the hidden state of the decoder at the current
            (batch, 128)
            ref: the set of hidden states from the encoder. 
            (batch, city_t, 128)
        """
        u1 = self.W_q(query).unsqueeze(-1).repeat(1,1,ref.size(1))# u1: (batch, 128, city_t)
        u2 = self.W_ref(ref.permute(0,2,1))# u2: (batch, 128, city_t)
        V = self.Vec.unsqueeze(0).unsqueeze(0).repeat(ref.size(0), 1, 1)
        u = torch.bmm(V, torch.tanh(u1 + u2)).squeeze(1)
        # V: (batch, 1, 128) * u1+u2: (batch, 128, city_t) => u: (batch, 1, city_t) => (batch, city_t)
        a = F.softmax(u, dim = 1)
        d = torch.bmm(u2, a.unsqueeze(2)).squeeze(2)
        # u2: (batch, 128, city_t) * a: (batch, city_t, 1) => d: (batch, 128)
        return d
def my_collate(x):
    return x


class PtrAgent:
    def __init__(self, config, update_writer=None) -> None:
        _, graph, _ = load_data_for_env(config, for_cpp=True)
        self.distance_matrix_resources = get_distance_matrix_resources(graph)
        self.distance_matrix = get_distance_matrix(graph)
        with np.load(to_absolute_path(f"../data/{config.year}/{config.area_name}_time_series.npz")) as data:
            self.per_day_events = {int(day): x for day, x in data.items()}


        self.batch_size = config.batch_size
        self.use_whole_path = config.use_whole_path
        self.encode_future = config.encode_future
        self.use_current_mask = config.use_current_mask

        self.train_data_set = PtrDataSet(graph=graph, config=config, distance_matrix_resources=self.distance_matrix_resources,
                                          per_day_events=self.per_day_events,
                                          distance_matrix=self.distance_matrix,
                                        data_split=DataSplit.TRAINING)
        self.train_loader = DataLoader(self.train_data_set, batch_size=None,
                                       batch_sampler=None, prefetch_factor=256, num_workers=6, 
                                       collate_fn=my_collate)

        self.val_data_set = PtrDataSet(graph=graph, 
                                       config=config,
                                       distance_matrix_resources=self.distance_matrix_resources,
                                          per_day_events=self.per_day_events,
                                          distance_matrix=self.distance_matrix,
                                        data_split=DataSplit.VALIDATION)
        self.val_loader = DataLoader(self.val_data_set, batch_size=None,
                                     batch_sampler=None,
                                     prefetch_factor=100, num_workers=2, collate_fn=my_collate)

        self.validation_env = _initialize_environment(
            DataSplit.VALIDATION, None, None, None, config)
        self.test_env = _initialize_environment(
            DataSplit.TEST, None, None, None, config)


        self.device = torch.device("cuda")
        self.ptr_net = PtrNet1(config.model).to(self.device)
        self.optim = torch.optim.Adam(self.ptr_net.parameters(), lr=config.lr)

        self.critic = PtrNet2(config.model).to(self.device)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)
        
        
        self.critic_loss = nn.MSELoss()
        
        self.validation_iterations = config.validation_iterations
        self.iterations_per_epoch = config.iterations_per_epoch
        self.number_of_epochs = config.number_of_epochs


        save_model_dir = pathlib.Path(config.save_model_dir).expanduser()
        save_model_dir.mkdir(parents=True, exist_ok=True)
        self.save_model_path = save_model_dir.absolute()

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
        
        self.evaluator = NoTrainEvaluator(self.writer)

        resources = pd.read_csv(to_absolute_path(
            f"../data/{config.year}/{config.area_name}_resources.csv"))
        resources["x"] = (resources["x"] - resources["x"].min()) / \
            (resources["x"].max() - resources["x"].min())
        resources["y"] = (resources["y"] - resources["y"].min()) / \
            (resources["y"].max() - resources["y"].min())

        self.resource_observation = resources[["x", "y"]].to_numpy()

        self.resource_id_to_edge_id_mapping = self.validation_env.get_resource_id_to_edge_id_mapping()
        self.edge_id_to_action_mapping = self.validation_env.get_edge_id_to_action_mapping()

        self.action_to_r_ids_mapping = {}
        
        for edge_id, action in self.edge_id_to_action_mapping.items():
            if action not in self.action_to_r_ids_mapping:
                self.action_to_r_ids_mapping[action]  = []
            
            self.action_to_r_ids_mapping[action].extend([r_id for r_id, edge_id_2 in enumerate(self.resource_id_to_edge_id_mapping) if edge_id_2 == edge_id])

        self.current_path = None
        
        if config.load_agent_model:
            self.load(f"{config.path_to_model}/checkpoint_{config.load_step}.pth")
    
    def load(self, path):
        loaded_data = torch.load(path, map_location=self.device)
        self.ptr_net.load_state_dict(loaded_data["model"])
        if "critic" in loaded_data:
            self.critic.load_state_dict(loaded_data["critic"])
    
    def save_model(self, epoch):
        ckpt_path = os.path.join(
            self.save_model_path, f"checkpoint_{epoch}.pth")
        torch.save({"model": self.ptr_net.state_dict(), "critic" : self.critic.state_dict()}, ckpt_path)
        return ckpt_path

    def run(self):
        self.train()
            
        self.current_path = None
        self.evaluator.evaluate(
            epoch=0, global_step=0, env=self.validation_env, agent=self, mode="validation")
        self.current_path = None
        self.evaluator.evaluate(epoch=0, global_step=0,
                                env=self.test_env, agent=self, mode="test")

    def act(self, observation):
        self.ptr_net.eval()
        
        
        violation_status = [r[1] == 1 for r in observation["violations"]]
        violations = [int(r[0]) for r in observation["violations"] if r[1] == 1]

        if sum(violation_status) == 0:
            self.current_path = None
            return int(observation["current_position"][0])  # current position

        
        if self.current_path is not None and self.use_whole_path:
            selected_action = None
            
            while len(self.current_path) > 0:
                action = self.current_path.pop(0)

                #no resource on edge is in violation (skip)
                if not any((r_id for r_id in self.action_to_r_ids_mapping[action] if r_id in violations)):
                    continue
                
                selected_action = action
                
            if len(self.current_path) == 0:
                self.current_path = None
            
            if selected_action is not None:
                return action

        
        if sum(violation_status) == 1:
            self.current_path = None
            return self.edge_id_to_action_mapping[self.resource_id_to_edge_id_mapping[violations[0]]]

        
        position = int(observation["current_position"][0]) #current position

        data = torch.tensor(
            np.hstack((self.resource_observation, np.array(violation_status)[:, None])), device=self.device, dtype=torch.float32).unsqueeze(0)

        start_node = self.resource_observation[self.distance_matrix[position].argmin()]
                      
        start_node = torch.tensor(start_node, device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                         
                                               
        with torch.no_grad():
            solution, _ = self.ptr_net.forward(None, 
                                               None,
                                               position,
                                               data, 
                                               self.train_data_set, 
                                               self.device, 
                                               start_point=start_node,
                                               greedy=True,
                                               encode_future=False,
                                               use_current_mask=self.use_current_mask)

        solution = solution.cpu().numpy()

        path = solution[0]  # we only have one path


        edge_path = np.array(
            [self.resource_id_to_edge_id_mapping[r_id] for r_id in path])
        edge_path, ind = np.unique(edge_path, return_index=True)
        unique_edge_path = edge_path[np.argsort(ind)]

        action_path = [self.edge_id_to_action_mapping[edge_id]
                       for edge_id in unique_edge_path]

        self.current_path = action_path

        action = self.current_path.pop(0)

        if len(self.current_path) == 0:
            self.current_path = None

        return action

    def train(self):
        train_iterator = iter(self.train_loader)

        for epoch in tqdm(range(self.number_of_epochs)):
            iteration = 0

            epoch_losses = []

            for _ in tqdm(range(self.iterations_per_epoch)):
                self.ptr_net.train()
                
                current_batch_i = 0

                critic_preds = []
                costs_list = []
                
                losses = []
                while current_batch_i < self.batch_size:
                    problem = next(train_iterator)
                    current_time, current_day, current_pos, resource_observations, start_point = problem
                    
                    if resource_observations[..., 2].sum() < 2:
                        continue
                    
                    resource_observations = torch.tensor(resource_observations, dtype=torch.float32, device=self.device).unsqueeze(0)
                    start_point = torch.tensor(start_point, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)

                    path, log_p = self.ptr_net.forward(current_time, 
                                                       current_day,
                                                       current_pos,
                                                       resource_observations, 
                                                       self.train_data_set, 
                                                       self.device, 
                                                       start_point=start_point,
                                                       greedy=False,
                                                       encode_future=self.encode_future,
                                                       use_current_mask=self.use_current_mask)
                    
                    costs = []
                    
                    for i, sol in enumerate(path.cpu().numpy()):
                        cost = self.train_data_set.evaluate_solution(sol, current_time, current_day, current_pos)
                        costs.append(cost)
                    
                    costs = torch.tensor(costs, dtype=torch.float32, device=log_p.device)
                    critic_pred = self.critic.forward(current_time, current_day, current_pos, resource_observations, self.device).unsqueeze(0)
                    
                    advantage = costs - critic_pred.detach()

                    critic_preds.append(critic_pred)
                    costs_list.append(costs)
                    
                    
                    loss = (advantage * log_p)
                    losses.append(loss)
                    
                    current_batch_i += 1
                
                
                loss_critic = self.critic_loss(torch.stack(critic_preds), torch.stack(costs_list))
                self.optim_critic.zero_grad()
                loss_critic.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm = 1., norm_type = 2)
                self.optim_critic.step()

                self.optim.zero_grad()
                loss = torch.stack(losses).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ptr_net.parameters(), max_norm = 1., norm_type = 2)
                self.optim.step()
                epoch_losses.append(loss.item())


                iteration += 1
                

            print(f"epoch[{epoch}]: loss: {np.mean(epoch_losses)}")
            self.writer.log_metrics(
                {"loss": np.mean(epoch_losses)}, prefix="train", epoch=epoch)
            
            with torch.no_grad():
                self.ptr_net.eval()
                self.save_model(epoch=epoch)

                print("saved")
                print("evaluate")

                self.current_path = None
                self.evaluator.evaluate(
                    epoch=epoch, global_step=epoch, env=self.validation_env, agent=self, mode="validation")





# cfg_ptr_debug = DictConfig({
#   "embed": 128,
#   "hidden": 128,  
#   "init_min": -0.08,
#   "init_max": 0.08,
#   "clip_logits": 10,
#   "softmax_T": 1.0,
#   "n_glimpse": 1
# })


# net = PtrNet1(cfg_ptr_debug)

# inputs = torch.randn(8,20,3)	
# path, ll = net.forward(inputs)

# a = 3

