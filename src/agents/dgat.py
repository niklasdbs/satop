import os
import pathlib
from typing import Iterator
from hydra.utils import to_absolute_path
import numpy as np
import pandas as pd
import torch
from torch import nn
import math
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import IterableDataset

from tqdm import tqdm
from agents.no_train_agent import NoTrainEvaluator
from agents.tianshou_agents import _initialize_environment, load_data_for_env

from datasets.datasets import DataSplit
from envs.utils import get_distance_matrix, get_distance_matrix_resources, load_graph_from_file
from scipy import stats

from utils.logging.logger import JSONOutput, WANDBLogger, Logger


# also see https://github.com/wouterkool/attention-learn-to-route/blob/master/nets/graph_encoder.py

class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        # See Attention is all you need
        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """

        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query,
                             graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous(
            ).view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # Alternative:
        # headst = heads.transpose(0, 1)  # swap the dimensions for batch and heads to align it for the matmul
        # # proj_h = torch.einsum('bhni,hij->bhnj', headst, self.W_out)
        # projected_heads = torch.matmul(headst, self.W_out)
        # out = torch.sum(projected_heads, dim=1)  # sum across heads

        # Or:
        # out = torch.einsum('hbni,hij->bnj', heads, self.W_out)

        return out


class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                )
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        )


class GraphAttentionEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(GraphAttentionEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(
            node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(
                n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None):

        assert mask is None, "mask not yet supported!"

        # Batch multiply to get initial embeddings of nodes
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()
                                                         [:2], -1) if self.init_embed is not None else x

        h = self.layers(h)

        return (
            h,  # (batch_size, graph_size, embed_dim)
            # average to get embedding of graph, (batch_size, embed_dim)
            h.mean(dim=1),
        )


class Decoder(nn.Module):
    def __init__(self, embed_dim) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.tanh_clipping = 10.0  # TODO do not hardcode
        self.mask_logits = True  # TODO do not hardcode

        # placeholders for first and last
        self.v_l = nn.Parameter(
            torch.FloatTensor(size=[1, 1, self.embed_dim]).uniform_())
        self.v_f = nn.Parameter(
            torch.FloatTensor(size=[1, 1, self.embed_dim]).uniform_())

        self.attention = MultiHeadAttention(
            n_heads=8, input_dim=self.embed_dim, embed_dim=self.embed_dim) 
        self.proj_k = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.proj_context = nn.Linear(
            3*self.embed_dim, self.embed_dim, bias=False)

    def forward(self, x, greedy=False, start_node=None):
        bsz, n_node, hidden_dim = x.size()

        graph_embedding = x.mean(dim=-2, keepdim=True)

        if start_node is None:
            first = self.v_f.repeat(bsz, 1, 1)
        else:
            first = start_node

        last = self.v_l.repeat(bsz, 1, 1)

        k = self.proj_k(x)

        mask = torch.zeros([bsz, n_node], device=x.device).bool()

        visited_idx = []
        log_prob = 0

        for i in range(n_node):
            h_c = torch.cat([graph_embedding, last, first], dim=-1)
            h_c = self.proj_context(h_c)

            q = self.attention(h_c, x, mask=mask)
            logits = torch.matmul(q, k.transpose(-2, -1)
                                  ).squeeze(-2) / math.sqrt(q.size(-1))

            if self.tanh_clipping > 0:
                logits = torch.tanh(logits) * self.tanh_clipping
            if self.mask_logits:
                logits[mask] = -math.inf

            if greedy:
                visit_idx = logits.max(-1)[1]
            else:
                m = torch.distributions.Categorical(logits=logits)
                visit_idx = m.sample()
                log_prob += m.log_prob(visit_idx)

            visit_idx = visit_idx.unsqueeze(-1)
            visited_idx += [visit_idx]
            mask = mask.scatter(1, visit_idx, True)

            visit_idx = visit_idx.unsqueeze(-1).repeat(1, 1, hidden_dim)
            last = torch.gather(x, 1, visit_idx)
            if len(visited_idx) == 1 and start_node is None:
                first = last

        visited_idx = torch.cat(visited_idx, -1)

        return visited_idx, log_prob


class DGAT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.encoder = GraphAttentionEncoder(
            n_heads=8, 
            embed_dim=config.model.embed_dim, #128
            n_layers=3, 
            node_dim=2, 
            normalization=config.model.normalization,  #batch
            feed_forward_hidden=512)  # TODO do not hardcode

        self.decoder = Decoder(
            embed_dim=config.model.embed_dim #128
            ) 

    def forward(self, x, greedy=False, start_node=None):
        node_embedding, _ = self.encoder(x)

        if start_node is not None:
            start_node = self.encoder.init_embed(start_node)

        path, log_prob = self.decoder(
            node_embedding, greedy=greedy, start_node=start_node)
        return path, log_prob


def _is_in_data_split(day_of_year, data_split: DataSplit):
    """Returns true, if the selected date is included in the selected dataplit."""
    if data_split is DataSplit.TRAINING:
        return day_of_year % 13 > 1
    elif data_split is DataSplit.VALIDATION:
        return day_of_year % 13 == 1
    elif data_split is DataSplit.TEST:
        return day_of_year % 13 == 0
    raise AssertionError


class DGATDataSet(IterableDataset):
    def __init__(self, 
                 graph, 
                 config, 
                 distance_matrix_resources, 
                 per_day_events, 
                 edge_to_r_dist_matrix,
                 use_all_resources,
                 edge_id_to_r_ids_mapping,
                 resource_id_to_edge_id_mapping,
                 edge_id_to_action_mapping,
                 data_split=DataSplit.TRAINING) -> None:

        self.area_name = config.area_name
        self.use_edge_path = config.use_edge_path
        self.skip_resources_not_in_violation = config.skip_resources_not_in_violation
        
        self.distance_matrix_resources = distance_matrix_resources
        self.resource_id_to_edge_id_mapping = resource_id_to_edge_id_mapping
        self.edge_id_to_action_mapping = edge_id_to_action_mapping
        self.edge_to_r_dist_matrix = edge_to_r_dist_matrix
        self.edge_id_to_r_ids_mapping  = edge_id_to_r_ids_mapping
        self.per_day_events = per_day_events

        self.days = [day for day in range(
            1, 366) if _is_in_data_split(day, data_split)]
        self.start = config.start_hour * 60 * 60
        self.end = config.end_hour * 60 * 60

        self.speed_in_ms = config.speed_in_kmh / 3.6

        resources = pd.read_csv(to_absolute_path(
            f"../data/{config.year}/{self.area_name}_resources.csv"))
        resources["x"] = (resources["x"] - resources["x"].min()) / \
            (resources["x"].max() - resources["x"].min())
        resources["y"] = (resources["y"] - resources["y"].min()) / \
            (resources["y"].max() - resources["y"].min())

        self.resource_observation = resources[["x", "y"]].to_numpy()
        
        self.use_all_resources = use_all_resources

    def __iter__(self) -> Iterator:
        while True:
            if self.use_all_resources:
                yield self.get_problem_instance_all_resources()
            else:
                yield self.get_problem_instance()

    def get_problem_instance(self):
        #current_time = np.random.randint(low=self.start, high=self.end, size=1)
        #current_day = np.random.choice(self.days, size=1, replace=True)
        current_day = np.random.choice(self.days)
        current_time = np.random.randint(low=self.start, high=self.end)

        events_per_resource = self.per_day_events[current_day]

        violation_resource_id_to_real_resource_id_mapping = {}
        resources_in_violation = []
        real_r_ids_in_violation = []

        start_point = np.random.randint(low=0, high=len(self.edge_to_r_dist_matrix))
        
        i = 0
        for r_id, events in enumerate(events_per_resource):
            relevant_events = events[(events[..., 0] < current_time)]

            if len(relevant_events) == 0:
                in_violation = False
            else:
                last_event = relevant_events[-1]

                in_violation = last_event[3] == 1

            if in_violation:
                violation_resource_id_to_real_resource_id_mapping[i] = r_id
                resources_in_violation.append(i)
                real_r_ids_in_violation.append(r_id)

                i += 1

        # new_distance_matrix = np.zeros((len(resources_in_violation), len(resources_in_violation)))

        # for viol_r_id, real_r_id in violation_resource_id_to_real_resource_id_mapping.items():
        #     for viol_id_target, real_r_id_target in violation_resource_id_to_real_resource_id_mapping.items():
        #         new_distance_matrix[viol_r_id, viol_id_target] = self.distance_matrix_resources[real_r_id, real_r_id_target]

        return current_time, current_day, violation_resource_id_to_real_resource_id_mapping, self.resource_observation[real_r_ids_in_violation], start_point, self.resource_observation[self.edge_to_r_dist_matrix[start_point].argmin()]

    def get_problem_instance_all_resources(self):
        current_day = np.random.choice(self.days)
        current_time = np.random.randint(low=self.start, high=self.end)

        events_per_resource = self.per_day_events[current_day]

        in_violation_array = np.zeros(self.resource_observation.shape[0])

        start_point = np.random.randint(low=0, high=len(self.edge_to_r_dist_matrix))
        
        i = 0
        for r_id, events in enumerate(events_per_resource):
            relevant_events = events[(events[..., 0] < current_time)]

            if len(relevant_events) == 0:
                in_violation = False
            else:
                last_event = relevant_events[-1]

                in_violation = last_event[3] == 1

            if in_violation:
                in_violation_array[r_id] = 1

                i += 1
        
        
        return current_time, current_day, in_violation_array, self.resource_observation, start_point, self.resource_observation[self.edge_to_r_dist_matrix[start_point].argmin()]

    
    def evaluate_solution(self, solution_path, current_time, current_day, violation_to_real_resource_mapping, start_point = None):
        current_time = current_time
        prev_r_id = None

        fined_resources = 0


        start = True
        
        
        if self.use_edge_path:
            prev_edge_id = None
            edge_path = np.array([self.resource_id_to_edge_id_mapping[violation_to_real_resource_mapping[r_id]] for r_id in solution_path])
            edge_path, ind = np.unique(edge_path, return_index=True)
            unique_edge_path = edge_path[np.argsort(ind)]
            
            for edge_id in unique_edge_path:
                
                if self.skip_resources_not_in_violation:
                    any_in_violation = False
                    
                    for r_id in self.edge_id_to_r_ids_mapping[edge_id]:
                        events = self.per_day_events[current_day][r_id]
                        relevant_events = events[(events[..., 0] < current_time)]

                        if len(relevant_events) == 0:
                            continue #not in violation skip
                        else:
                            last_event = relevant_events[-1]
                            in_violation = last_event[3] == 1

                            if not in_violation:
                                continue
                            else:
                                any_in_violation = True
                                break
                            
                    if not any_in_violation:
                        continue #skip no violation on edge
                    
                if start:
                    if start_point is None:
                        travel_time = 42 / self.speed_in_ms
                    else:
                        travel_time = self.edge_to_r_dist_matrix[start_point, self.edge_id_to_r_ids_mapping[edge_id][0]] / self.speed_in_ms
                    start = False
                else:
                    travel_time = self.edge_to_r_dist_matrix[self.edge_id_to_action_mapping[prev_edge_id], self.edge_id_to_r_ids_mapping[edge_id][0]] / self.speed_in_ms


                current_time += int(travel_time)

                if current_time > self.end:
                    break
                
                for r_id in self.edge_id_to_r_ids_mapping[edge_id]:
                    events = self.per_day_events[current_day][r_id]

                    relevant_events = events[(events[..., 0] < current_time)]

                    if len(relevant_events) == 0:
                        pass  # not in violation nothing to do here
                    else:
                        last_event = relevant_events[-1]
                        in_violation = last_event[3] == 1

                        if in_violation:
                            fined_resources += 1

                prev_edge_id = edge_id
        else:
            for viol_r_id in solution_path:
                
                if self.skip_resources_not_in_violation:
                    events = self.per_day_events[current_day][violation_to_real_resource_mapping[viol_r_id]]
                    relevant_events = events[(events[..., 0] < current_time)]

                    if len(relevant_events) == 0:
                        continue #not in violation skip
                    else:
                        last_event = relevant_events[-1]
                        in_violation = last_event[3] == 1

                        if not in_violation:
                            continue

                
                if start:
                    if start_point is None:
                        travel_time = 42 / self.speed_in_ms
                    else:
                        travel_time = self.edge_to_r_dist_matrix[start_point][violation_to_real_resource_mapping[viol_r_id]] / self.speed_in_ms
                    start = False
                else:
                    travel_time = self.distance_matrix_resources[violation_to_real_resource_mapping[
                        prev_r_id], violation_to_real_resource_mapping[viol_r_id]] / self.speed_in_ms

                current_time += int(travel_time)

                if current_time > self.end:
                    break

                events = self.per_day_events[current_day][violation_to_real_resource_mapping[viol_r_id]]

                relevant_events = events[(events[..., 0] < current_time)]

                if len(relevant_events) == 0:
                    pass  # not in violation nothing to do here
                else:
                    last_event = relevant_events[-1]
                    in_violation = last_event[3] == 1

                    if in_violation:
                        fined_resources += 1

                prev_r_id = viol_r_id

        return 1.0 - fined_resources/len(solution_path)
    
    def evaluate_solution_all_resources(self, solution_path, current_time, current_day, in_violation_array, start_point = None):
        current_time = current_time
        prev_r_id = None

        fined_resources = 0

        number_of_violations = in_violation_array.sum()
        start = True

        
        for viol_r_id in solution_path:
            events = self.per_day_events[current_day][viol_r_id]
            relevant_events = events[(events[..., 0] < current_time)]

            if len(relevant_events) == 0:
                continue #not in violation skip
            else:
                last_event = relevant_events[-1]
                in_violation = last_event[3] == 1

                if not in_violation:
                    continue

            
            
            if start:
                if start_point is None:
                    travel_time = 42 / self.speed_in_ms
                else:
                    travel_time = self.edge_to_r_dist_matrix[start_point, viol_r_id] / self.speed_in_ms
                start = False
            else:
                travel_time = self.distance_matrix_resources[prev_r_id, viol_r_id] / self.speed_in_ms

            current_time += int(travel_time)

            if current_time > self.end:
                break


            relevant_events = events[(events[..., 0] < current_time)]

            if len(relevant_events) == 0:
                pass  # not in violation nothing to do here
            else:
                last_event = relevant_events[-1]
                in_violation = last_event[3] == 1

                if in_violation:
                    fined_resources += 1

            prev_r_id = viol_r_id

        return 1.0 - fined_resources/number_of_violations


def my_collate(x):
    return x



def collate_all_resources(x):
    current_times = []
    current_days = []
    in_violation_arrays = []
    resource_observations = []
    start_points = []
    start_observs = []
    
    for problem in x:
        current_time, current_day, in_violation_array, resource_observation, start_point, start_obs = problem
        current_times.append(current_time)
        current_days.append(current_day)
        in_violation_arrays.append(in_violation_array)
        resource_observations.append(resource_observation)
        start_points.append(start_point)
        start_observs.append(start_obs)
    
    
    return current_times, current_days, in_violation_arrays, np.array(resource_observations), start_points, np.array(start_observs)


#data = torch.rand(size=(16, 64, 2))
#dgat.forward(data, greedy=True)


class DGATAgent:
    def __init__(self, config) -> None:
        _, graph, _ = load_data_for_env(config, for_cpp=True)
        self.distance_matrix_resources = get_distance_matrix_resources(graph)
        self.distance_matrix = get_distance_matrix(graph)
        with np.load(to_absolute_path(f"../data/{config.year}/{config.area_name}_time_series.npz")) as data:
            self.per_day_events = {int(day): x for day, x in data.items()}

        self.always_update = config.always_update
        self.use_all_resources = config.use_all_resources
        self.batch_size = config.batch_size
        self.validation_env = _initialize_environment(
            DataSplit.VALIDATION, None, None, None, config)
        self.test_env = _initialize_environment(
            DataSplit.TEST, None, None, None, config)
        
        self.resource_id_to_edge_id_mapping = self.validation_env.get_resource_id_to_edge_id_mapping()
        self.edge_id_to_action_mapping = self.validation_env.get_edge_id_to_action_mapping()

        self.action_to_r_ids_mapping = {}
        
        for edge_id, action in self.edge_id_to_action_mapping.items():
            if action not in self.action_to_r_ids_mapping:
                self.action_to_r_ids_mapping[action]  = []
            
            self.action_to_r_ids_mapping[action].extend([r_id for r_id, edge_id_2 in enumerate(self.resource_id_to_edge_id_mapping) if edge_id_2 == edge_id])

        
        self.edge_id_to_r_ids_mapping = {}

        for r_id, edge_id in enumerate(self.resource_id_to_edge_id_mapping):
            if edge_id not in self.edge_id_to_r_ids_mapping:
                self.edge_id_to_r_ids_mapping[edge_id]  = []
            
            self.edge_id_to_r_ids_mapping[edge_id].append(r_id)


        self.train_data_set = DGATDataSet(graph=graph, 
                                          config=config, 
                                          distance_matrix_resources=self.distance_matrix_resources,
                                          per_day_events=self.per_day_events,
                                          edge_to_r_dist_matrix=self.distance_matrix,
                                          use_all_resources=self.use_all_resources, 
                                          edge_id_to_r_ids_mapping=self.edge_id_to_r_ids_mapping, 
                                          resource_id_to_edge_id_mapping = self.resource_id_to_edge_id_mapping,
                                          edge_id_to_action_mapping=self.edge_id_to_action_mapping,
                                          data_split=DataSplit.TRAINING)
        self.train_loader = DataLoader(self.train_data_set, batch_size=self.batch_size if self.use_all_resources else None,
                                       batch_sampler=None, prefetch_factor=256, num_workers=6,
                                       collate_fn=collate_all_resources if self.use_all_resources else my_collate)

        self.val_data_set = DGATDataSet(graph=graph, 
                                        config=config,
                                        distance_matrix_resources=self.distance_matrix_resources,
                                        per_day_events=self.per_day_events,
                                        edge_to_r_dist_matrix=self.distance_matrix,
                                        use_all_resources=self.use_all_resources, 
                                        edge_id_to_r_ids_mapping=self.edge_id_to_r_ids_mapping, 
                                        resource_id_to_edge_id_mapping = self.resource_id_to_edge_id_mapping,
                                        edge_id_to_action_mapping=self.edge_id_to_action_mapping,
                                        data_split=DataSplit.VALIDATION)
        self.val_loader = DataLoader(self.val_data_set, batch_size=self.batch_size if self.use_all_resources else None,
                                     batch_sampler=None,
                                     prefetch_factor=100, num_workers=2, collate_fn=collate_all_resources if self.use_all_resources else my_collate)


        self.use_whole_path = config.use_whole_path

        self.device = torch.device("cuda")
        self.dgat = DGAT(config)
        self.dgat = self.dgat.to(self.device)
        self.optim = torch.optim.Adam(self.dgat.parameters(), lr=config.lr)

        self.validation_iterations = config.validation_iterations
        self.iterations_per_epoch = config.iterations_per_epoch
        self.number_of_epochs = config.number_of_epochs
        self.max_norm = config.max_norm

        self.target_net = DGAT(config)
        self.target_net = self.target_net.to(self.device)
        self.target_net.load_state_dict(self.dgat.state_dict())
        self.target_net.eval()

        save_model_dir = pathlib.Path(config.save_model_dir).expanduser()
        save_model_dir.mkdir(parents=True, exist_ok=True)
        self.save_model_path = save_model_dir.absolute()

        output_loggers = [
            #TensorboardOutput(log_dir=".", comment=f""),
            JSONOutput(log_dir=os.getcwd())
        ]

        if not config.experiment_name == "debug":
            output_loggers.append(WANDBLogger(config=config))

        self.writer = Logger(output_loggers)
        self.evaluator = NoTrainEvaluator(self.writer)

        resources = pd.read_csv(to_absolute_path(
            f"../data/{config.year}/{config.area_name}_resources.csv"))
        resources["x"] = (resources["x"] - resources["x"].min()) / \
            (resources["x"].max() - resources["x"].min())
        resources["y"] = (resources["y"] - resources["y"].min()) / \
            (resources["y"].max() - resources["y"].min())

        self.resource_observation = resources[["x", "y"]].to_numpy()


        
        self.current_path = None

    def save_model(self, epoch):
        ckpt_path = os.path.join(
            self.save_model_path, f"checkpoint_{epoch}.pth")
        torch.save({"model": self.target_net.state_dict()}, ckpt_path)
        return ckpt_path

    def run(self):
        self.train()

        self.target_net.eval()
        self.current_path = None
        self.evaluator.evaluate(
            epoch=0, global_step=0, env=self.validation_env, agent=self, mode="validation")
        self.current_path = None
        self.evaluator.evaluate(epoch=0, global_step=0,
                                env=self.test_env, agent=self, mode="test")

    def act(self, observation):
        self.target_net.eval()
        violation_resources = [
            r for r in observation["violations"] if r[1] == 1]

        if len(violation_resources) == 0:
            self.current_path = None
            return int(observation["current_position"][0])  # current position

        
        # contains a list of the resource ids in violation
        violations = [int(r[0]) for r in violation_resources]
        
        if self.current_path is not None and self.use_whole_path:
            selected_action = None
            
            while len(self.current_path) > 0:
                action = self.current_path.pop(0)

                #no resource on edge is in violation (skip)
                if not any((r_id for r_id in self.action_to_r_ids_mapping[action] if r_id in violations)):
                    continue
                else:
                    selected_action = action
                    break
                    
            if len(self.current_path) == 0:
                self.current_path = None
            
            if selected_action is not None:
                return action

        
        if len(violation_resources) == 1:
            self.current_path = None
            return self.edge_id_to_action_mapping[self.resource_id_to_edge_id_mapping[violations[0]]]

        
        position = int(observation["current_position"][0]) #current position

        data = torch.tensor(
            self.resource_observation[violations], device=self.device, dtype=torch.float32).unsqueeze(0)

        start_node = self.resource_observation[self.distance_matrix[position].argmin()]
                      
        start_node = torch.tensor(start_node, device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                         
                                               
        with torch.no_grad():
            solution, _ = self.target_net.forward(data, greedy=True, start_node=start_node)

        solution = solution.cpu().numpy()

        violations = np.array(violations)

        path = violations[solution]
        path = path[0]  # we only have one path


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
                self.dgat.train()
                
                
                if self.use_all_resources:
                    problem = next(train_iterator)

                    current_time, current_day, in_violation_array, resource_observations, start_point, start_obs = problem
                    data = torch.tensor(
                        resource_observations, device=self.device, dtype=torch.float32)
                    
                    start_obs = torch.tensor(
                        start_obs, device=self.device, dtype=torch.float32).unsqueeze(1)
                    
                    solutions, log_p = self.dgat.forward(data, greedy=False, start_node=start_obs)

                    with torch.no_grad():
                        base_line_solutions, baseline_logp = self.target_net.forward(
                            data, greedy=True, start_node=start_obs)
                    base_line_solutions_np = base_line_solutions.cpu().numpy()
                    
                    advantages = []
                    
                    for i, sol in enumerate(solutions.cpu().numpy()):
                        costs = self.train_data_set.evaluate_solution_all_resources(
                            sol, 
                            current_time=current_time[i], 
                            current_day=current_day[i], 
                            in_violation_array=in_violation_array[i],
                            start_point=start_point[i])
                        costs_bl = self.train_data_set.evaluate_solution_all_resources(
                            base_line_solutions_np[i], 
                            current_time=current_time[i], 
                            current_day=current_day[i], 
                            in_violation_array=in_violation_array[i],
                            start_point=start_point[i])

                        advantage = costs - costs_bl
                        advantages.append(advantage)
                    
                    advantages = torch.tensor(advantages, device=self.device, dtype=torch.float32)
                    loss = (advantages * log_p).mean()

                else:
                    current_batch_i = 0

                    while current_batch_i < self.batch_size:
                        problem = next(train_iterator)
                                        
                        current_time, current_day, violation_resource_id_to_real_resource_id_mapping, resource_observations, start_point, start_obs = problem
                        if len(violation_resource_id_to_real_resource_id_mapping) < 2:
                            continue

                        data = torch.tensor(
                            resource_observations, device=self.device, dtype=torch.float32).unsqueeze(0)
                        
                        start_obs = torch.tensor(
                            start_obs, device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                        solutions, log_p = self.dgat.forward(data, greedy=False, start_node=start_obs)

                        with torch.no_grad():
                            base_line_solutions, baseline_logp = self.target_net.forward(
                                data, greedy=True, start_node=start_obs)
                        base_line_solutions_np = base_line_solutions.cpu().numpy()
                        losses = []

                        for i, sol in enumerate(solutions.cpu().numpy()):
                            costs = self.train_data_set.evaluate_solution(
                                sol, 
                                current_time=current_time, 
                                current_day=current_day, 
                                violation_to_real_resource_mapping=violation_resource_id_to_real_resource_id_mapping,
                                start_point=start_point)
                            costs_bl = self.train_data_set.evaluate_solution(
                                base_line_solutions_np[i], 
                                current_time=current_time, 
                                current_day=current_day, 
                                violation_to_real_resource_mapping=violation_resource_id_to_real_resource_id_mapping,
                                start_point=start_point)

                            advantage = costs - costs_bl

                            loss = advantage * log_p[i]

                            losses.append(loss)
                            current_batch_i += 1
                            # if advantage < 1.0:
                            #    print(f"epoch {epoch}: {advantage}")

                    loss = torch.stack(losses).mean()
                    
                    
                self.optim.zero_grad()
                loss.backward()

                grad_norms = torch.nn.utils.clip_grad_norm_(
                    self.dgat.parameters(), self.max_norm)
                self.optim.step()
                epoch_losses.append(loss.item())

                iteration += 1

            print(f"epoch[{epoch}]: loss: {np.mean(epoch_losses)}")
            self.writer.log_metrics(
                {"loss": np.mean(epoch_losses)}, prefix="train", epoch=epoch)
            with torch.no_grad():
                self.target_net.eval()
                self.dgat.eval()

                baseline_cost = []
                policy_cost = []

                iteration_val = 0
                for _, problem in enumerate(self.val_loader):
                    current_time, current_day, violation_resource_id_to_real_resource_id_mapping, resource_observations, start_point, start_obs = problem
                    if len(violation_resource_id_to_real_resource_id_mapping) < 2:
                        continue

                    data = torch.tensor(
                        resource_observations, device=self.device, dtype=torch.float32).unsqueeze(0)
                    
                    start_obs = torch.tensor(
                        start_obs, device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                    
                    solutions, _ = self.dgat.forward(data, greedy=True, start_node=start_obs)
                    base_line_solutions, _ = self.target_net.forward(
                        data, greedy=True, start_node=start_obs)

                    bl_sol_np = base_line_solutions.cpu().numpy()
                    for i, sol in enumerate(solutions.cpu().numpy()):
                        costs = self.val_data_set.evaluate_solution(
                            sol,
                            current_time=current_time,
                            current_day=current_day, 
                            violation_to_real_resource_mapping=violation_resource_id_to_real_resource_id_mapping,
                            start_point=start_point)
                        costs_bl = self.val_data_set.evaluate_solution(
                            bl_sol_np[i], 
                            current_time=current_time, 
                            current_day=current_day,
                            violation_to_real_resource_mapping=violation_resource_id_to_real_resource_id_mapping,
                            start_point=start_point)
                        baseline_cost.append(costs_bl)
                        policy_cost.append(costs)

                    iteration_val += 1

                    if iteration_val > self.validation_iterations:
                        break

                baseline_cost = np.array(baseline_cost)
                policy_cost = np.array(policy_cost)

                baseline_mean = baseline_cost.mean()
                candidate_mean = policy_cost.mean()

                print(f"candidate: {candidate_mean}  base: {baseline_mean}")
                
                if candidate_mean - baseline_mean < 0 or self.always_update:
                    _, p_value = stats.ttest_rel(policy_cost, baseline_cost)
                    p_value = p_value / 2  # one-sided

                    print(f"p_value: {p_value}")

                    if p_value < 0.05 or self.always_update:  # TODO do not hardcode
                        self.target_net.load_state_dict(self.dgat.state_dict())
                        print("improve")

                        self.save_model(epoch=epoch)

                        print("saved")
                        print("evaluate")

                        self.current_path = None
                        self.evaluator.evaluate(
                            epoch=epoch, global_step=epoch, env=self.validation_env, agent=self, mode="validation")
