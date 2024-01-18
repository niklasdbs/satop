import copy
import math
from typing import Any, Dict, Tuple, Union
import numpy as np
import torch
from torch.nn.modules.container import ModuleList
from torch.nn.parameter import Parameter

from utils.torch.models.mlp import MLP
from torch import nn
import torch.nn.functional as F

from utils.torch.models.skip_connection import SkipConnection




class EdgeConvBlock(nn.Module):
    
    def __init__(self, adjacency,in_features, out_features, bias=True, use_only_distance=False, norm_edge_conv_adj=False, edge_conv_adjacency_activation="ELU") -> None:
        super().__init__()
        
        self.use_only_distance = use_only_distance
        
        self.adjacency = adjacency 
        
        self.adjacency_limit = (self.adjacency[:,:,0]>0).float()
        
        
        if norm_edge_conv_adj:
            self.adjacency[...,0] = self.adjacency[...,0]/3000.0
            self.adjacency[...,1] = self.adjacency[...,1]/40.0

        
        if self.use_only_distance:
            self.adjacency = self.adjacency[:,:,0].unsqueeze(-1)


        self.edge_net = MLP(1 if self.use_only_distance else 2, 1, number_of_layers=2, hidden_size=256, activation_after_last_layer=nn.Tanh, activation=edge_conv_adjacency_activation)
        
        self.fc = nn.Linear(in_features, out_features, bias=False)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        for layer in self.edge_net.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        
        stdv = 1. / math.sqrt(self.fc.weight.size(1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    
    def forward(self, x):
        adjacency_h = self.edge_net(self.adjacency).squeeze(-1)

        adjacency_h = self.adjacency_limit * adjacency_h

        x_h = adjacency_h @ self.fc(x)

        if self.bias is not None:
            return x_h + self.bias
        else:
            return x_h


class EdgeConv(nn.Module):
    def __init__(self, edge_conv_layer, num_layers, hidden_dim, use_activation=False, use_norm=True) -> None:
        super().__init__()
        modules = []
        for i in range(num_layers):
            e_conv = copy.deepcopy(edge_conv_layer)
            activ = getattr(torch.nn, use_activation)() if isinstance(use_activation, str) else nn.Identity()
            
            modules.append(SkipConnection(nn.Sequential(e_conv, activ)))
            
            if use_norm:
                modules.append(nn.LayerNorm(hidden_dim))

        self.module = nn.Sequential(*modules)
            
    def forward(self, x):
        return self.module.forward(x)
        
        

class ResourceToEdgeConv(nn.Module):
    def __init__(self, adjacency, in_features, out_features, bias=True, use_activation=False):
        super().__init__()
        self.adjacency = adjacency 

        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        
        self.activation = getattr(torch.nn, use_activation)() if isinstance(use_activation, str) else nn.Identity()
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.fc.weight.size(1))
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
    def forward(self, resource_embedding):
        x = self.fc(resource_embedding)
        h = self.adjacency @ x #AxH
        if self.bias is not None:
            return self.activation(h + self.bias)
        else:
            return self.activation(h)
        
        
    
class SATOPModule(nn.Module):

    def __init__(self, edge_to_edge_matrix, adjacency_matrix_resources, distance_matrix, sparse_dist_matrix, observation_space, device, config, area_name):
        super().__init__()
                
        self.next_action_embedding = config.next_action_embedding #use an embedding of possible next actions
        self.dqn = True

        self.use_distance_to_action = config.use_distance_to_action
        self.action_targets_only = config.action_targets_only
        self.resource_hidden_dim = config.resource_hidden_dim #hidden dim of each resource
        self.edge_aggregated_resources_hidden_dim = config.edge_aggregated_resources_hidden_dim
        self.edge_info_hidden_dim = config.edge_info_hidden_dim #hidden dim of the combined edge info
        self.use_edge_conv = config.use_edge_conv
        
        self.normalize_route_distance_aggr = config.normalize_route_distance_aggr
        self.learnable_distance_route_norm = config.learnable_distance_route_norm
        
        if self.normalize_route_distance_aggr:
            self.learnable_dist_route_param = nn.Parameter(torch.tensor(-1.0,dtype=torch.float32), requires_grad=self.learnable_distance_route_norm)
        
        self.norm_sim_matrix = config.norm_sim_matrix
        
        self.device = device
        self.number_of_actions = observation_space["distance_to_action"].shape[0]
        self.resource_dim = observation_space["resource_observations"].shape[1]
        self.number_of_resources = observation_space["resource_observations"].shape[0]
        
        self.distance_future_aggregation = config.distance_future_aggregation
        if self.distance_future_aggregation:
            self.distance_matrix = torch.exp(-1.0 * distance_matrix / distance_matrix.max())
        else:
            self.distance_matrix = None
        
        self.sparse_dist_matrix = sparse_dist_matrix #AxAxR
        self.adjacency_matrix_resources = adjacency_matrix_resources #AxR
        self.edge_to_edge_matrix = edge_to_edge_matrix #AxA
        
        self.use_resource_id_embedding = config.use_resource_id_embedding #create an embedding per resource
        self.resource_id_embedding_dim = config.resource_id_embedding_dim
        self.use_per_edge_resource = config.use_per_edge_resource
        self.use_only_distance_in_edge_conv = config.use_only_distance_in_edge_conv
        
        self.edge_conv_without_route = config.edge_conv_without_route
        
        if self.edge_conv_without_route:
            self.action_net = MLP(self.edge_aggregated_resources_hidden_dim+self.edge_info_hidden_dim, self.edge_info_hidden_dim, number_of_layers=4, hidden_size=512, activation_after_last_layer=True)
        
        if self.use_resource_id_embedding:
            self.register_buffer("resource_ids", torch.arange(0, self.number_of_resources, 1))
            self.resource_id_embedding = nn.Embedding(self.number_of_resources, self.resource_id_embedding_dim)

        self.resource_embedding_net = MLP(self.resource_dim + (self.resource_id_embedding_dim if self.use_resource_id_embedding else 0), 
                                          self.resource_hidden_dim, 
                                          **config.resource_embedding_net)
        
        self.resource_to_edge_conv = ResourceToEdgeConv(self.distance_matrix if self.distance_future_aggregation else self.adjacency_matrix_resources, 
                                                        self.resource_hidden_dim, 
                                                        self.edge_aggregated_resources_hidden_dim, 
                                                        bias=True,
                                                        use_activation=config.resource_to_edge_conv_activation)

        
        #combines per edge aggregated resource info, action embedding (resources aggregated by the distance to the other resources)
        self.edge_info_combination_net = MLP(1 + (self.edge_aggregated_resources_hidden_dim if self.use_per_edge_resource else 0) +  self.resource_hidden_dim* (2 if self.next_action_embedding else 1), 
                                             self.edge_info_hidden_dim, 
                                             **config.edge_info_combination_net)
                
        
        
        if self.use_edge_conv:
            edge_conv_block = EdgeConvBlock(self.edge_to_edge_matrix, 
                                            self.edge_info_hidden_dim, 
                                            self.edge_info_hidden_dim, 
                                            bias=True, 
                                            use_only_distance=self.use_only_distance_in_edge_conv,
                                            norm_edge_conv_adj=config.norm_edge_conv_adj,
                                            edge_conv_adjacency_activation=config.edge_conv_adjacency_activation)
            
            
            self.edge_conv = EdgeConv(edge_conv_block,
                                        config.number_of_edge_conv_layers,
                                    hidden_dim=self.edge_info_hidden_dim, 
                                    use_activation=config.use_activation_after_conv)
        
        self.q_net = MLP(self.edge_info_hidden_dim, 
                         1, 
                         **config.q_net)
        

    def forward(self, batch, state=None, info={}):        
        resource_observations = torch.as_tensor(batch["resource_observations"], device=self.device)
        distance_to_action = torch.as_tensor(batch["distance_to_action"], device=self.device)
        current_position = torch.as_tensor(batch["current_position"].squeeze(-1), device=self.device, dtype=torch.long)


        distance_to_resource_for_each_action = torch.index_select(self.sparse_dist_matrix,0, current_position).to_dense() #AxR
        if self.normalize_route_distance_aggr:
            distance_to_resource_for_each_action = self.learnable_dist_route_param * distance_to_resource_for_each_action/3000.0
                
        if self.norm_sim_matrix:
            distance_to_resource_for_each_action = distance_to_resource_for_each_action / (distance_to_resource_for_each_action.sum(-1, keepdim=True) + 1e-10)
        
        if self.use_resource_id_embedding:
            resource_input = torch.cat([resource_observations, self.resource_id_embedding(self.resource_ids).unsqueeze(0).expand(resource_observations.size(0), -1, -1)], dim=-1)
        else:
            resource_input = resource_observations

        
        resource_embedding = self.resource_embedding_net(resource_input) #RxH
                
        
        action_embedding = torch.bmm(distance_to_resource_for_each_action, resource_embedding) #A x H
        if self.action_targets_only:
            action_embedding = action_embedding * 0
            action_embedding = action_embedding.detach()
        
        
        if self.next_action_embedding:
            resource_embedding_next = resource_embedding.unsqueeze(1).repeat(1, self.number_of_actions, 1, 1)
            next_action_embedding_all = torch.stack([torch.bmm(self.sparse_dist_matrix, resource_embedding_next[i]) for i in range(resource_embedding.size(0))]) #AxAxH 
        
            next_action_embedding_reduced = next_action_embedding_all.mean(-2) #AxH

        if self.use_per_edge_resource:
            per_edge_resource = self.resource_to_edge_conv(resource_embedding) #action target encoder #AxH
        
        if not self.use_distance_to_action:
            distance_to_action = distance_to_action * 0
        
        if self.next_action_embedding:
            per_edge_resource = self.edge_info_combination_net(torch.cat((per_edge_resource, action_embedding, next_action_embedding_reduced, distance_to_action.unsqueeze(-1)), dim=-1))
        else:
            if self.use_per_edge_resource:
                per_edge_resource = self.edge_info_combination_net(torch.cat((per_edge_resource, action_embedding, distance_to_action.unsqueeze(-1)), dim=-1))
            else:
                per_edge_resource = self.edge_info_combination_net(torch.cat((action_embedding, distance_to_action.unsqueeze(-1)), dim=-1))

            
        if self.edge_conv_without_route:
            edge_r = self.resource_to_edge_conv(resource_embedding) #AxH
            
            if self.use_edge_conv:
                edge_embedding = self.edge_conv(edge_r)      
            else:
                edge_embedding = edge_r

            edge_embedding = self.action_net(torch.cat((per_edge_resource, edge_embedding), dim=-1))

        else:
            if self.use_edge_conv:
                edge_embedding = self.edge_conv(per_edge_resource)     
            else:
                edge_embedding = per_edge_resource
        
        q =  self.q_net(edge_embedding).squeeze(-1)
            
            
        return q, state

        
    
        
class TimeEmbedding(nn.Module):
    def __init__(self, adjacency_matrix_resources, number_of_resources, cfg) -> None:
        super().__init__()
        
        self.resource_history_hidden_dim = cfg.resource_history_hidden_dim #64
        self.resource_history_feature_dim = 18 #TODO
        self.number_of_resources = number_of_resources
        
        self.final_output_size = cfg.final_output_size

        
        self.with_r_id = cfg.with_r_id
        self.to_edge = cfg.to_edge
        self.resource_id_embedding_dim = cfg.resource_id_embedding_dim
        
        
        #self.resource_history_net = nn.Linear(self.resource_history_feature_dim, self.resource_history_hidden_dim) #TODO
        
        
        #if self.edge_aggregation_first: #TODO
        
        if self.to_edge:
            self.resource_to_edge_conv = ResourceToEdgeConv(adjacency_matrix_resources, self.resource_history_hidden_dim, self.final_output_size, bias=True) #TODO
        
        
        self.time_series_module = nn.GRU(input_size=self.resource_history_feature_dim + (self.resource_id_embedding_dim if self.with_r_id else 0),
                                         hidden_size=self.resource_history_hidden_dim, 
                                         num_layers=cfg.number_of_gru_layers,
                                         batch_first=True, 
                                         dropout=0)#TODO 
        
        if not self.to_edge:
            self.after_ts = nn.Linear(self.resource_history_hidden_dim, self.final_output_size) #TODO
        
        
    def forward(self, time_series_packed, batch_size, current_time):
        #if True: #TODO
        #    h = self.resource_history_net(time_series_packed.data)
        #    h = torch.nn.utils.rnn.PackedSequence(h, time_series_packed.batch_sizes, time_series_packed.sorted_indices, time_series_packed.unsorted_indices)
            
    #   packed_input = nn.utils.rnn.pack_padded_sequence(history, lengths=lengths, batch_first=True, enforce_sorted=False)
        self.time_series_module.flatten_parameters()
        _, h_t = self.time_series_module(time_series_packed)

        time_series_representation = h_t[-1] #B*R x H  #TODO
        
        time_series_representation = time_series_representation.unflatten(dim=0, sizes=(batch_size, self.number_of_resources)) #BxRxH
        #e,f = torch.nn.utils.rnn.pad_packed_sequence(_, batch_first=True)
        #test = torch.gather(e.cpu(),0, (f-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 128))
        #torch.allclose(test[:, 0,:], time_series_representation[0].cpu())
        
        if not self.to_edge:
            time_series_representation = self.after_ts(time_series_representation)
        
        if self.to_edge:
            time_series_representation = self.resource_to_edge_conv(time_series_representation) #BxAxH

        return time_series_representation

def length_to_mask(length, max_length):
    #mask: True = do not attend, False = attend
    mask = torch.arange(max_length).expand(len(length), max_length) >= length.unsqueeze(-1)
    ##TODO
    
    #seq mask = N*mxSxS
    #seq_mask = (torch.triu(torch.ones(max_length, max_length), diagonal=1)).bool()
    
    return mask



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 64):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1),:]
        return self.dropout(x)

class TransformerTimeSeries(nn.Module):
    def __init__(self, adjacency_matrix_resources, number_of_resources, cfg, number_of_actions) -> None:
        super().__init__()
        
        self.with_r_id = cfg.with_r_id #True
        self.with_pos_enc = cfg.with_pos_enc #True
        self.hidden_size = cfg.time_series_hidden_dim # 256
        self.with_current_time = cfg.with_current_time #False
        self.with_time_mlp = cfg.with_time_mlp #True
        self.with_skip_connection_time_mlp = cfg.with_skip_connection_time_mlp #False
        self.final_output_size = cfg.final_output_size
        
        
        if self.with_r_id:
            self.resource_id_embedding_dim = cfg.resource_id_embedding_dim #256
            resource_ids =  torch.arange(0, number_of_resources, 1) #.to(device)
            
            self.register_buffer('resource_ids', resource_ids, persistent=False)
            
            self.resource_id_embedding = torch.nn.Embedding(number_of_resources, self.resource_id_embedding_dim) #.to(device)

        
        #TODO norm_first, actviation F.gelu
        layer = nn.TransformerEncoderLayer(self.hidden_size, 
                                           nhead=cfg.transformer_encoder_layer.nhead, #4 
                                           dropout=cfg.transformer_encoder_layer.dropout, #0.0
                                           batch_first=True, 
                                           norm_first=cfg.transformer_encoder_layer.norm_first, #False
                                           activation=cfg.transformer_encoder_layer.activation) #gelu
        
        
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.number_of_transformer_layers)#TODO !!!!!!!
        self.linear = nn.Linear(18+(self.resource_id_embedding_dim if self.with_r_id else 0) + (1 if self.with_current_time else 0), self.hidden_size, bias=False) #TODO!! 
        #self.adjacency_matrix_resources = adjacency_matrix_resources
        
        self.use_second_transformer = cfg.use_second_transformer
        self.with_time_mlp2 = cfg.with_time_mlp2

        if self.use_second_transformer:
            self.second_transformer = nn.TransformerEncoder(layer, num_layers=cfg.number_of_transformer_layers)#TODO !!!!!!!
            self.positional_encoding_edges = PositionalEncoding(self.hidden_size, dropout=0, max_len=number_of_actions) #TODO !!!!
            self.time_mlp2 = MLP(input_size=self.hidden_size, 
                                output_size=self.final_output_size, 
                                **cfg.time_mlp)

        
        self.number_of_resources = number_of_resources
        
        if self.with_pos_enc:
            self.positional_encoding = PositionalEncoding(self.hidden_size, dropout=0, max_len=16) #TODO !!!!
        
        self.to_edge = cfg.to_edge
        if self.to_edge:
            self.resource_to_edge_conv = ResourceToEdgeConv(adjacency_matrix_resources, self.hidden_size, self.hidden_size, bias=True) #TODO

        if self.with_time_mlp:
            #TODO !! n_layer=6 activ_last_layer=True
            self.time_mlp = MLP(input_size=self.hidden_size, 
                                output_size=self.hidden_size if self.use_second_transformer else self.final_output_size, 
                                **cfg.time_mlp)
        
        
    def forward(self, time_series, batch_size, current_time, lengths):
        pad_mask = length_to_mask(lengths, time_series.shape[2])
        pad_mask = pad_mask.to(time_series.device)
        
        
        if self.with_r_id:
            if self.with_current_time:
                time_series = torch.cat((time_series, self.resource_id_embedding(self.resource_ids).unsqueeze(0).unsqueeze(2).expand(batch_size, -1, time_series.shape[-2], -1), current_time.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(batch_size, time_series.shape[1], time_series.shape[2], 1)), dim=-1)            
            else:
                time_series = torch.cat((time_series, self.resource_id_embedding(self.resource_ids).unsqueeze(0).unsqueeze(2).expand(batch_size, -1, time_series.shape[-2], -1)), dim=-1)            

                
        time_series = time_series.flatten(end_dim=1)
        
        h = self.linear(time_series)#TODO
        
        if self.with_pos_enc:
            h = h * math.sqrt(self.hidden_size)
            h = self.positional_encoding(h)
        
        out = self.transformer.forward(h, src_key_padding_mask=pad_mask)
        
        last = out[:,0,:] #select the first one because the last may be padded...
        #torch.take_along_dim(test, (lengths-1).unsqueeze(-1).unsqueeze(-1), 1).squeeze().shape
        last =  last.unflatten(dim=0, sizes=(batch_size, self.number_of_resources))#TODO     
        
        if self.with_time_mlp:
            if self.with_skip_connection_time_mlp:
                last = last + self.time_mlp.forward(last)
            else:
                last = self.time_mlp.forward(last)
        
        
        if self.to_edge:
            last = self.resource_to_edge_conv(last)
            
        if self.use_second_transformer:
            last = last * math.sqrt(self.hidden_size)
            last = self.positional_encoding_edges(last)

            last = self.second_transformer.forward(last)
            
            
            if self.with_time_mlp2:
                if self.with_skip_connection_time_mlp:
                    last = last + self.time_mlp2.forward(last)
                else:
                    last = self.time_mlp2.forward(last)
        
        return last
    


class BatchNorm(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features)

    def forward(self,h, *args, **kwargs):
        return self.norm(h.reshape(-1, h.size(-1))).reshape(*h.shape)


class TianTimeAT(nn.Module):

    def __init__(self, distance_matrix, adjacency_matrix, distance_matrix_resources, observation_space, device, config):
        super().__init__()
        self.time_ac = True
        self.device = device
        
        self.number_of_actions = observation_space["distance_to_action"].shape[0]
        
        self.resource_hidden_dim = config.resource_hidden_dim #256
        self.resource_id_embedding_dim = config.resource_id_embedding_dim #64
        self.use_resource_id_embedding = config.use_resource_id_embedding #True
        self.use_resource_mlp_net = config.use_resource_mlp_net #True
        self.attend_resources_to_resources = config.attend_resources_to_resources #True
        self.use_resource_batch_norm = config.use_resource_batch_norm #True
        self.use_edge_batch_norm = config.use_edge_batch_norm #True
        self.use_different_token_for_each_edge = config.use_different_token_for_each_edge #True
        
        self.distance_matrix = (distance_matrix / distance_matrix.max()).to(self.device)
        self.distance_matrix_resources = (distance_matrix_resources /distance_matrix_resources.max()).to(self.device)
        self.scaling_net = MLP(input_size=self.distance_matrix.size(1),
                        output_size=distance_matrix.size(1), hidden_size=1024, number_of_layers=2, activation_after_last_layer=False, layer_normalization=True, activation=nn.GELU)

        self.resource_dim = observation_space["resource_observations"].shape[1]
        self.number_of_resources = observation_space["resource_observations"].shape[0]
        
        if self.use_resource_id_embedding:
            self.register_buffer("resource_ids", torch.arange(0, self.number_of_resources, 1))
            self.resource_id_embedding = nn.Embedding(self.number_of_resources, self.resource_id_embedding_dim)

        self.scaling_net_resource_to_resource = MLP(input_size=self.distance_matrix_resources.size(1), 
                                                    output_size=self.distance_matrix_resources.size(1), 
                                                    number_of_layers=2, 
                                                    hidden_size=1024,
                                                    activation_after_last_layer=nn.GELU,
                                                    layer_normalization=True)
        
        if self.attend_resources_to_resources:
            self.resource_distance_attention = nn.MultiheadAttention(self.resource_hidden_dim, num_heads=4, batch_first=True, dropout=0)
            self.resource_distance_ff_block = SkipConnection(MLP(input_size=self.resource_hidden_dim, output_size=self.resource_hidden_dim, number_of_layers=2, hidden_size=1024, layer_normalization=False, activation_after_last_layer=False, activation=nn.GELU))
            self.resource_distance_norm1 = nn.LayerNorm(self.resource_hidden_dim)
            self.resource_distance_norm2 = nn.LayerNorm(self.resource_hidden_dim)

        self.resource_embedding_first = nn.Linear(self.resource_dim+ (self.resource_id_embedding_dim if self.use_resource_id_embedding else 0), self.resource_hidden_dim)
        
        if self.use_resource_mlp_net:
            self.resource_embedding_net = SkipConnection(MLP(input_size=self.resource_hidden_dim, output_size=self.resource_hidden_dim, **config.resource_embedding_net))
        
        
        
        edge_encoder_layer = nn.TransformerEncoderLayer(d_model=self.resource_hidden_dim, nhead=4, dropout=0.0, batch_first=True)
        self.edge_attention_net = nn.TransformerEncoder(edge_encoder_layer, num_layers=config.edge_attention_net.num_layers)
                
        self.agent_embedding_net = MLP(input_size=observation_space["distance_to_action"].shape[0],output_size=observation_space["distance_to_action"].shape[0], number_of_layers=2, hidden_size=256, activation_after_last_layer=False, layer_normalization=True, activation=nn.GELU) #TODO
        
        representation_layer = nn.TransformerEncoderLayer(d_model=2*self.resource_hidden_dim, nhead=4, dropout=0.0, batch_first=True)
        self.representation_net = nn.TransformerEncoder(representation_layer, num_layers=config.representation_net.num_layers)
        
        
        self.special_token = torch.Tensor([0]).long().to(self.device)

        if self.use_different_token_for_each_edge:
            self.regular_token = torch.arange(1, self.number_of_actions+1, step=1, dtype=torch.long, device=self.device)
        else:
            self.regular_token = torch.Tensor([1]).long().to(self.device)
        
        self.token_type_embedding = nn.Embedding(1 + self.number_of_actions if self.use_different_token_for_each_edge else 2, self.resource_hidden_dim) #0 = special attention node, 1 = regular
        
        if self.use_resource_batch_norm:
            self.resource_batch_norm_before_attention = BatchNorm(self.resource_hidden_dim)
            self.resource_batch_norm = BatchNorm(self.resource_hidden_dim)
        
        if self.use_edge_batch_norm:
            self.edge_batch_norm = BatchNorm(self.resource_hidden_dim)
            self.edge_batch_norm_after_agent = BatchNorm(self.resource_hidden_dim)
    
        self.output_dim = 512
        
        if not self.time_ac:
            self.final_net = MLP(input_size=512, output_size=self.output_dim, hidden_size=1024, number_of_layers=2)

    def forward(self, batch, state=None, info={}):
        batch = {
            key: torch.as_tensor(value, dtype=torch.float, device=self.device)
            for key, value in batch.items()
        }
        resource_observations = batch["resource_observations"]
        distance_to_action = batch["distance_to_action"]
        #resource_history = state["resource_history"] #BxAxRxTxD
        #resource_history_lengths = state["resource_history_lengths"].long().cpu() #BxAxRx1 #TODO the cpu call is ugly because unnecessary transfer to GPU and back
        
        #agent_information = state["agent_information"]
        
        if self.use_resource_id_embedding:
            resource_input = torch.cat([resource_observations, self.resource_id_embedding(self.resource_ids).unsqueeze(0).expand(resource_observations.size(0), -1, -1)], dim=-1)
        else:
            resource_input = resource_observations
        
        #resource embedding
        resource_embedding = self.resource_embedding_first(resource_input)
        
        if self.use_resource_mlp_net:
            resource_embedding = self.resource_embedding_net(resource_embedding)
        
        
        if self.use_resource_batch_norm:
            resource_embedding = self.resource_batch_norm_before_attention(resource_embedding)
        
        
        #TODO include time series
        
        batch_size = resource_embedding.size(0)
        
        if self.attend_resources_to_resources:
            scaling_resource_to_resource = self.scaling_net_resource_to_resource(self.distance_matrix_resources)
            resource_distance_attention, _ = self.resource_distance_attention.forward(query=resource_embedding, key=resource_embedding, value=scaling_resource_to_resource @ resource_embedding)
            
            resource_distance_attention = self.resource_distance_norm1(resource_distance_attention + resource_embedding)
            
            resource_distance_attention =  self.resource_distance_norm2(resource_distance_attention + self.resource_distance_ff_block(resource_distance_attention))
            
        else:
            resource_distance_attention = resource_embedding
        
        if self.use_resource_batch_norm:
            resource_distance_attention = self.resource_batch_norm(resource_distance_attention)
        
        # distance matrix is in format edges_with_resources x resources
        # similarity_matrix = self.scaling_net(
        # self.distance_matrix.unsqueeze(-1)).squeeze(-1)

        similarity_matrix = self.scaling_net(self.distance_matrix)
        
        ## similarity_matrix = similarity_matrix / similarity_matrix.sum(-1, keepdim=True)
        #similarity_matrix = F.softmax(similarity_matrix, dim=-1)
        similarity_matrix = F.gelu(similarity_matrix)
        
        edge_embedding = similarity_matrix @ resource_distance_attention

        edge_embedding = self.edge_attention_net(edge_embedding) + edge_embedding
        
        
        if self.use_edge_batch_norm:
            edge_embedding = self.edge_batch_norm(edge_embedding)
        
        agent_embedding = self.agent_embedding_net(distance_to_action)
        
        agent_embedding = F.gelu(agent_embedding).unsqueeze(-1)
        #agent_embedding = F.softmax(agent_embedding, dim=-1).unsqueeze(-1)
        
        edge_embedding = agent_embedding * edge_embedding + edge_embedding
        
        if self.use_edge_batch_norm:
            edge_embedding = self.edge_batch_norm_after_agent(edge_embedding)
        
        
        if self.use_different_token_for_each_edge:
            edge_inputs = torch.cat([self.token_type_embedding(self.regular_token).unsqueeze(0).expand(batch_size, -1, -1), edge_embedding], dim=-1)
        else:
            edge_inputs = torch.cat([self.token_type_embedding(self.regular_token).unsqueeze(0).expand(*edge_embedding.shape[:-1], -1), edge_embedding], dim=-1)
            
        edge_token = self.token_type_embedding(self.special_token).unsqueeze(0).expand(edge_embedding.size(0),1, -1)
        special_token_input = torch.cat([edge_token, torch.ones_like(edge_token)], dim=-1)

        input = torch.cat([edge_inputs, special_token_input], dim=-2)
        
        x = self.representation_net(input)
        
        if self.time_ac:
            return x, state
        
        x = x[:,-1,:]
        x = self.final_net(x) + x
        
        return x, state
    
    
class TimeATActor(nn.Module):
    def __init__(self, net, action_shape, model_config, softmax_output=True) -> None:
        super().__init__()
        self.net = net
        self.softmax_output = softmax_output
        self.independent = model_config.actor.independent
        
        
        if not self.independent:
            self.per_output_net = MLP(input_size=self.net.output_dim, output_size=model_config.actor.before_merge_dim, **model_config.actor.per_output_net)
            self.merge = MLP(input_size=model_config.actor.before_merge_dim*action_shape, output_size=action_shape, **model_config.actor.merge_net)
        else:
            self.per_output_net = MLP(input_size=self.net.output_dim, output_size=1, number_of_layers=1, hidden_size=64, activation_after_last_layer=False)

        
    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        logits, hidden = self.net(obs, state)
        
        logits = logits[:,:-1, :]
        if self.independent:
            logits = self.per_output_net(logits).flatten(-2)
        else: 
            logits = self.per_output_net(logits)
            logits = self.merge(logits.flatten(-2))
        
        
        if self.softmax_output:
            logits = F.softmax(logits, dim=-1)
        return logits, hidden


class TimeATCritic(nn.Module):
    def __init__(self, net, last_size=1) -> None:
        super().__init__()
        self.net = net
        
        self.final = MLP(input_size=self.net.output_dim, output_size=last_size, number_of_layers=2, hidden_size=256, activation_after_last_layer=False, layer_normalization=True)
        
        
    def forward(
        self, obs: Union[np.ndarray, torch.Tensor], **kwargs: Any
    ) -> torch.Tensor:
        """Mapping: s -> V(s)."""
        logits, _ = self.net(obs, state=kwargs.get("state", None))
        logits = logits[:,-1,:]

        return self.final(logits)
