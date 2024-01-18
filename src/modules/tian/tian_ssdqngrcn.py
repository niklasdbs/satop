import torch

from utils.torch.models.mlp import MLP



class TianGRCN(torch.nn.Module):
    def __init__(self, observation_space, device, distance_matrix, config) -> None:
        super().__init__()
        self.device = device
        self.add_distance_to_action = config.add_distance_to_action
        self.dqn = True
        self.resource_dim = observation_space["resource_observations"].shape[1]
        self.action_embedding_dim = 1 if self.dqn else 8
        self.number_of_actions = observation_space["distance_to_action"].shape[0]
        self.resource_embedding_dim = config.resource_embedding_dim
        self.output_dim = 1024

        self.distance_matrix = (distance_matrix / distance_matrix.max()).to(self.device)

        self.q_net = MLP(input_size=self.resource_embedding_dim + (1 if self.add_distance_to_action else 0),
                         output_size=self.action_embedding_dim,
                         **config.q_net)
        if not self.dqn:
            self.final_net = MLP(input_size=self.number_of_actions*self.action_embedding_dim,
                                output_size=self.output_dim,
                                number_of_layers=4,
                                hidden_size=2048)
        
        self.resource_embedding_net = MLP(input_size=self.resource_dim,
                                          output_size=self.resource_embedding_dim,
                                          **config.resource_embedding_net)

        self.use_nn_scaling = config.use_nn_scaling

        if self.use_nn_scaling:
            self.scaling_net = MLP(input_size=1,
                                   output_size=1,
                                   **config.scaling_net)
        else:
            self.scaling = 1  #todo

    def forward(self, batch, state=None, info={}):
        batch = {
            key: torch.as_tensor(value, dtype=torch.float, device=self.device)
            for key, value in batch.items()
        }
        resource_observations = batch["resource_observations"]

        # state is in format resource x resource_features
        # distance matrix is in format edges_with_resources x resources
        resource_encoding = self.resource_embedding_net(resource_observations)
        # resource_encoding is in format resources x resource_embedding_dim

        # similarity matrix has format edges_with_resources x resources
        similarity_matrix = self.calculate_similarity_matrix()

        x = similarity_matrix @ resource_encoding
        # x has shape edges_with_resources x resource_embedding_dim

        if self.add_distance_to_action:
            distance_to_action = batch["distance_to_action"]
            x = torch.cat([x, distance_to_action.unsqueeze(-1)], dim=-1)

        q = self.q_net(x).squeeze(-1)
        # q has shape edges_with_resources
        if self.dqn:
            return q, state
        else:
            hidden_rep = self.final_net(q.flatten(-2))
            
            return hidden_rep, state

    def calculate_similarity_matrix(self):
        if self.use_nn_scaling:
            # distance matrix is in format edges_with_resources x resources
            similarity_matrix = self.scaling_net(
                self.distance_matrix.unsqueeze(-1)).squeeze(-1)

            similarity_matrix = similarity_matrix / similarity_matrix.sum(-1, keepdim=True)
            # similarity_matrix = similarity_matrix / \
            #     (similarity_matrix != 0).sum(-1, keepdim=True)

        else:
            similarity_matrix = torch.exp(-self.scaling * self.distance_matrix)

        return similarity_matrix

