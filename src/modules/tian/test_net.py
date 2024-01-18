import torch

from utils.torch.models.mlp import MLP

class TestNet(torch.nn.Module):
    def __init__(self, observation_space, device) -> None:
        super().__init__()
        self.net = MLP(observation_space["resource_observations"].shape[0] *
                       observation_space["resource_observations"].shape[1], 
                       256, hidden_size=512, number_of_layers=4)
        self.net = self.net.to(device)
        self.output_dim = 256
        self.device = device

    def forward(self, batch, state=None, info={}):
        batch = {
            key: torch.as_tensor(value, dtype=torch.float, device=self.device)
            for key, value in batch.items()
        }

        resource_observations = batch["resource_observations"]

        hidden = self.net(resource_observations.flatten(-2))

        return hidden, state
