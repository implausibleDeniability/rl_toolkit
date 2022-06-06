import torch
from rl_toolkit.mlp import MLP


class DQN(torch.nn.Module):
    def __init__(
        self,
        observation_dimension: int,
        action_dimension: int,
        hidden_layers: list[int] = [],
    ):
        super().__init__()
        layer_dims = [observation_dimension] + hidden_layers + [action_dimension]
        self.main = MLP(layer_dims)

    def forward(self, observation):
        action_logits = self.main(observation)
        return action_logits
