import torch


class MLP(torch.nn.Module):
    def __init__(self, layer_dims: list[int]):
        super().__init__()
        n_linear_transformations = len(layer_dims) - 1
        linear_input_dims = layer_dims[:-1]
        linear_output_dims = layer_dims[1:]
        layers = []
        for idx, (input_dim, output_dim) in enumerate(
            zip(linear_input_dims, linear_output_dims)
        ):
            if idx == n_linear_transformations - 1:
                layer = self._make_last_linear_layer(input_dim, output_dim)
            else:
                layer = self._make_intermediate_linear_layer(input_dim, output_dim)
            layers.append(layer)
        self.mlp = torch.nn.Sequential(*layers)

    def _make_intermediate_linear_layer(
        self, input_dim: int, output_dim: int
    ) -> torch.nn.Module:
        layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(output_dim),
        )
        return layer

    def _make_last_linear_layer(
        self, input_dim: int, output_dim: int
    ) -> torch.nn.Module:
        layer = torch.nn.Linear(input_dim, output_dim)
        return layer

    def forward(self, x):
        return self.mlp(x)
