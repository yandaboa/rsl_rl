import torch
import torch.nn as nn

from rsl_rl.networks import MLP
from rsl_rl.utils import resolve_nn_activation

from functools import reduce


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation
    paper: https://arxiv.org/pdf/1709.07871
    """
    def __init__(self, num_input_channels: int, hiddens: list[int], num_features: int):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.num_features = num_features

        hiddens = list(hiddens)  # avoid mutating caller's list
        self.embedding_dim = hiddens[0]
        hiddens.append(num_features * 2)

        self.encoder = nn.Sequential(
            nn.Linear(num_input_channels, self.embedding_dim),
            nn.ReLU(),
        )
        layers = []
        for i in range(len(hiddens) - 1):
            layers.append(nn.Linear(hiddens[i], hiddens[i + 1]))
            if i < len(hiddens) - 2:  # no activation on final output layer
                layers.append(nn.ReLU())
        self.generator = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_input_channels
        assert features.shape[-1] == self.num_features

        embed = self.encoder(x)
        out = self.generator(embed)
        scale, shift = torch.split(out, self.num_features, dim=-1)
        return scale * features + shift

class MLP_FiLM(MLP):
    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int] | list[int],
        hidden_dims: tuple[int] | list[int],
        film_num_input_channels: int,
        activation: str = "elu",
        last_activation: str | None = None,
        film_hiddens: list[int] = [256],
    ) -> None:
        nn.Sequential.__init__(self) 

        activation_mod = resolve_nn_activation(activation)
        last_activation_mod = resolve_nn_activation(last_activation) if last_activation is not None else None

        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

        # Create layers sequentially
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims_processed[0]))
        layers.append(FiLM(film_num_input_channels, film_hiddens, hidden_dims_processed[0]))
        layers.append(activation_mod)

        for layer_index in range(len(hidden_dims_processed) - 1):
            layers.append(nn.Linear(hidden_dims_processed[layer_index], hidden_dims_processed[layer_index + 1]))
            layers.append(FiLM(film_num_input_channels, film_hiddens, hidden_dims_processed[layer_index + 1]))
            layers.append(activation_mod)

        # Add last layer
        if isinstance(output_dim, int):
            layers.append(nn.Linear(hidden_dims_processed[-1], output_dim))
        else:
            # Compute the total output dimension
            total_out_dim = reduce(lambda x, y: x * y, output_dim)
            # Add a layer to reshape the output to the desired shape
            layers.append(nn.Linear(hidden_dims_processed[-1], total_out_dim))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

        # Add last activation function if specified
        if last_activation_mod is not None:
            layers.append(last_activation_mod)

        # Register the layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    def forward(self, x: torch.Tensor, film_input: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self):
            if i == len(self) - 1:
                self.last_features = x
            if isinstance(layer, FiLM):
                x = layer(film_input, x)
            else:
                x = layer(x)
        return x