import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation

from functools import reduce


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation
    paper: https://arxiv.org/pdf/1709.07871
    We expect an embedding, which we then linearly project into the feature space
    """
    def __init__(self, num_input_channels: int, num_features: int):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.num_features = num_features
        self.generator = nn.Linear(self.num_input_channels, self.num_features * 2)

    def forward(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_input_channels
        assert features.shape[-1] == self.num_features

        out = self.generator(x)
        scale, shift = torch.split(out, self.num_features, dim=-1)
        return scale * features + shift

class MLP_FiLM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int] | list[int],
        hidden_dims: tuple[int] | list[int],
        film_num_input_channels: int,
        activation: str = "elu",
        last_activation: str | None = None,
        film_hiddens: tuple[int] | list[int] = (256,),
    ) -> None:
        super().__init__()

        activation_mod = resolve_nn_activation(activation)
        last_activation_mod = resolve_nn_activation(last_activation) if last_activation is not None else None

        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

        # create film encoder
        film_dims = [film_num_input_channels, *film_hiddens]
        film_encoder_layers = []
        for i in range(len(film_dims) - 1):
            film_encoder_layers.append(nn.Linear(film_dims[i], film_dims[i + 1]))
            if i < len(film_dims) - 2:
                film_encoder_layers.append(activation_mod)
        self.film_encoder = nn.Sequential(*film_encoder_layers)
        self.film_embed_dim = film_dims[-1]

        # Create layers sequentially
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims_processed[0]))
        layers.append(FiLM(self.film_embed_dim, hidden_dims_processed[0]))
        layers.append(activation_mod)

        for layer_index in range(len(hidden_dims_processed) - 1):
            layers.append(nn.Linear(hidden_dims_processed[layer_index], hidden_dims_processed[layer_index + 1]))
            layers.append(FiLM(self.film_embed_dim, hidden_dims_processed[layer_index + 1]))
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

        # Register the MLP layers separately so they don't mix with film_encoder.
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, film_input: torch.Tensor) -> torch.Tensor:

        film_embed = self.film_encoder(film_input)
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                self.last_features = x
            if isinstance(layer, FiLM):
                x = layer(film_embed, x)
            else:
                x = layer(x)
        return x

    def init_weights(self, scales: float | tuple[float]) -> None:
        def get_scale(idx: int) -> float:
            return scales[idx] if isinstance(scales, (list, tuple)) else scales

        for idx, module in enumerate(self.layers):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=get_scale(idx))  # type: ignore[arg-type]
                nn.init.zeros_(module.bias)

    def get_features(self) -> torch.Tensor:
        if getattr(self, "last_features", None) is None:
            raise ValueError("No features have been computed yet. Call forward() first, or make sure your forward() method caches features")
        return self.last_features

    def __getitem__(self, idx):
        raise RuntimeError("Runing through a sliced version of MLP_FiLM is not supported. Use the forward() method instead.")