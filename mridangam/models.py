"""
Models used for Mridangam tasks
"""
from typing import List
from typing import Optional

import torch


class MLP(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden: List[int],
        out_features: int,
    ) -> None:
        super().__init__()
        first_out = hidden[0] if hidden else out_features
        model = [torch.nn.Linear(in_features=in_features, out_features=first_out)]
        for i in range(len(hidden)):
            next_out = hidden[i + 1] if i + 1 < len(hidden) else out_features
            model.extend(
                [
                    torch.nn.ReLU(),
                    torch.nn.Linear(in_features=hidden[i], out_features=next_out),
                ]
            )
        self.model = torch.nn.Sequential(*model)
        self.num_classes = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _get_activation(activation: str):
    return getattr(torch.nn, activation)()


class FiLM(torch.nn.Module):
    """
    Feature Independent Layer-wise Modulation
    """

    def __init__(self, in_channels: int, embedding_size: int) -> None:
        super().__init__()
        self.norm = torch.nn.BatchNorm1d(in_channels, affine=False)
        self.net = torch.nn.Linear(embedding_size, in_channels * 2)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor):
        assert embedding.ndim == 3
        assert embedding.shape[1] == 1
        embedding = embedding.squeeze(1)

        film = self.net(embedding)
        gamma, beta = film.chunk(2, dim=-1)
        x = self.norm(x)
        return x * gamma[..., None] + beta[..., None]


class DilatedResidualConvolution(torch.nn.Module):
    """
    A single layer with a 1D residual convoluation

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolution kernel
        dilation: Dilation of the convolution
        activation: Activation function to use, defaults to GELU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        activation: str = "GELU",
        use_film: bool = False,
        film_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.norm = torch.nn.BatchNorm1d(in_channels)
        self.convolution = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding="same",
        )
        if use_film:
            self.film = FiLM(out_channels, film_size)
        self.activation = _get_activation(activation)
        self.residual = torch.nn.Conv1d(in_channels, out_channels, 1)

    def forward(
        self, x: torch.Tensor, embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert x.dim() == 3
        assert x.shape[1] == self.convolution.in_channels
        y = self.norm(x)
        y = self.convolution(y)
        if hasattr(self, "film"):
            assert embedding is not None
            y = self.film(y, embedding)
        y = self.activation(y)
        return y + self.residual(x)


class TCN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dilation_base: int = 2,
        num_layers: int = 8,
        kernel_size: int = 3,
        activation: str = "GELU",
        use_film: bool = False,
        film_size: int = None,
    ) -> None:
        super().__init__()
        if use_film:
            assert film_size is not None, "Must pass in film embedding size"

        self.in_projection = torch.nn.Conv1d(in_channels, hidden_channels, 1)
        self.out_projection = torch.nn.Conv1d(hidden_channels, out_channels, 1)

        net = []
        for i in range(num_layers):
            dilation = dilation_base**i
            net.append(
                DilatedResidualConvolution(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    activation=activation,
                    use_film=use_film,
                    film_size=film_size,
                )
            )
        self.net = torch.nn.ModuleList(net)

    def forward(
        self, x: torch.Tensor, embedding: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.in_projection(x)

        # Apply all the convolutionas and FiLM (if using)
        for layer in self.net:
            x = layer(x, embedding)

        x = self.out_projection(x)
        return x
