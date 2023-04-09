"""
Models used for Mridangam tasks
"""
from typing import List

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _get_activation(activation: str):
    return getattr(torch.nn, activation)()


class DilatedResidualConvolution(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        activation: str = "GELU",
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
        self.activation = _get_activation(activation)
        self.residual = torch.nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3
        assert x.shape[1] == self.convolution.in_channels
        y = self.norm(x)
        y = self.convolution(y)
        y = self.activation(y)
        return y + self.residual(x)
