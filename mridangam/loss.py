"""
Loss functions
"""
from typing import Literal

import auraloss
import torch


class MSS(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = auraloss.freq.MultiResolutionSTFTLoss()

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(y_hat, y)


class STFTDiff(torch.nn.Module):
    def __init__(
        self, diff: Literal["time", "frequency"], n_fft: int = 2048, hop_size: int = 128
    ):
        super().__init__()
        assert diff in ["time", "frequency"]
        self.diff = -1 if diff == "time" else -2
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.window = self.register_buffer("window", torch.hann_window(self.n_fft))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        assert y.dim() == 3, "Input must be of shape (batch, channels, length)"
        assert y.shape[1] == 1, "Input must be mono"

        Y = self._magnitude_stft(y.squeeze(1))
        Y_diff = torch.diff(Y, dim=self.diff)

        loss = torch.sum(torch.square(Y[..., 1:] - Y_diff))
        loss = loss / torch.sum(torch.square(Y))

        return loss

    def _magnitude_stft(self, x: torch.Tensor) -> torch.Tensor:
        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            window=self.window,
            return_complex=True,
            normalized=False,
            onesided=True,
        )
        return torch.abs(X)


class StationaryRegularization(torch.nn.Module):
    def __init__(self, n_fft: int = 2048, hop_size: int = 128, eps: float = 1e-8):
        self.diff_time = STFTDiff("time", n_fft=n_fft, hop_size=hop_size)
        self.diff_freq = STFTDiff("frequency", n_fft=n_fft, hop_size=hop_size)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.diff_time(x) / (self.diff_freq(x) + self.eps)


class TransientRegularization(torch.nn.Module):
    def __init__(self, n_fft: int = 2048, hop_size: int = 128, eps: float = 1e-8):
        self.diff_time = STFTDiff("time", n_fft=n_fft, hop_size=hop_size)
        self.diff_freq = STFTDiff("frequency", n_fft=n_fft, hop_size=hop_size)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.diff_freq(x) / (self.diff_time(x) + self.eps)
