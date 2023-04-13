"""
Lightning tasks for mridangam drum problems
"""
from typing import Optional
from typing import Tuple

import pytorch_lightning as pl
import torch


class MridangamTonicClassification(pl.LightningModule):
    """
    Lightning task for estimating the tonic of a given mridangam sound

    Args:
        model: a model to produce a fixed embedding
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _do_step(self, batch: Tuple[torch.Tensor, torch.Tensor, str]):
        _, embedding, label = batch
        y_hat = self(embedding)
        y_hat = y_hat.squeeze(1)
        loss = self.loss_fn(y_hat, label)
        return loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, str], batch_idx: int
    ):
        loss = self._do_step(batch)
        self.log("train/loss", loss, on_epoch=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, str], batch_idx: int
    ):
        loss = self._do_step(batch)
        self.log("validation/loss", loss)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, str], batch_idx: int):
        loss = self._do_step(batch)
        self.log("test/loss", loss)
        return loss


class TransientStationarySeparation(pl.LightningModule):
    """
    Lightning Module for performaing transient vs. stationary separation from an
    audio signal.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        reconstruction_loss: torch.nn.Module,
        transient_loss: torch.nn.Module,
        stationary_loss: torch.nn.Module,
        reconstruction_weight: float = 1.0,
        transient_weight: float = 1.0,
        stationary_weight: float = 1.0,
        film_encoder: Optional[torch.nn.Module] = None,
        learning_rate: float = 1e-4,
    ) -> None:
        super().__init__()
        self.model = model
        self.r_loss = reconstruction_loss
        self.t_loss = transient_loss
        self.s_loss = stationary_loss
        self.r_weight = reconstruction_weight
        self.t_weight = transient_weight
        self.s_weight = stationary_weight
        self.film_encoder = film_encoder
        self.learning_rate = learning_rate

    def forward(
        self, x: torch.Tensor, embedding: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.film_encoder is not None:
            embedding = self.film_encoder(embedding)

        y = self.model(x, embedding)
        assert y.shape[1] == 2, "Model output must have two channels"
        transient = y[:, 0:1]
        stationary = y[:, 1:2]
        return transient, stationary

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _do_step(self, batch: Tuple[torch.tensor, str]):
        audio, embedding, _ = batch
        y_trans, y_stat = self(audio, embedding)
        y_hat = y_trans + y_stat

        r_loss = self.r_weight * self.r_loss(y_hat, audio)
        t_loss = self.t_weight * self.t_loss(y_trans)
        s_loss = self.s_weight * self.s_loss(y_stat)

        return r_loss, t_loss, s_loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, str], batch_idx: int
    ):
        r_loss, t_loss, s_loss = self._do_step(batch)
        self.log("train/reconstruction_loss", r_loss, on_epoch=True)
        self.log("train/transient_loss", t_loss, on_epoch=True)
        self.log("train/sustain_loss", s_loss, on_epoch=True)

        loss = r_loss + t_loss + s_loss
        self.log("train/loss", loss, on_epoch=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, str], batch_idx: int
    ):
        r_loss, t_loss, s_loss = self._do_step(batch)
        self.log("validation/reconstruction_loss", r_loss)
        self.log("validation/transient_loss", t_loss)
        self.log("validation/sustain_loss", s_loss)

        loss = r_loss + t_loss + s_loss
        self.log("validation/loss", loss)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, str], batch_idx: int):
        r_loss, t_loss, s_loss = self._do_step(batch)
        self.log("test/reconstruction_loss", r_loss)
        self.log("test/transient_loss", t_loss)
        self.log("test/sustain_loss", s_loss)

        loss = r_loss + t_loss + s_loss
        self.log("test/loss", loss)
        return loss
