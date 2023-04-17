"""
Lightning tasks for mridangam drum problems
"""
from typing import Optional
from typing import Tuple

import auraloss
import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy


class MridangamTonicClassification(pl.LightningModule):
    """
    Lightning task for estimating the tonic of a given mridangam sound

    Args:
        model: a model to produce a fixed embedding
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        reduce_factor: float = 0.5,
        reduce_patience: int = 25,
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.reduce_factor = reduce_factor
        self.reduce_patience = reduce_patience
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy("multiclass", num_classes=self.model.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=self.reduce_factor,
            patience=self.reduce_patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "validation/loss",
                "interval": "epoch",
            },
        }

    def _do_step(self, batch: Tuple[torch.Tensor, torch.Tensor, str]):
        _, embedding, label = batch
        y_hat = self(embedding)
        y_hat = y_hat.squeeze(1)
        loss = self.loss_fn(y_hat, label)
        return loss, y_hat

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, str], batch_idx: int
    ):
        loss, _ = self._do_step(batch)
        self.log("train/loss", loss, on_epoch=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, str], batch_idx: int
    ):
        loss, _ = self._do_step(batch)
        self.log("validation/loss", loss)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, str], batch_idx: int):
        loss, y_hat = self._do_step(batch)
        self.log("test/loss", loss)
        self.log("test/accuracy", self.accuracy(y_hat, batch[-1]))
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
        reduce_factor: float = 0.5,
        reduce_patience: int = 25,
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
        self.lr = learning_rate
        self.reduce_factor = reduce_factor
        self.reduce_patience = reduce_patience

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=self.reduce_factor,
            patience=self.reduce_patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "validation/loss",
                "interval": "epoch",
            },
        }

    def _do_step(self, batch: Tuple[torch.tensor, str]):
        audio, embedding, _ = batch
        y_trans, y_stat = self(audio, embedding)
        y_hat = y_trans + y_stat

        r_loss = self.r_weight * self.r_loss(y_hat, audio)
        t_loss = self.t_weight * self.t_loss(y_trans)
        s_loss = self.s_weight * self.s_loss(y_stat)

        return r_loss, t_loss, s_loss, y_hat

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, str], batch_idx: int
    ):
        r_loss, t_loss, s_loss, _ = self._do_step(batch)
        self.log("train/reconstruction_loss", r_loss, on_epoch=True)
        self.log("train/transient_loss", t_loss, on_epoch=True)
        self.log("train/sustain_loss", s_loss, on_epoch=True)

        loss = r_loss + t_loss + s_loss
        self.log("train/loss", loss, on_epoch=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, str], batch_idx: int
    ):
        r_loss, t_loss, s_loss, _ = self._do_step(batch)
        self.log("validation/reconstruction_loss", r_loss)
        self.log("validation/transient_loss", t_loss)
        self.log("validation/sustain_loss", s_loss)

        loss = r_loss + t_loss + s_loss
        self.log("validation/loss", loss)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, str], batch_idx: int):
        r_loss, t_loss, s_loss, y_hat = self._do_step(batch)
        self.log("test/reconstruction_loss", r_loss)
        self.log("test/transient_loss", t_loss)
        self.log("test/sustain_loss", s_loss)

        loss = r_loss + t_loss + s_loss
        self.log("test/loss", loss)

        l1error = torch.nn.L1Loss()(y_hat, batch[0])
        mss = auraloss.freq.MultiResolutionSTFTLoss(hop_sizes=[512, 2014, 256])
        msserror = mss(y_hat, batch[0])

        self.log("test/waveform_error", l1error)
        self.log("test/mss_error", msserror)

        return loss
