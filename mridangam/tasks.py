"""
Lightning tasks for mridangam drum problems
"""
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

    def _do_step(self, batch: Tuple[torch.tensor, str]):
        _, embedding, label = batch
        y_hat = self(embedding)
        y_hat = y_hat.squeeze(1)
        loss = self.loss_fn(y_hat, label)
        return loss

    def training_step(self, batch: Tuple[torch.Tensor, str], batch_idx: int):
        loss = self._do_step(batch)
        self.log("train/loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss = self._do_step(batch)
        self.log("validation/loss", loss)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss = self._do_step(batch)
        self.log("test/loss", loss)
        return loss
