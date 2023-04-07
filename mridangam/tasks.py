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
        embedding_model: a model to produce a fixed embedding
    """

    def __init__(self, embedding_model: torch.nn.Module) -> None:
        super().__init__()
        self.embedding_model = embedding_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding_model(x)

    def _do_step(self, batch: Tuple[torch.tensor, str]):
        return 0.0

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
