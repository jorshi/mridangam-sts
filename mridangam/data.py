"""
Dataset and dataloader for mridangam stroke dataset
"""
from pathlib import Path

import pytorch_lightning as pl
import torch


class MridangamDataModule(pl.LightningDataModule):
    def __init__(self, dataset_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        assert (
            self.dataset_dir.exists()
        ), f"Dataset directory {self.dataset_dir} does not exist"
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # download, split, etc...
        pass


class PitchDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir: str):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        assert (
            self.dataset_dir.exists()
        ), f"Dataset directory {self.dataset_dir} does not exist"

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
