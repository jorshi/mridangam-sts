"""
Dataset and dataloader for mridangam stroke dataset
"""
from pathlib import Path
from typing import List
from typing import Literal
from typing import Tuple

import pytorch_lightning as pl
import torch
import torchaudio
import torchcrepe
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from tqdm import tqdm


class MridangamDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        num_workers: int,
        attribute: Literal["tonic", "stroke"],
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        assert (
            self.dataset_dir.exists()
        ), f"Dataset directory {self.dataset_dir} does not exist"
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.audio_files = None
        self.stokes = None
        self.tonics = None
        self.attribute = attribute
        self.sample_rate = torchcrepe.SAMPLE_RATE

    def prepare_data(self):
        # What preprocessing is needed?
        self.audio_files = []
        self.strokes = []
        self.tonics = []

        # Find all audio files
        file_list = list(self.dataset_dir.glob("**/*.wav"))
        print(f"Found {len(file_list)} audio files in dataset directory")

        # Load audio files
        print(f"Loading audio files and resampling to {self.sample_rate}Hz...")
        for file in tqdm(file_list):
            # Prepare audio file
            x, sr = torchaudio.load(file)
            x = torchaudio.functional.resample(
                x, sr, self.sample_rate, lowpass_filter_width=512
            )

            # Pad or trim to 1 second
            if x.shape[1] < self.sample_rate:
                x = torch.nn.functional.pad(x, (0, self.sample_rate - x.shape[1]))
            elif x.shape[1] > self.sample_rate:
                x = x[:, : self.sample_rate]

            self.audio_files.append(x)

            # Get annotations from file name
            annotations = Path(file).stem.split("_")[-1].split("-")
            self.strokes.append(annotations[0])
            self.tonics.append(annotations[1])

    def preprocess_crepe(self):
        pass

    def setup(self, stage: str):
        """
        Assign train/val/test datasets for use in dataloaders.

        Args:
            stage: Current stage (fit, validate, test)
        """
        if self.attribute == "tonic":
            attr = self.tonics
        elif self.attribute == "stroke":
            attr = self.stokes

        if stage == "fit":
            self.train_dataset = AudioDataset(self.audio_files, attr, split="train")
            self.val_dataset = AudioDataset(self.audio_files, attr, split="val")
        elif stage == "validate":
            self.val_dataset = AudioDataset(self.audio_files, attr, split="val")
        elif stage == "test":
            self.test_dataset = AudioDataset(self.audio_files, attr, split="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )


class AudioDataset(Dataset):
    def __init__(
        self,
        audio_files: List[torch.Tensor],
        annotations: List[str],
        split: str,
        seed: int = 42,
    ):
        super().__init__()
        self.audio_files = audio_files
        self.annotations = annotations
        assert len(self.audio_files) == len(self.annotations)

        # Create class labels for annotations
        self.labels = set(self.annotations)
        self.label_key = {k: i for i, k in enumerate(sorted(self.labels))}
        self.class_label = [self.label_key[j] for j in self.annotations]

        # Split the dataset into train, validation, and test sets
        self.seed = seed
        self.split = split
        self.ids = list(range(len(self.audio_files)))
        self._random_split(split)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        index = self.ids[idx]
        return self.audio_files[index], self.class_label[index]

    @property
    def num_classes(self):
        return len(self.labels)

    def _random_split(self, split: str):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            split: Split to return. Must be one of 'train', 'val', or 'test'.
        """
        if split not in ["train", "val", "test"]:
            raise ValueError("Invalid split. Must be one of 'train', 'val', or 'test'.")

        splits = random_split(
            self.ids,
            [0.8, 0.1, 0.1],
            generator=torch.Generator().manual_seed(self.seed),
        )

        # Set the file list based on the split
        if split == "train":
            self.ids = splits[0]
        elif split == "val":
            self.ids = splits[1]
        elif split == "test":
            self.ids = splits[2]
