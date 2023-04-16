"""
Dataset and dataloader for mridangam stroke dataset
"""
import random
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
        unprocessed_dir: str,
        sample_rate: int = 48000,
        device: str = "cpu",
        max_files: int = None,
        train_size: float = 0.8,
        val_size: float = 0.1,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.attribute = attribute
        self.unprocessed_dir = Path(unprocessed_dir)
        self.sample_rate = sample_rate
        self.device = device
        self.max_files = max_files
        self.train_size = train_size
        self.val_size = val_size

        self.audio = None
        self.embeddings = None
        self.stokes = None
        self.tonics = None

    def prepare_data(self):
        # Storing all audio and embeddings in memory
        self.audio = []
        self.embeddings = []
        self.strokes = []
        self.tonics = []

        if not self.dataset_dir.exists():
            self.preprocess_crepe()

        # Find all audio files
        file_list = list(self.dataset_dir.glob("**/*.wav"))
        print(f"Found {len(file_list)} audio files in dataset directory")

        # Load audio files and pre-computed embeddings
        for file in tqdm(file_list):
            # Prepare audio file
            x, sr = torchaudio.load(file)
            assert sr == self.sample_rate, "Incorrect sample rate found!"
            self.audio.append(x)

            # Load embedding files
            embed_file = str(file).replace(".wav", ".pt")
            embed = torch.load(embed_file)
            self.embeddings.append(embed)

            # Get annotations from file name
            annotations = Path(file).stem.split("_")[-1].split("-")
            self.strokes.append(annotations[0])
            self.tonics.append(annotations[1])

    def preprocess_crepe(self):
        """
        Preprocess audio and save precomputing crepe embeddings
        """
        # Find all audio files
        assert self.unprocessed_dir.exists()
        file_list = list(self.unprocessed_dir.glob("**/*.wav"))
        print(f"Found {len(file_list)} audio files in unprocessed dataset directory")

        if self.max_files is not None:
            file_list = random.sample(file_list, self.max_files)

        # Create dataset dir
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        for file in tqdm(file_list):
            outfile = self.dataset_dir.joinpath(file.name)
            outfile_embed = str(outfile).replace(".wav", ".pt")

            # Get an embedding from crepe
            embed = torchcrepe.embed_from_file(file, device=self.device)
            embed = embed.detach().cpu()
            embed = embed.flatten(2)
            embed = torch.mean(embed, dim=1)
            torch.save(embed, outfile_embed)

            # Now save preprocessed audio
            x, sr = torchaudio.load(file)
            x = torchaudio.functional.resample(
                x, sr, self.sample_rate, lowpass_filter_width=512
            )

            # Pad or trim to 1 second
            if x.shape[1] < self.sample_rate:
                x = torch.nn.functional.pad(x, (0, self.sample_rate - x.shape[1]))
            elif x.shape[1] > self.sample_rate:
                x = x[:, : self.sample_rate]

            torchaudio.save(outfile, x, self.sample_rate)

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
            self.train_dataset = AudioDataset(
                self.audio,
                self.embeddings,
                attr,
                split="train",
                train_size=self.train_size,
                val_size=self.val_size,
            )
            self.val_dataset = AudioDataset(
                self.audio,
                self.embeddings,
                attr,
                split="val",
                train_size=self.train_size,
                val_size=self.val_size,
            )
        elif stage == "validate":
            self.val_dataset = AudioDataset(
                self.audio,
                self.embeddings,
                attr,
                split="val",
                train_size=self.train_size,
                val_size=self.val_size,
            )
        elif stage == "test":
            self.test_dataset = AudioDataset(
                self.audio,
                self.embeddings,
                attr,
                split="test",
                train_size=self.train_size,
                val_size=self.val_size,
            )

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
        audio: List[torch.Tensor],
        embeddings: List[torch.Tensor],
        annotations: List[str],
        split: str,
        seed: int = 42,
        train_size: float = 0.8,
        val_size: float = 0.1,
    ):
        super().__init__()
        self.audio = audio
        self.embeddings = embeddings
        self.annotations = annotations
        assert len(self.audio) == len(self.annotations) == len(embeddings)

        # Create class labels for annotations
        self.labels = set(self.annotations)
        self.label_key = {k: i for i, k in enumerate(sorted(self.labels))}
        self.class_label = [self.label_key[j] for j in self.annotations]

        # Split the dataset into train, validation, and test sets
        self.seed = seed
        self.split = split
        self.ids = list(range(len(self.audio)))
        self.train_size = train_size
        self.val_size = val_size
        self._random_split(split)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, str]:
        index = self.ids[idx]
        return self.audio[index], self.embeddings[index], self.class_label[index]

    @property
    def num_classes(self):
        return len(self.labels)

    def get_label_from_idx(self, idx):
        for key, value in self.label_key.items():
            if value == idx:
                return key
        return None

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
            [self.train_size, self.val_size, 1 - self.train_size - self.val_size],
            generator=torch.Generator().manual_seed(self.seed),
        )

        # Set the file list based on the split
        if split == "train":
            self.ids = splits[0]
        elif split == "val":
            self.ids = splits[1]
        elif split == "test":
            self.ids = splits[2]
