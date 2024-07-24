import os
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as TF
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from einops import rearrange
import pytorch_lightning as pl


class SequenceDataset(pl.LightningDataModule):
    def __init__(self, data_dir: str = None, **kwargs):
        super().__init__()
        self.data_dir = Path(data_dir) if data_dir is not None else None

        self.save_hyperparameters()

        self.dataset_train = self.dataset_val = self.dataset_test = None

        self.d_data = 1
        self.num_classes = None

        self.init()

    def init(self):
        pass

    def setup(self):
        pass

    def split_train_val(self, val_split):
        train_len = int(len(self.dataset_train) * (1.0 - val_split))
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            self.dataset_train,
            (train_len, len(self.dataset_train) - train_len),
            generator=torch.Generator().manual_seed(getattr(self, 'seed', 42)),
        )

    def _dataloader(self, dataset, **kwargs):
        return torch.utils.data.DataLoader(dataset, **kwargs)

    def train_dataloader(self, **kwargs):
        return self._dataloader(self.dataset_train, **kwargs)

    def val_dataloader(self, **kwargs):
        return self._dataloader(self.dataset_val, **kwargs)

    def test_dataloader(self, **kwargs):
        return self._dataloader(self.dataset_test, **kwargs)


