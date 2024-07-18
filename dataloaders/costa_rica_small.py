import os
import os.path
import torch
import glob
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from dataloaders.base import SequenceDataset
from dataloaders.utils.costa_rica_utils import get_metadata, format_label


class CostaRicaSmall(Dataset):
    """
    Most basic dataset for Costa Rica waveforms. Will only return one channel. The items returned
    consist of (x, y), both with dimensions [sample_len, 1]. The sequences are shifted by one element.
    The resulting batches have dimensions [batch_size, sample_len, 1].
    """
    def __init__(self, file: str = None, sample_len: int = 2048, train: bool = True):
        """
        :param file: directory containing waveforms
        :param sample_len: length of returned sequence, should be smaller than approx 8'500'000
        :param train: if true, the first 95% of files will be used for training and validation
                        else: the remaining 5% of files will be used for testing
        """
        super().__init__()

        self.file = file
        self.sample_len = sample_len
        self.file_paths = glob.glob(os.path.join(self.file, '*.pt'))
        self.train = train

        self.num_test_examples = int(len(self.file_paths) * 0.95)

        if self.train:
            self.file_paths = self.file_paths[:self.num_test_examples]
        else:
            self.file_paths = self.file_paths[self.num_test_examples:]

        # mapping from label strings to file paths
        self.label_to_path = {}
        for file_path in self.file_paths:
            label = format_label(file_path.split('/')[-1])
            self.label_to_path[label] = file_path

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = torch.load(file_path)

        start_idx = np.random.randint(0, data.shape[-1] - self.sample_len - 1)
        x = data[start_idx:start_idx + self.sample_len].type(torch.FloatTensor)
        y = data[start_idx + 1: start_idx + 1 + self.sample_len].type(torch.FloatTensor)

        # expand sequences to accommodate 'channel' (data is 1d)
        return x.unsqueeze(1), y.unsqueeze(-1)


class CostaRicaSmallLighting(SequenceDataset):
    def __init__(self, data_dir: str = None, **kwargs):
        super().__init__(data_dir, **kwargs)
        self.setup()

    def setup(self):
        self.dataset_train = CostaRicaSmall(self.data_dir, sample_len=self.hparams.sample_len, train=True)
        self.dataset_test = CostaRicaSmall(self.data_dir, sample_len=self.hparams.sample_len, train=False)
        self.split_train_val(self.hparams.val_split)


def plot_first_examples(dataloader, num_examples=3, plot_len=None):
    fig, ax = plt.subplots(num_examples, 1)
    fig.suptitle('first three examples costa rica small')
    x, y = next(iter(dataloader))
    if plot_len is None:
        # set plot_len to sequence length
        plot_len = x.shape[1]
    for i, a in enumerate(ax):
        a.plot(x[i, :plot_len, 0], label='x')
        a.plot(y[i, :plot_len, 0], label='y')
    plt.legend()
    plt.show()


def run_tests():
    data_path = 'data/costa_rica/small_subset'
    sample_length = 1024
    batch_size = 32
    expected_num_train_files = 50
    expected_num_test_files = 3

    train_dataset = CostaRicaSmall(file=data_path, sample_len=sample_length, train=True)
    test_dataset = CostaRicaSmall(file=data_path, sample_len=sample_length, train=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print('#' * 32)
    print('Basic tests for CostaRicaSmall dataset')
    print('#' * 32, '\n')

    train_file_paths = train_dataset.file_paths
    test_file_paths = test_dataset.file_paths
    print(f'number of files in train dataset: {len(train_file_paths)}, expected: {expected_num_train_files}')
    print(f'number of files in test dataset: {len(test_file_paths)}, expected: {expected_num_test_files}')

    x, y = next(iter(train_loader))

    print(f'x shape: {x.shape}, y shape: {y.shape}, expected shapes: {batch_size}, {sample_length}, {1}')

    # plot 3 examples, x and y should be shifted by one position
    plot_first_examples(train_loader, plot_len=100)

    test_sanity_check = False
    for f in train_file_paths:
        test_sanity_check = test_sanity_check or f in test_file_paths

    print(f'test files found in training files: {test_sanity_check}')

    val_split = 0.5
    data_config = {'sample_len': sample_length, 'val_split': val_split}
    loader_config = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0, 'pin_memory': True}
    dataset_lightning = CostaRicaSmallLighting(data_dir=data_path, **data_config)
    train_loader_pl = dataset_lightning.train_dataloader(**loader_config)
    val_loader_pl = dataset_lightning.val_dataloader(**loader_config)
    test_loader_pl = dataset_lightning.test_dataloader(**loader_config)

    print('\n', '\n', '#' * 32)
    print('Basic tests for CostaRicaSmall Lightning dataset')
    print('#' * 32, '\n')

    print(f'number of training samples: {len(train_loader_pl.dataset)}, '
          f'expected: {int(expected_num_train_files * (1 - val_split))}')
    print(f'number of validation samples: {len(val_loader_pl.dataset)}, '
          f'expected: {int(expected_num_train_files * val_split)}')
    print(f'number of test samples: {len(test_loader_pl.dataset)}, '
          f'expected: {expected_num_test_files}')


if __name__ == '__main__':
    run_tests()
