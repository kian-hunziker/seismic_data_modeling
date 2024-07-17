import os.path
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from dataloaders.base import SequenceDataset


class SineWaveformDataset(Dataset):
    def __init__(self, file: str = None, sample_len: int = 2048):
        super().__init__()
        self.file = file
        self.sample_len = sample_len
        self.waveform_len = None
        self.data = None
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )
        self.setup()

    def setup(self):
        if os.path.isfile(self.file):
            self.data = np.load(self.file)
        else:
            self._generate_dataset()
        self.waveform_len = self.data.shape[1]

    def _generate_dataset(self):
        num_waveforms = 500
        len_waveform = 32_000
        f_sampling = 16_000
        n_frequencies = 200
        min_freq = 80
        max_freq = 4_000

        frequencies = np.linspace(min_freq, max_freq, n_frequencies)
        t = np.linspace(0, len_waveform / f_sampling, len_waveform)

        data = np.zeros((len(frequencies)*5, len_waveform))

        for i in range(data.shape[0]):
            f=frequencies[i//5]
            data[i, :] = np.sin(2 * np.pi * f * t)*np.random.random()

        #data = np.zeros((num_waveforms, len_waveform))
        #for i in range(num_waveforms):
        #    data[i, :] = np.sin(2 * np.pi * 100 * t) + 0.05 * np.random.normal(size=len_waveform)

        print(f'saving waveform data to {self.file}')
        np.save(self.file, data)

        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        start_idx = np.random.randint(0, self.waveform_len - self.sample_len - 1)
        x = self.data[idx, start_idx:start_idx + self.sample_len].astype(np.float32)
        y = self.data[idx, start_idx + 1: start_idx + self.sample_len + 1].astype(np.float32)

        return x[:, np.newaxis], y[:, np.newaxis]


class SineWaveLightningDataset(SequenceDataset):
    def __init__(self, data_dir: str = None, **kwargs):
        super().__init__(data_dir, **kwargs)
        self.setup()

    def setup(self):
        self.dataset_train = SineWaveformDataset(self.data_dir, self.hparams.sample_len)
        self.split_train_val(self.hparams.val_split)


def plot_examples(dataloader, num_examples=3, title=''):
    data_iter = iter(dataloader)
    # plot 3 examples
    num_examples = 3
    fig, ax = plt.subplots(num_examples)
    for a in ax:
        x, y = next(data_iter)
        a.plot(x[0].numpy())
    plt.suptitle(f'{title}\n{num_examples} random examples')
    plt.show()


def sine_waveform_test():
    dataset = SineWaveformDataset(file='data/basic_waveforms/sine_waveform_2.npy', sample_len=200)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    plot_examples(dataloader, title='Standard Dataloader')

    print(len(dataloader))


def sine_waveforms_lightning_test():
    dataset = SineWaveLightningDataset(data_dir='data/basic_waveforms/sine_waveform_2.npy',
                                       **{'sample_len': 200, 'val_split': 0.1})
    dataloader = dataset.train_dataloader(**{'batch_size': 32, 'shuffle': False})
    plot_examples(dataloader, title='Lightning Dataloader')
    print(dataset)


if __name__ == '__main__':
    sine_waveform_test()
    sine_waveforms_lightning_test()
