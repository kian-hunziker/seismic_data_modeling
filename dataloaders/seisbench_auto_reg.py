import os
import os.path
import torch
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import dataloaders.data_utils.costa_rica_utils as cu

from scipy.signal import decimate
from torch.utils.data import Dataset, DataLoader
from dataloaders.base import SeisbenchDataLit
from dataloaders.data_utils.seisbench_utils.augmentations import QuantizeAugmentation, FilterZChannel, \
    FillMissingComponents, AutoregressiveShift, TransposeLabels, TransposeSeqChannels, RandomMask
from evaluation.eval_sashimi import moving_average

from dataloaders.data_utils.signal_encoding import quantize_encode, decode_dequantize, normalize_11_torch, normalize_11
import seisbench
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding

from obspy.clients.fdsn import Client
from obspy import UTCDateTime

from dataloaders.data_utils.signal_encoding import quantize_encode

phase_dict = {
    "trace_p_arrival_sample": "P",
    "trace_pP_arrival_sample": "P",
    "trace_P_arrival_sample": "P",
    "trace_P1_arrival_sample": "P",
    "trace_Pg_arrival_sample": "P",
    "trace_Pn_arrival_sample": "P",
    "trace_PmP_arrival_sample": "P",
    "trace_pwP_arrival_sample": "P",
    "trace_pwPm_arrival_sample": "P",
    "trace_s_arrival_sample": "S",
    "trace_S_arrival_sample": "S",
    "trace_S1_arrival_sample": "S",
    "trace_Sg_arrival_sample": "S",
    "trace_SmS_arrival_sample": "S",
    "trace_Sn_arrival_sample": "S",
}


def apply_training_fraction(
        training_fraction,
        train_data,
) -> None:
    """
  Reduces the size of train_data to train_fraction by inplace filtering.
  Filter blockwise for efficient memory savings.

  Args:
    training_fraction: Training fraction between 0 and 1.
    train_data: Training dataset

  Returns:
    None
  """

    if not 0.0 < training_fraction <= 1.0:
        raise ValueError("Training fraction needs to be between 0 and 1.")

    if training_fraction < 1:
        blocks = train_data["trace_name"].apply(lambda x: x.split("$")[0])
        unique_blocks = blocks.unique()
        np.random.shuffle(unique_blocks)
        target_blocks = unique_blocks[: int(training_fraction * len(unique_blocks))]
        target_blocks = set(target_blocks)
        mask = blocks.isin(target_blocks)
        train_data.filter(mask, inplace=True)


def remove_unused_augmentations(augmentations):
    return [a for a in augmentations if a is not None]


def get_eval_augmentations(sample_len: int = 4096, d_data: int = 3, bits: int = 0, norm_type: str = 'peak'):
    augmentations = [
        sbg.SteeredWindow(windowlen=sample_len, strategy='pad'),
        sbg.ChangeDtype(np.float32),
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type=norm_type),
        TransposeSeqChannels() if d_data == 3 else None,
    ]
    augmentations = remove_unused_augmentations(augmentations)
    '''
            sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=sample_len // 3, windowlen=sample_len,
                               selection="random",
                               strategy="variable"),
        # sbg.RandomWindow(windowlen=self.sample_len, strategy="pad"),
        FillMissingComponents(),
        sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=30, dim=0),
        FilterZChannel() if d_data == 1 else None,
        sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
        sbg.ChangeDtype(np.float32),
        QuantizeAugmentation(bits=bits),
        TransposeSeqChannels() if d_data == 3 else None,
        TransposeLabels(),
    '''
    return augmentations


class SeisBenchAutoReg(SeisbenchDataLit):
    def __init__(self,
                 sample_len: int = 2048,
                 bits: int = 8,
                 d_data: int = 1,
                 preload: bool = False,
                 normalize_first: bool = False,
                 dataset_name: str = 'ETHZ',
                 norm_type: str = 'peak',
                 masking: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.sample_len = sample_len
        self.d_data = d_data
        self.bits = bits
        self.preload = preload
        self.normalize_first = normalize_first
        self.norm_type = norm_type
        self.masking = masking

        dataset_kwargs = {
            'sampling_rate': 100,
            'component_order': 'ZNE',
            'dimension_order': 'NCW'
        }
        if self.preload:
            dataset_kwargs['cache'] = 'full'

        if dataset_name == 'ETHZ':
            self.data = sbd.ETHZ(**dataset_kwargs)
        elif dataset_name == 'GEOFON':
            self.data = sbd.GEOFON(**dataset_kwargs)
        elif dataset_name == 'STEAD':
            self.data = sbd.STEAD(**dataset_kwargs)
        elif dataset_name == 'INSTANCE':
            self.data = sbd.InstanceCountsCombined(**dataset_kwargs)
        else:
            print(f'Unknown dataset: {dataset_name}')

        if self.preload:
            self.data.preload_waveforms(pbar=True)

        self.setup()

    def setup(self):
        train, dev, test = self.data.train_dev_test()

        window_len = self.sample_len + 1 if self.masking == 0 else self.sample_len

        augmentations = [
            sbg.ChangeDtype(np.float32) if self.normalize_first else None,
            sbg.Normalize(
                demean_axis=-1,
                amp_norm_axis=-1,
                amp_norm_type=self.norm_type,
            ) if self.normalize_first else None,
            sbg.WindowAroundSample(list(phase_dict.keys()),
                                   samples_before=self.sample_len,
                                   windowlen=2 * self.sample_len,
                                   selection="random",
                                   strategy="variable"),
            sbg.RandomWindow(windowlen=window_len, strategy="pad"),
            FillMissingComponents(),
            # sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=30, dim=0),
            FilterZChannel() if self.d_data == 1 else None,
            sbg.ChangeDtype(np.float32) if not self.normalize_first else None,
            sbg.Normalize(
                demean_axis=-1,
                amp_norm_axis=-1,
                amp_norm_type=self.norm_type,
            ) if self.normalize_first else None,
            QuantizeAugmentation(bits=self.bits) if self.bits > 0 else None,
            TransposeSeqChannels() if self.d_data == 3 else None,
            AutoregressiveShift() if self.masking == 0 else RandomMask(p=self.masking),
        ]

        augmentations = remove_unused_augmentations(augmentations)

        self.dataset_train = sbg.GenericGenerator(train)
        self.dataset_val = sbg.GenericGenerator(dev)
        self.dataset_test = sbg.GenericGenerator(test)

        self.dataset_train.add_augmentations(augmentations)
        self.dataset_val.add_augmentations(augmentations)
        self.dataset_test.add_augmentations(augmentations)

        if self.bits > 0:
            self.num_classes = 2 ** self.bits


class SeisBenchPhasePick(SeisbenchDataLit):
    def __init__(
            self,
            sample_len: int = 2048,
            bits: int = 8,
            d_data: int = 1,
            preload: bool = False,
            sample_boundaries=(None, None),
            sigma=20,
            dataset_name: str = 'ETHZ',
            norm_type: str = 'peak',
            training_fraction: float = 1.0,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.sample_len = sample_len
        self.bits = bits
        self.d_data = d_data
        self.preload = preload
        self.sample_boundaries = sample_boundaries
        self.sigma = sigma
        self.norm_type = norm_type
        self.training_fraction = training_fraction

        dataset_kwargs = {
            'sampling_rate': 100,
            'component_order': 'ZNE',
            'dimension_order': 'NCW'
        }
        if self.preload:
            dataset_kwargs['cache'] = 'full'

        if dataset_name == 'ETHZ':
            data = sbd.ETHZ(**dataset_kwargs)
        elif dataset_name == 'GEOFON':
            data = sbd.GEOFON(**dataset_kwargs)
        elif dataset_name == 'STEAD':
            data = sbd.STEAD(**dataset_kwargs)
        elif dataset_name == 'INSTANCE':
            data = sbd.InstanceCountsCombined(**dataset_kwargs)
        else:
            print(f'Unknown dataset: {dataset_name}')
        self.train, self.dev, self.test = data.train_dev_test()

        if self.training_fraction < 1.0:
            apply_training_fraction(training_fraction=training_fraction, train_data=self.train)

        if self.preload:
            self.train.preload_waveforms(pbar=True)
            self.dev.preload_waveforms(pbar=True)

        self.setup()

    def setup(self):

        augmentations = [
            # In 2/3 of the cases, select windows around picks, to reduce amount of noise traces in training.
            # Uses strategy variable, as padding will be handled by the random window.
            # In 1/3 of the cases, just returns the original trace, to keep diversity high.
            sbg.OneOf(
                [
                    sbg.WindowAroundSample(
                        list(phase_dict.keys()),
                        samples_before=self.sample_len,
                        windowlen=2 * self.sample_len,
                        selection="random",
                        strategy="variable",
                    ),
                    sbg.NullAugmentation(),
                ],
                probabilities=[2, 1],
            ),
            sbg.RandomWindow(
                low=self.sample_boundaries[0],
                high=self.sample_boundaries[1],
                windowlen=self.sample_len,
                strategy="pad",
            ),
            FillMissingComponents(),
            FilterZChannel() if self.d_data == 1 else None,
            sbg.ChangeDtype(np.float32),
            sbg.Normalize(
                demean_axis=-1,
                amp_norm_axis=-1,
                amp_norm_type=self.norm_type,
            ),
            QuantizeAugmentation(bits=self.bits) if self.bits > 0 else None,
            sbg.ProbabilisticLabeller(
                label_columns=phase_dict, sigma=self.sigma, dim=0
            ),
            TransposeSeqChannels() if self.d_data == 3 else None,
            TransposeLabels(),
        ]
        augmentations = remove_unused_augmentations(augmentations)

        self.dataset_train = sbg.GenericGenerator(self.train)
        self.dataset_val = sbg.GenericGenerator(self.dev)
        self.dataset_test = sbg.GenericGenerator(self.test)

        self.dataset_train.add_augmentations(augmentations)
        self.dataset_val.add_augmentations(augmentations)
        self.dataset_test.add_augmentations(augmentations)

        if self.bits > 0:
            self.num_classes = 2 ** self.bits


def phase_pick_test():
    data_config = {
        'sample_len': 4096,
        'bits': 0,
        'd_data': 3,
        'normalize_first': True,
        'dataset_name': 'ETHZ',
        'training_fraction': 0.1,
        'masking': 0.5,
        'norm_type': 'std'
    }
    loader_config = {
        'batch_size': 128,
        'num_workers': 0,
        'shuffle': True,
    }
    dataset = SeisBenchAutoReg(**data_config)
    train_loader = DataLoader(dataset.dataset_train, **loader_config)

    batch = next(iter(train_loader))
    print(len(train_loader.dataset))

    total_avg = 0
    for i, batch in enumerate(train_loader):
        autoreg_mse = torch.nn.functional.mse_loss(batch['X'], batch['y'])
        total_avg += autoreg_mse
        print(f'autoreg_mse: {autoreg_mse :.4f}, log_mse: {torch.log(autoreg_mse):.4f}')
    print('total_avg: ', total_avg / len(train_loader))


if __name__ == "__main__":
    phase_pick_test()
