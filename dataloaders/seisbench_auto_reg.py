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
from dataloaders.data_utils.seisbench_utils.augmentations import QuantizeAugmentation, FilterZChannel, FillMissingComponents, AutoregressiveShift
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


class SeisBenchAutoReg(SeisbenchDataLit):
    def __init__(self, sample_len: int = 2048, bits: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.sample_len = sample_len
        self.bits = bits
        self.setup()

    def setup(self):
        data = sbd.ETHZ(sampling_rate=100)
        train, dev, test = data.train_dev_test()

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

        augmentations = [
            sbg.WindowAroundSample(list(phase_dict.keys()), samples_before=6000, windowlen=16000, selection="random",
                                   strategy="variable"),
            sbg.RandomWindow(windowlen=self.sample_len + 1, strategy="pad"),
            FillMissingComponents(),
            # sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=30, dim=0),
            FilterZChannel(),
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak"),
            sbg.ChangeDtype(np.float32),
            QuantizeAugmentation(bits=self.bits),
            AutoregressiveShift()
        ]

        self.dataset_train = sbg.GenericGenerator(train)
        self.dataset_val = sbg.GenericGenerator(dev)
        self.dataset_test = sbg.GenericGenerator(test)

        self.dataset_train.add_augmentations(augmentations)
        self.dataset_val.add_augmentations(augmentations)
        self.dataset_test.add_augmentations(augmentations)

        self.num_classes = 2**self.bits
