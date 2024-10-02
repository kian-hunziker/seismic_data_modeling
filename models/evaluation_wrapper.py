import seisbench
import seisbench.models as sbm
import seisbench.generate as sbg

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
from abc import abstractmethod, ABC

# Allows to import this file in both jupyter notebook and code
try:
    from .augmentations import DuplicateEvent
except ImportError:
    from dataloaders.data_utils.seisbench_utils.augmentations import DuplicateEvent

from evaluation.eval_utils import load_checkpoint, get_pipeline_components, print_hparams, get_model_summary
from dataloaders.seisbench_auto_reg import get_eval_augmentations
from models.phasenet_wrapper import PhaseNetWrapper
import torch
import torch.nn as nn

# Phase dict for labelling. We only study P and S phases without differentiating between them.
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


def vector_cross_entropy(y_pred, y_true, eps=1e-5):
    """
    Cross entropy loss

    :param y_true: True label probabilities
    :param y_pred: Predicted label probabilities
    :param eps: Epsilon to clip values for stability
    :return: Average loss across batch
    """
    h = y_true * torch.log(y_pred + eps)
    if y_pred.ndim == 3:
        h = h.mean(-1).sum(
            -1
        )  # Mean along sample dimension and sum along pick dimension
    else:
        h = h.sum(-1)  # Sum along pick dimension
    h = h.mean()  # Mean over batch axis
    return -h


class SeisBenchModuleLit(pl.LightningModule, ABC):
    """
    Abstract interface for SeisBench lightning modules.
    Adds generic function, e.g., get_augmentations
    """

    @abstractmethod
    def get_augmentations(self):
        """
        Returns a list of augmentations that can be passed to the seisbench.generate.GenericGenerator

        :return: List of augmentations
        """
        pass

    def get_train_augmentations(self):
        """
        Returns the set of training augmentations.
        """
        return self.get_augmentations()

    def get_val_augmentations(self):
        """
        Returns the set of validation augmentations for validations during training.
        """
        return self.get_augmentations()

    @abstractmethod
    def get_eval_augmentations(self):
        """
        Returns the set of evaluation augmentations for evaluation after training.
        These augmentations will be passed to a SteeredGenerator and should usually contain a steered window.
        """
        pass

    @abstractmethod
    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        """
        Predict step for the lightning module. Returns results for three tasks:

        - earthquake detection (score, higher means more likely detection)
        - P to S phase discrimination (score, high means P, low means S)
        - phase location in samples (two integers, first for P, second for S wave)

        All predictions should only take the window defined by batch["window_borders"] into account.

        :param batch:
        :return:
        """
        score_detection = None
        score_p_or_s = None
        p_sample = None
        s_sample = None
        return score_detection, score_p_or_s, p_sample, s_sample


class PhasePickerLit(SeisBenchModuleLit):
    def __init__(self, ckpt_path=None, pretrained_name=None, pretrained_dataset=None, avg_latent=False):
        super().__init__()
        self.save_hyperparameters()
        self.avg_latent = avg_latent
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if pretrained_name is not None and pretrained_dataset is not None:
            seisbench.use_backup_repository()
            self.pretrained_benchmark = True
            if pretrained_name == 'PhaseNet':
                self.model = sbm.PhaseNet.from_pretrained(pretrained_dataset)
                self.encoder = nn.Identity()
                self.device = nn.Identity()
                self.sample_len = 3001
            if pretrained_name == 'EQTransformer':
                self.model = sbm.EQTransformer.from_pretrained(pretrained_dataset)
                self.sample_len = 6000
        else:
            self.pretrained_benchmark = False
            pl_module, hparams, specific_ckpt = load_checkpoint(ckpt_path, location=device, return_path=True, simple=True)
            try:
                self.sample_len = hparams['dataset']['sample_len']
            except:
                print('Could not determine sample length from checkpoint file')
                self.sample_len = 4096

            self.model = pl_module.model
            self.encoder = pl_module.encoder
            self.decoder = pl_module.decoder

    def forward(self, x):
        if self.pretrained_benchmark:
            x = x.transpose(1, 2)
            x = self.model(x, logits=True)
            if isinstance(self.model, sbm.EQTransformer):
                x = torch.stack(x, dim=-1)
            else:
                x = x.transpose(1, 2)
        else:
            x = self.encoder(x)
            x, _ = self.model(x, None)
            if self.avg_latent:
                # we want to evaluate the representation learned by the model and skip the decoder
                return x
            x = self.decoder(x, None)
        return x

    def shared_step(self, batch):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass

    def get_augmentations(self):
        return get_eval_augmentations(sample_len=self.sample_len, d_data=3, bits=0)

    def get_eval_augmentations(self):
        return get_eval_augmentations(sample_len=self.sample_len, d_data=3, bits=0)

    def predict_step(self, batch, batch_idx=None, dataloader_idx=None):
        x = batch['X']
        window_borders = batch['window_borders']

        pred = self.forward(x)
        if self.avg_latent:
            m = torch.mean(pred, dim=1)
            return m
        else:
            pred = F.softmax(pred, dim=-1)

            score_detection = torch.zeros(pred.shape[0])
            score_p_or_s = torch.zeros(pred.shape[0])
            p_sample = torch.zeros(pred.shape[0], dtype=int)
            s_sample = torch.zeros(pred.shape[0], dtype=int)

            for i in range(pred.shape[0]):
                start_sample, end_sample = window_borders[i]
                local_pred = pred[i, start_sample:end_sample, :]

                score_detection[i] = torch.max(1 - local_pred[:, -1])
                score_p_or_s[i] = torch.max(local_pred[:, 0]) / torch.max(local_pred[:, 1])
                p_sample[i] = torch.argmax(local_pred[:, 0])
                s_sample[i] = torch.argmax(local_pred[:, 1])
            return score_detection, score_p_or_s, p_sample, s_sample
