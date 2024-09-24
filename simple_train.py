import datetime
import os
import threading
import time
import traceback

import hydra
import omegaconf
import re

import psutil
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary, LayerSummary
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataloaders.MNISTdataloader import MNISTdataset
from dataloaders.simple_waveform import SineWaveLightningDataset
from models.simple_test_models import ConvNet
from tasks.encoders import instantiate_encoder, load_encoder_from_file, instantiate_encoder_simple
from tasks.decoders import instantiate_decoder, load_decoder_from_file, instantiate_decoder_simple
from tasks.task import task_registry
from dataloaders.base import SeisbenchDataLit

from torch.utils.data import DataLoader

from utils.config_utils import instantiate
from utils import registry
from omegaconf import DictConfig, OmegaConf
from seisbench.util import worker_seeding


class SimpleSeqModel(pl.LightningModule):
    def __init__(self, config, d_data: int = 3):
        super().__init__()
        self.save_hyperparameters(config)
        self.d_data = d_data

        self.model = instantiate(registry.model, self.hparams.model)
        try:
            d_model = self.hparams.model.d_model
        except:
            d_model = 0
        self.encoder = instantiate_encoder_simple(self.hparams.encoder, d_data=self.d_data, d_model=d_model)
        self.decoder = instantiate_decoder_simple(self.hparams.decoder, d_data=self.d_data, d_model=d_model)

        self.task = instantiate(task_registry, self.hparams.task)
        self.criterion = self.task.loss
        self.loss_val = self.task.loss_val
        self.metrics = self.task.metrics

    def forward(self, batch, batch_idx=None):
        if isinstance(batch, dict):
            x = batch['X']
            y = batch['y']
        else:
            x, y = batch

        # encode
        x = self.encoder(x)

        # forward pass
        x, _ = self.model(x, None)

        # decode
        x = self.decoder(x, None)

        return x, y

    def _step_with_metrics(self, batch, batch_idx, prefix='train'):
        x, y = self.forward(batch, batch_idx)

        if prefix == 'train':
            loss = self.criterion(x, y)
        else:
            loss = self.loss_val(x, y)

        metrics = self.metrics(x, y)
        metrics['loss'] = loss
        metrics = {f'{prefix}/{metric}': val for metric, val in metrics.items()}

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
        )
        return loss

    def training_step(self, batch, batch_idx=None):
        loss = self._step_with_metrics(batch, batch_idx, prefix='train')

        # logging
        loss_epoch = {'trainer/loss': loss, 'trainer/epoch': self.current_epoch}
        self.log_dict(
            loss_epoch,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            add_dataloader_idx=False,
        )
        return loss

    def validation_step(self, batch, batch_idx=None):
        loss = self._step_with_metrics(batch, batch_idx, prefix='val')
        return loss

    def test_step(self, batch, batch_idx=None):
        loss = self._step_with_metrics(batch, batch_idx, prefix='test')
        return loss

    def configure_optimizers(self):
        # Normal parameters
        all_params = list(self.parameters())
        params = [p for p in all_params if not hasattr(p, "_optim")]

        args = self.hparams.optimizer.copy()
        with omegaconf.open_dict(args):
            del args['_name_']
        optimizer = instantiate(registry.optimizer, self.hparams.optimizer, params)
        print(f"Optimizer: {optimizer}")

        # Configure scheduler
        if "scheduler" not in self.hparams:
            return optimizer
        lr_scheduler = instantiate(
            registry.scheduler, self.hparams.scheduler, optimizer
        )
        scheduler = {
            "scheduler": lr_scheduler,
            "interval": self.hparams.train.interval,  # 'epoch' or 'step'
            "monitor": self.hparams.train.monitor,
            "name": "trainer/lr",  # default is e.g. 'lr-AdamW'
        }
        return [optimizer], [scheduler]


def create_trainer(config):
    current_date = datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S")
    model_name = config.model['_name_']
    experiment_name = config.experiment_name

    # setup logger
    if config.train.get("ckpt_path", None) is not None:
        current_date = config.train.get("ckpt_path").split('/')[-1]

    logger_t = TensorBoardLogger(
        save_dir='wandb_logs',
        name='MA',
        default_hp_metric=False,
        version=current_date,
    )

    logger_wab = WandbLogger(
        project='MA',
        save_dir='wandb_logs',
        name=experiment_name,
        version=current_date,
    )

    loggers = [logger_t, logger_wab]

    # monitor learning rate
    lr_monitor = pl.callbacks.LearningRateMonitor(
        logging_interval='step',
        # log_momentum=True,
    )
    top_checkpoints = ModelCheckpoint(
        save_top_k=3,
        monitor="val/loss",
        mode="min",
        dirpath=f'wandb_logs/MA/{current_date}/checkpoints',
        filename="callback-{epoch:d}-{step:d}",
    )

    last_checkpoints = ModelCheckpoint(
        save_top_k=2,
        monitor="epoch",
        mode="max",
        dirpath=f'wandb_logs/MA/{current_date}/checkpoints',
    )

    # initialize trainer
    trainer = pl.Trainer(logger=loggers, callbacks=[lr_monitor, top_checkpoints, last_checkpoints], **config.trainer)
    return trainer


def _extract_step_number(filename):
    match = re.search(r'step=(\d+)\.ckpt$', filename)
    if match:
        return int(match.group(1))
    return None


def load_checkpoint(checkpoint_path: str, location: str = 'cpu', return_path: bool = False) -> tuple[
    SimpleSeqModel, dict]:
    """
    Load checkpoint and hparams.yaml from specified path. Model is loaded to cpu.
    If no checkpoint is specified, the folder is searched for checkpoints and the one with the highest
    step number is returned.
    :param checkpoint_path: path to checkpoint file. The hparams file is extracted automatically
    :return: LightningSequenceModel, hparams
    """
    if not checkpoint_path.endswith('.ckpt'):
        # the path does not directly lead to checkpoint, we search for checkpoints in directory
        all_files = []

        # Walk through directory and subdirectories
        for root, dirs, files in os.walk(checkpoint_path):
            for file in files:
                file_path = os.path.join(root, file)
                step_number = _extract_step_number(file)
                if step_number is not None:
                    all_files.append((step_number, file_path))
        all_files.sort(key=lambda x: x[0])
        checkpoint_path = all_files[-1][1]

    hparam_path = '/'.join(checkpoint_path.split('/')[:-2]) + '/hparams.yaml'

    if not os.path.isfile(checkpoint_path):
        print('NO CHECKPOINT FOUND')
        return None
    if not os.path.isfile(hparam_path):
        print('NO HPARAM FOUND')
        hparams = None
    else:
        with open(hparam_path, 'r') as f:
            hparams = yaml.safe_load(f)

    print(f'Loading checkpoint from {checkpoint_path}')
    if hparams is not None:
        name = hparams['experiment_name']
        print(f'Experiment name: {name}')

    model = SimpleSeqModel.load_from_checkpoint(checkpoint_path, map_location=location)
    if return_path:
        return model, hparams, checkpoint_path
    else:
        return model, hparams


@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def main(config: OmegaConf) -> None:
    try:
        print('*' * 32)
        print('CONFIGURATION')
        print(OmegaConf.to_yaml(config))
        print('*' * 32, '\n\n')

        print(f'Cuda available: {torch.cuda.is_available()}')

        trainer = create_trainer(config)


        preload = config.dataset.get('_name_') in registry.preloadable_datasets
        if preload:
            dataset = instantiate(registry.dataset, config.dataset, preload=preload)
        else:
            dataset = instantiate(registry.dataset, config.dataset)

        if isinstance(dataset, SeisbenchDataLit):
            print('\nInitializing Seisbench Loaders\n')
            train_loader = DataLoader(
                dataset.dataset_train,
                shuffle=True,
                worker_init_fn=worker_seeding,
                **config.loader
            )
            val_loader = DataLoader(
                dataset.dataset_val,
                shuffle=False,
                worker_init_fn=worker_seeding,
                **config.loader
            )
        else:
            print('\nInitializing Standard Loaders\n')
            train_loader = DataLoader(
                dataset.dataset_train,
                shuffle=True,
                **config.loader
            )
            val_loader = DataLoader(
                dataset.dataset_val,
                shuffle=False,
                **config.loader
            )

        d_data = dataset.d_data

        if config.train.get("ckpt_path", None) is not None:
            print(f'\nLoading checkpoint from {config.train.ckpt_path}\n')
            model, hparams, ckpt_path = load_checkpoint(config.train.ckpt_path, return_path=True)
        else:
            model = SimpleSeqModel(config, d_data=d_data)

        summary = ModelSummary(model, max_depth=1)
        print('\n', '*' * 32, '\n')
        print('SUMMARY')
        print(summary)

        print('\n', '*' * 32, '\n')
        print('ENCODER')
        print(model.encoder)
        print('\n', '*' * 32, '\n')

        print('DECODER')
        print(model.decoder)
        print('*' * 32, '\n\n')

        trainer.fit(model, train_loader, val_loader)

        print('\n', '*' * 32, '\n')
        print('DONE')
        print('\n', '*' * 32, '\n')
    except Exception as e:
        traceback.print_exc()
    finally:
        # FIXME: workaround to prevent wandb from blocking the termination of runs on sciCORE slurm
        def aux(pid, timeout=60):
            time.sleep(timeout)
            print(f"Program did not terminate successfully, killing process tree")
            parent = psutil.Process(pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()

        shutdown_cleanup_thread = threading.Thread(target=aux, args=(os.getpid(), 60), daemon=True)
        shutdown_cleanup_thread.start()


if __name__ == '__main__':
    main()
