import datetime
import os
import threading
import time
import traceback
import pickle

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

from tasks.encoders import instantiate_encoder, load_encoder_from_file, instantiate_encoder_simple
from tasks.decoders import instantiate_decoder, load_decoder_from_file, instantiate_decoder_simple
from tasks.task import task_registry
from dataloaders.base import SeisbenchDataLit

from torch.utils.data import DataLoader

from utils.config_utils import instantiate
from utils import registry
from utils.optim_utils import print_optim, add_optimizer_hooks
from omegaconf import DictConfig, OmegaConf
from seisbench.util import worker_seeding
import seisbench
import logging
# ignore INFO level logging
seisbench.logger.setLevel(logging.WARNING)


class SimpleSeqModel(pl.LightningModule):
    def __init__(self, config, d_data: int = 3):
        super().__init__()
        self.save_hyperparameters(config)
        self.d_data = d_data

        self.l2_norm = config.train.get('l2', False)
        if config.model.get('pretrained', None) is not None:
            # load pretrained model
            print('\nLoading pretrained model\n')

            # extract checkpoint path
            ckpt_path = config.model.pretrained

            # check if the model should be randomly initialized (for sanity checks)
            rand_init = config.model.get('rand_init', False)
            print(f'model random initialization: {rand_init}')

            # look for updated model parameters
            # for example: updated dropout value for fine-tuning
            if config.get('model_update', None) is not None:
                update_configs = config.model_update

                # load model from checkpoint
                ckpt, _ = load_checkpoint(
                    ckpt_path,
                    updated_model_config=update_configs,
                    d_data=d_data,
                    rand_init=rand_init
                )
            else:
                ckpt, _ = load_checkpoint(
                    checkpoint_path=ckpt_path,
                    d_data=d_data,
                    rand_init=rand_init,
                )

            # extract main model from checkpoint
            self.model = ckpt.model
            # freeze model parameters
            if config.model.get('freeze', None) is not None and config.model.get('freeze', False):
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False

            if config.train.get('only_final_layer', False):
                num_layers_to_train = config.train.get('num_layers', 1)
                self.model.fix_all_but_last_layer(num_layers=num_layers_to_train)

            # save parameters for L2 norm
            if self.l2_norm:
                # TODO: check if there is a better way to clone the model
                self.l2_lambda = config.train.get('l2_lambda', 0.1)
                ref_ckpt, _ = load_checkpoint(config.model.pretrained, d_data=d_data)
                self.reference_model = ref_ckpt.model.eval()
                for param in self.reference_model.parameters():
                    param.requires_grad = False
        else:
            # initialize new model
            self.model = instantiate(registry.model, self.hparams.model)

        try:
            d_model = self.model.d_model
        except:
            print('could not infer d_model from model')
            d_model = 0

        freeze_encoder = config.encoder.get('freeze', False)
        encoder_config = self.hparams.encoder
        if config.encoder.get('freeze', None) is not None:
            encoder_config.pop('freeze')

        if config.encoder.get('pretrained', None) is not None:
            print('\nLoading pretrained encoder\n')
            self.encoder = ckpt.encoder
        else:
            self.encoder = instantiate_encoder_simple(encoder_config, d_data=self.d_data, d_model=d_model)

        if freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        if config.decoder.get('pretrained', None) is not None:
            print('\nLoading pretrained decoder\n')
            self.decoder = ckpt.decoder
            if config.decoder.get('freeze', None) is not None and config.decoder.get('freeze', False):
                self.decoder.eval()
                for param in self.decoder.parameters():
                    param.requires_grad = False
        else:
            self.decoder = instantiate_decoder_simple(self.hparams.decoder, d_data=self.d_data, d_model=d_model)

        self.task = instantiate(task_registry, self.hparams.task)
        self.criterion = self.task.loss
        self.loss_val = self.task.loss_val
        self.metrics = self.task.metrics

    def forward(self, batch, batch_idx=None):
        masked = False
        if isinstance(batch, dict):
            x = batch['X']
            y = batch['y']
            # TODO: clean this. Masking might not be the only time x is a list
            if isinstance(x, list):
                masked = True
                x, mask = x
        else:
            x, y = batch

        # encode
        x = self.encoder(x)

        # forward pass
        x, _ = self.model(x, None)

        # decode
        x = self.decoder(x, None)

        if masked:
            return x[mask], y[mask]

        return x, y

    def _l2_norm(self):
        # compute l2 norm between current model weights and reference (pretrained) model weights
        l2_norm = 0
        for param, ref_param in zip(self.model.parameters(), self.reference_model.parameters()):
            # |param - ref_param|^2 / 2
            l2_norm += (param - ref_param).norm(2)
        return l2_norm

    def _step_with_metrics(self, batch, batch_idx, prefix='train'):
        x, y = self.forward(batch, batch_idx)
        metrics = self.metrics(x, y)

        if prefix == 'train':
            data_loss = self.criterion(x, y)
            if self.l2_norm:
                l2_loss = self._l2_norm()
                metrics['data_loss'] = data_loss
                metrics['l2_norm'] = l2_loss
                loss = data_loss + self.l2_lambda * l2_loss
            else:
                loss = data_loss
        else:
            loss = self.loss_val(x, y)

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
        if 'optimizer_param_grouping' in self.hparams.train:
            add_optimizer_hooks(self.model, **self.hparams.train.optimizer_param_grouping)

        # Normal parameters
        all_params = list(self.parameters())
        params = [p for p in all_params if not hasattr(p, "_optim")]

        args = self.hparams.optimizer.copy()
        with omegaconf.open_dict(args):
            del args['_name_']
        optimizer = instantiate(registry.optimizer, self.hparams.optimizer, params)
        print(f"Optimizer: {optimizer}")

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
        hps = [
            # dict(s) for s in set(frozenset(hp.items()) for hp in hps)
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
            # dict(s) for s in dict.fromkeys(frozenset(hp.items()) for hp in hps)
        ]  # Unique dicts
        print("Hyperparameter groups", hps)
        for hp in hps:
            params = [p for p in all_params if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **self.hparams.optimizer, **hp}
            )

        keys = set([k for hp in hps for k in hp.keys()])  # Special hparams
        print_optim(optimizer, keys)
        print('\n\n')

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


def load_checkpoint(
        checkpoint_path: str,
        location: str = 'cpu',
        return_path: bool = False,
        updated_model_config: OmegaConf = None,
        d_data: int = 3,
        rand_init: bool = False,
) -> tuple[SimpleSeqModel, dict]:
    """
    Load checkpoint and hparams.yaml from specified path. Model is loaded to cpu.
    If no checkpoint is specified, the folder is searched for checkpoints and the one with the highest
    step number is returned.

    :param checkpoint_path: path to checkpoint file. The hparams file is extracted automatically
    :param location: device to load to. Defaults to cpu as Lightning handles the devices
    :param return_path: If true, returns the path to the specific checkpoint
    :param updated_model_config: Config with updated parameters. We initially load the configurations from
    the checkpoint and, if provided, use the updated_model_config to overwrite certain parameters. Mostly
    used for fine-tuning e.g. a way to add dropout
    :param d_data: data dimensionality. Passed to the constructor of the model.
    :param rand_init: If True, the model architecture is determined by the provided checkpoint but the trained
    weights are NOT loaded. Instead, the model is returned as is with default initialization.
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

    print(f'Loading hparams from {checkpoint_path}')
    if hparams is not None:
        name = hparams['experiment_name']
        print(f'Experiment name: {name}')

    # hacky fix to load checkpoint, where the old parameter name complex (instead of is_complex) is used
    if hparams['model']['_name_'] == 'mamba-sashimi':
        if 'complex' in hparams['model'].keys():
            hparams['model']['is_complex'] = hparams['model'].pop('complex')

    # initialize model based on loaded resp. updated configuration
    if updated_model_config is not None:
        # create and update omega config
        full_config = OmegaConf.create(hparams)
        full_config.model.update(updated_model_config)
        model = SimpleSeqModel(full_config, d_data=d_data)
        #model.load_state_dict(torch.load(checkpoint_path, map_location=location)['state_dict'])
    else:
        model = SimpleSeqModel(OmegaConf.create(hparams), d_data=d_data)

    # load state dict or return randomly initialized model
    if not rand_init:
        print(f'Loading state dict from checkpoint')
        model.load_state_dict(torch.load(checkpoint_path, map_location=location)['state_dict'])
        #model = SimpleSeqModel.load_from_checkpoint(checkpoint_path, map_location=location)
    else:
        print(f'Returning randomly initialized model')

    # optionally return full path
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

        if config.train.get('seq_warmup', None) is not None:
            seq_warmup = True
            final_sample_len = config.train.get('final_sample_len', 4096)
            final_batch_size = config.train.get('final_batch_size', 128)
            num_epochs_warmup = config.train.get('num_epochs_warmup', 2)
            min_seq_len = config.train.get('min_seq_len', 256)
        else:
            seq_warmup = False

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

        ############################
        # fit the model
        ############################

        if seq_warmup is False:
            # fit model without sequence length warmup (standard)
            trainer.fit(model, train_loader, val_loader)
        else:
            # sequence length warmup
            # for now, the lr resets for every new batch size. Needs to be fixed, see below
            print('\n', '*' * 32, '\n')
            print('Start Training With Batch Size Warmup')
            print('\n', '*' * 32, '\n')

            # compute batch sizes and sequence lengths
            sample_lengths = []
            batch_sizes = []

            b_size = final_batch_size
            s_len = final_sample_len
            while s_len >= min_seq_len:
                sample_lengths.append(s_len)
                batch_sizes.append(b_size)
                b_size = b_size * 2
                s_len = s_len // 2
            batch_sizes = list(reversed(batch_sizes))
            sample_lengths = list(reversed(sample_lengths))

            # print batch sizes and sequence lengths
            for i in range(len(batch_sizes)):
                print(f'Batch Size: {batch_sizes[i] :3d}, Sample Length: {sample_lengths[i]:6d}')

            # Main training loop
            for i in range(len(batch_sizes)):
                print('\n', '*' * 32, '\n')
                if i == len(batch_sizes) - 1:
                    n_epochs = config.trainer.max_epochs - (i + 1) * num_epochs_warmup
                else:
                    n_epochs = num_epochs_warmup
                print(f'Train for {n_epochs} epochs. Batch_size {batch_sizes[i]}, seq_len: {sample_lengths[i]}')
                print('\n', '*' * 32, '\n')

                dataset.sample_len = sample_lengths[i]
                dataset.setup()

                # TODO: fix dataloader worker init fn. worker_seeding should only be used for seisbench data
                train_loader = DataLoader(
                    dataset.dataset_train,
                    shuffle=True,
                    worker_init_fn=worker_seeding,
                    pin_memory=config.loader.pin_memory,
                    num_workers=config.loader.num_workers,
                    batch_size=batch_sizes[i],
                )
                val_loader = DataLoader(
                    dataset.dataset_val,
                    shuffle=False,
                    worker_init_fn=worker_seeding,
                    pin_memory=config.loader.pin_memory,
                    num_workers=config.loader.num_workers,
                    batch_size=batch_sizes[i],
                )
                # print(next(iter(train_loader))['X'].shape)
                # print('len dataset.train', len(dataset.dataset_train))
                # print('train_loader.batch_size', train_loader.batch_size)

                # TODO: figure out why the lr is reset
                if i > 0:
                    model, _ = load_checkpoint('wandb_logs/MA/' + trainer.logger.version)
                    # print('trainer.logger.version', trainer.logger.version)
                    # trainer = create_trainer(config)
                if i == len(batch_sizes) - 1:
                    trainer.fit_loop.max_epochs = config.trainer.max_epochs  # - (i + 1) * num_epochs_warmup
                else:
                    trainer.fit_loop.max_epochs = (i + 1) * num_epochs_warmup
                # print('trainer.fit_loop.max_epochs', trainer.fit_loop.max_epochs)
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
