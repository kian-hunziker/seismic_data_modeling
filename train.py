import datetime
import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary, LayerSummary
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataloaders.MNISTdataloader import MNISTdataset
from dataloaders.simple_waveform import SineWaveLightningDataset
from models.simple_test_models import ConvNet
from tasks.encoders import instantiate_encoder
from tasks.decoders import instantiate_decoder
from tasks.task import task_registry

from utils.config_utils import instantiate
from utils import registry
from omegaconf import DictConfig, OmegaConf


class LightningSequenceModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        # initialize dataset:
        self.dataset = instantiate(registry.dataset, self.hparams.dataset)
        '''
        if self.hparams.dataset._name_ == "mnist":
            self.dataset = MNISTdataset(data_dir='dataloaders/data', **self.hparams.dataset)
        elif self.hparams.dataset._name_ == "sine":
            self.dataset = SineWaveLightningDataset(
                data_dir='dataloaders/data/basic_waveforms/sine_waveform.npy',
                **self.hparams.dataset,
            )
        else:
            print(f"Unknown dataset name: {self.hparams.dataset._name_}")
            self.dataset = None
        '''
        self.setup()

    def setup(self, stage=None):
        self.model = instantiate(registry.model, self.hparams.model)

        self.encoder = instantiate_encoder(self.hparams.encoder, self.dataset, self.model)
        '''LayerNormClassEncoder(
            in_features=1,
            out_features=self.hparams.model.d_model,
            num_classes=256
        )'''
        self.decoder = instantiate_decoder(self.hparams.decoder, self.dataset, self.model)

        #if self.hparams.model._name_ == "conv_net":
        #    self.model = ConvNet(self.hparams.model.in_channels, self.hparams.model.img_size)

        self.task = instantiate(task_registry, self.hparams.task, dataset=self.dataset, model=self.model)
        self.criterion = self.task.loss
        self.loss_val = self.task.loss_val
        self.metrics = self.task.metrics

        self._initialize_state()

    def _initialize_state(self):
        self._state = None

    def _on_epoch_start(self):
        self._initialize_state()

    def forward(self, batch, batch_idx):
        x, y = batch
        x = self.encoder(x)
        x, state = self.model(x, state=self._state)
        self._state = state
        x = self.decoder(x, state=state)
        return x, y

    def step(self, x_t):
        x_t = self.encoder(x_t)
        x_t, state = self.model.step(x_t, state=self._state)
        self._state = state
        x_t = self.decoder.step(x_t, state=self._state)
        return x_t

    def _step_with_metrics(self, batch, batch_idx, prefix='train'):
        x, y = self.forward(batch, batch_idx)

        if 'context_len' in self.hparams.task:
            args = {'context_len': self.hparams.task.context_len}
        else:
            args = {}

        if prefix == 'train':
            loss = self.criterion(x, y, **args)
        else:
            loss = self.loss_val(x, y, **args)

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

    def on_train_epoch_start(self) -> None:
        self._on_epoch_start()
        
    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        
    def on_validation_epoch_start(self) -> None:
        self._on_epoch_start()
    
    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()

    def on_test_epoch_start(self) -> None:
        self._on_epoch_start()
        
    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()

    def training_step(self, batch, batch_idx):
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

    def validation_step(self, batch, batch_idx):
        loss = self._step_with_metrics(batch, batch_idx, prefix='val')
        print(f'Validation Loss: {loss:.4f}')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._step_with_metrics(batch, batch_idx, prefix='test')
        print(f'Test Loss: {loss:.4f}')
        return loss

    def configure_optimizers(self):
        # Set zero weight decay for some params

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

    def train_dataloader(self):
        return self.dataset.train_dataloader(**self.hparams.loader)

    def val_dataloader(self):
        return self.dataset.val_dataloader(**self.hparams.loader)

    def test_dataloader(self):
        return self.dataset.test_dataloader(**self.hparams.loader)


def create_trainer(config):
    current_date = datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S")
    model_name = config.model['_name_']
    experiment_name = config.experiment_name

    # setup logger

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

    # initialize trainer
    trainer = pl.Trainer(logger=loggers, **config.trainer)
    return trainer


@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def main(config: OmegaConf) -> None:
    print('*' * 32)
    print('CONFIGURATION')
    print(OmegaConf.to_yaml(config))
    print('*' * 32, '\n\n')

    trainer = create_trainer(config)
    model = LightningSequenceModel(config)

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

    trainer.fit(model)

    print('\n', '*' * 32, '\n')
    print('DONE')
    print('\n', '*' * 32, '\n')


if __name__ == '__main__':
    main()
