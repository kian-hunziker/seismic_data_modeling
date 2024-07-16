import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataloaders.MNISTdataloader import MNISTdataset
from models.simple_test_models import ConvNet
from tasks.encoders import DummyEncoder
from tasks.decoders import DummyDecoder
from tasks.task import task_registry

from utils.config_utils import instantiate

from omegaconf import DictConfig, OmegaConf


class LightningSequenceModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        # initialize dataset:
        if self.hparams.dataset._name_ == "mnist":
            self.dataset = MNISTdataset(data_dir='dataloaders/data', **self.hparams.dataset)
        else:
            print(f"Unknown dataset name: {self.hparams.dataset._name_}")
            self.dataset = None
        self.setup()

    def setup(self, stage=None):
        self.encoder = DummyEncoder()
        self.decoder = DummyDecoder()

        if self.hparams.model._name_ == "conv_net":
            self.model = ConvNet(self.hparams.model.in_channels, self.hparams.model.img_size)

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
        optimizer = hydra.utils.instantiate(args, params)
        # optimizer = hydra.utils.instantiate(self.hparams.optimizer)
        print(f"Optimizer: {optimizer}")
        return optimizer

    def train_dataloader(self):
        return self.dataset.train_dataloader(**self.hparams.loader)

    def val_dataloader(self):
        return self.dataset.val_dataloader(**self.hparams.loader)

    def test_dataloader(self):
        return self.dataset.test_dataloader(**self.hparams.loader)


def create_trainer(config):
    trainer = pl.Trainer(**config.trainer)
    return trainer


@hydra.main(version_base=None, config_path="configs", config_name="config.yaml")
def main(config: OmegaConf) -> None:
    print('Hello world')
    print(OmegaConf.to_yaml(config))

    trainer = create_trainer(config)
    model = LightningSequenceModel(config)

    trainer.fit(model)

    print(model)


if __name__ == '__main__':
    main()
