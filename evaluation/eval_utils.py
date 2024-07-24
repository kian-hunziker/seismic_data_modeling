import os
import yaml
import torch
import numpy as np
from pytorch_lightning.utilities.model_summary import ModelSummary
from train import LightningSequenceModel
from models.sashimi.sashimi_standalone import Sashimi


def load_checkpoint(checkpoint_path: str) -> tuple[LightningSequenceModel, dict]:
    """
    Load checkpoint and hparams.yaml from specified path. Model is loaded to cpu
    :param checkpoint_path: path to checkpoint file. The hparams file is extracted automatically
    :return: LightningSequenceModel, hparams
    """
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

    model = LightningSequenceModel.load_from_checkpoint(checkpoint_path, map_location='cpu')

    return model, hparams


def get_pipeline_components(pl_module: LightningSequenceModel):
    """
    Extract encoder, decoder and model backbone from LightningSequenceModel.
    The components are put in eval mode. If the model is a Sashimi, the RNN is set up.
    :param pl_module: LightningSequenceModel (e.g. loaded from checkpoint)
    :return: Encoder, Decoder, Model
    """
    encoder = pl_module.encoder.eval()
    decoder = pl_module.decoder.eval()
    model = pl_module.model.eval()

    if isinstance(model, Sashimi):
        model.setup_rnn()

    return encoder, decoder, model


def print_hparams(hparams: dict):
    print(yaml.dump(hparams))


def get_model_summary(model: LightningSequenceModel, max_depth=1):
    summary = ModelSummary(model, max_depth=max_depth)
    return summary
