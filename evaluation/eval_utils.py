import os
import yaml
import torch
import re
import os
import glob
import numpy as np
from pytorch_lightning.utilities.model_summary import ModelSummary
from train import LightningSequenceModel
from dataloaders.data_utils.costa_rica_utils import get_metadata
from models.sashimi.sashimi_standalone import Sashimi


def _extract_step_number(filename):
    match = re.search(r'step=(\d+)\.ckpt$', filename)
    if match:
        return int(match.group(1))
    return None


def load_checkpoint(checkpoint_path: str, location: str = 'cpu', return_path: bool = False,) -> tuple[LightningSequenceModel, dict]:
    """
    Load checkpoint and hparams.yaml from specified path. Model is loaded to cpu.
    If no checkpoint is specified, the folder is searched for checkpoints and the one with the highest
    step number is returned.
    :param return_path: if true, the path to the checkpoint will be returned
    :param location: device to map the checkpoint to (e.g. cuda or cpu). Defaults to 'cpu'
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

    model = LightningSequenceModel.load_from_checkpoint(checkpoint_path, map_location=location)
    if return_path:
        return model, hparams, checkpoint_path
    else:
        return model, hparams


def get_pipeline_components(pl_module: LightningSequenceModel):
    """
    Extract encoder, decoder and model backbone from LightningSequenceModel.
    The components are put in eval mode.
    :param pl_module: LightningSequenceModel (e.g. loaded from checkpoint)
    :return: Encoder, Decoder, Model
    """
    encoder = pl_module.encoder.eval()
    decoder = pl_module.decoder.eval()
    model = pl_module.model.eval()

    #if isinstance(model, Sashimi):
        #model.setup_rnn()

    return encoder, decoder, model


def print_hparams(hparams: dict):
    print(yaml.dump(hparams))


def get_model_summary(model: LightningSequenceModel, max_depth=1):
    summary = ModelSummary(model, max_depth=max_depth)
    return summary


def get_sorted_file_list(dir_path: str):
    file_paths = glob.glob(os.path.join(dir_path, '*.pt'))
    year_day_list = []
    for file in file_paths:
        name = file.split('/')[-1]
        meta = get_metadata(name)
        year_day_list.append((meta['year'], meta['day'], file))

    # sort list by year and day
    year_day_list = sorted(year_day_list, key=lambda x: x[0] + x[1])

    file_paths = [f for y, d, f in year_day_list]
    return file_paths