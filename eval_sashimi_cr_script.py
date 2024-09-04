import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import decimate
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

from evaluation.eval_utils import get_sorted_file_list
from models.sashimi.sashimi_standalone import Sashimi
import evaluation.beam_search
import evaluation.eval_utils
import evaluation.eval_sashimi as e_sash

try:
    from models.sashimi.sashimi_mamba import MambaSashimi
    import evaluation.eval_mamba_sashimi as e_mamba
    from models.mamba_inference_params import InferenceParams
except:
    print(f'No Mamba installation found')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval_model(
        model_path: str,
        num_pred: int = 10,
        num_days_to_predict: int = 3,
        sampling_mode: str = 'prob',
        k: int = 10,
        p: float = 0.0,
):
    print('*' * 32)
    print('Evaluating')
    print('*' * 32, '\n\n')
    print(f'Using device: {device}\n\n')

    # load checkpoint and print hparams
    pl_module, hparams = evaluation.eval_utils.load_checkpoint(model_path, location=device)
    evaluation.eval_utils.print_hparams(hparams)

    # create directory for plots
    save_dir = os.path.join(model_path, 'eval_plots')

    # Check if the 'plots' directory exists
    if not os.path.exists(save_dir):
        # If it does not exist, create it
        os.makedirs(save_dir)
        print(f"Directory 'eval_plots' created at {save_dir}")
    else:
        print(f"Directory 'eval_plots' already exists at {save_dir}")

    # check if dataset has quantized parameter. If the parameter does not exist, we assume the data was quantized
    try:
        quantized = hparams['dataset']['quantize']
    except:
        quantized = True
    print(f'\nData is quantized: {quantized}')

    try:
        bits = hparams['dataset']['bits']
    except:
        bits = 8

    downsample = hparams['dataset']['downsample']
    print(f'downsample: {downsample}')

    # load data
    data_files_sorted = get_sorted_file_list('dataloaders/data/costa_rica/small_subset')

    num_training_days = 50
    offset = 0
    train_data = []
    for i in range(num_training_days):
        train_data.append(torch.load(data_files_sorted[i + offset]))
    train_data = torch.cat(train_data)
    test_data = []
    for i in range(3):
        test_data.append(torch.load(data_files_sorted[-3 + i]))
    test_data = torch.cat(test_data)

    print(f'Train data shape: {train_data.shape}')
    print(f'Test data shape: {test_data.shape}')

    if quantized:
        train_data_encoded = evaluation.eval_sashimi.prepare_data(train_data, downsample=downsample, bits=bits)
        test_data_encoded = evaluation.eval_sashimi.prepare_data(test_data, downsample=downsample, bits=bits)
    else:
        normalize_const = hparams['dataset']['normalize_const']
        train_data_encoded = decimate(train_data / float(normalize_const), q=downsample)
        test_data_encoded = decimate(test_data / float(normalize_const), q=downsample)
    # train_data_downsampled = decimate(train_data, q=downsample)
    # test_data_downsampled = decimate(test_data, q=downsample)
    print(f'Train data encoded shape: {train_data_encoded.shape}')
    print(f'Test data encoded shape: {test_data_encoded.shape}')

    # plot train and test data
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(train_data_encoded.numpy())
    plt.suptitle(f'Train data, num days: {num_training_days}, offset: {offset}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '{}.png'.format('Train_data')))
    plt.close()

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(test_data_encoded.numpy())
    plt.suptitle(f'Test data, num days: {3}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '{}.png'.format('Test_data')))
    plt.close()

    encoder, decoder, model = evaluation.eval_utils.get_pipeline_components(pl_module)

    is_mamba = not isinstance(model, Sashimi)
    print(f'Model is Mamba: {is_mamba}')

    num_samples_per_day = int(8_640_000 / downsample)

    conditioning_day_numbers = [1, 2, 4, 8, 12]
    batch_size = 32
    eval_length_days = num_days_to_predict
    num_predictions = num_pred
    sampling_mode = sampling_mode  # one of ['greedy', 'prob', 'top_k', 'top_p']

    # main evaluation loop
    for conditioning_day in conditioning_day_numbers:
        print(f'Conditioning on data of {conditioning_day} days. '
              f'Total num samples: {conditioning_day * num_samples_per_day}')

        context_len = conditioning_day * num_samples_per_day
        prediction_len = eval_length_days * num_samples_per_day

        if is_mamba:
            assert isinstance(model, MambaSashimi)

            # condition the model
            cp, conditioned_state, conditioned_inference_params = e_mamba.condition(
                encoder=encoder,
                decoder=decoder,
                sashimi=model,
                context=torch.stack(
                    [train_data_encoded[i * num_samples_per_day:i * num_samples_per_day + context_len] for i in
                     range(batch_size)]).squeeze(),
                seq_len=prediction_len,
                quantized=quantized,
            )
            # auto-regressive generation
            predictions = e_mamba.auto_regressive_generation(
                encoder=encoder,
                decoder=decoder,
                sashimi=model,
                state=copy.deepcopy(conditioned_state),
                inference_params=copy.deepcopy(conditioned_inference_params),
                initial_input=cp[:, -1, :],
                seq_len=prediction_len,
                quantized=quantized,
                num_predictions=num_predictions,
                temperature=1.0,
                sampling_mode=sampling_mode,
                k=k,
                p=p,
            )
        else:
            # not mamba, the model is a standard sashimi
            assert isinstance(model, Sashimi)

            # condition the model
            cp, conditioned_state = e_sash.sash_condition(
                encoder=encoder,
                decoder=decoder,
                sashimi=model,
                context=torch.stack(
                    [train_data_encoded[i * num_samples_per_day:i * num_samples_per_day + context_len] for i in
                     range(batch_size)]).squeeze(),
                quantized=quantized,
                rnn_mode='Dense',
                device=device
            )
            # auto-regressive generation
            predictions = e_sash.auto_regressive_generation(
                encoder=encoder,
                decoder=decoder,
                sashimi=model,
                initial_state=copy.deepcopy(conditioned_state),
                initial_input=cp[:, -1, :],
                seq_len=prediction_len,
                quantized=quantized,
                num_predictions=num_predictions,
                temperature=1.0,
                sampling_mode=sampling_mode,
                k=k,
                p=p,
                rnn_mode='dense',
                device=device
            )

        # plot and save predictions
        print('saving plots...')
        evaluation.eval_sashimi.plot_batched_multiple_predictions(
            full_context=torch.stack(
                [train_data_encoded[
                 i * num_samples_per_day:i * num_samples_per_day + context_len + prediction_len + 1]
                 for i in range(batch_size)]
            ).squeeze().cpu(),
            predicted_context=cp.cpu(),
            auto_reg_prediction=predictions,
            len_context=context_len,
            title=f'{sampling_mode}_sampling_{conditioning_day:02}_days_context',
            fig_size=(20, 5),
            line_width=2,
            save_path=save_dir,
            show=False,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, required=True, help='path to checkpoint')
    parser.add_argument('--sampling_mode', type=str, default='prob', choices=['greedy', 'prob', 'top_k', 'top_p'], help='sampling mode')
    parser.add_argument('--k', type=int, default=10, help='k for top_k sampling')
    parser.add_argument('--p', type=float, default=0.0, help='p for top_p sampling')
    parser.add_argument('--num_pred', type=int, default=10, help='number of predictions')
    parser.add_argument('--pred_days', type=int, default=3, help='number of days to predict')
    args = parser.parse_args()

    eval_model(
        model_path=args.path,
        num_pred=args.num_pred,
        num_days_to_predict=args.pred_days,
        sampling_mode=args.sampling_mode,
        k=args.k,
        p=args.p
    )
