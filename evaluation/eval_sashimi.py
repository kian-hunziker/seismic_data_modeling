import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import decimate
from dataloaders.data_utils.signal_encoding import quantize_encode, normalize_11_torch
import numpy as np
from tqdm import tqdm

from evaluation.multiclass_sampling import greedy_prediction, multinomial_prediction, top_k_prediction
from evaluation.multiclass_sampling import top_k_top_p_filtering
from models.sashimi.sashimi_standalone import Sashimi
from evaluation.eval_utils import load_checkpoint, get_pipeline_components
from dataloaders.data_utils.costa_rica_utils import find_data_min_and_max
import matplotlib.pyplot as plt


def moving_average(signal: torch.Tensor | np.ndarray, window_size: int = 10) -> torch.Tensor:
    """
    Calculate moving average over signal.
    :param signal: Signal to average. [1, signal_length] or [signal_length]
    :param window_size: Length of moving average. Defaults to 10
    :return: Averaged signal with the same length as the input
    """
    if isinstance(signal, np.ndarray):
        signal = torch.from_numpy(signal)

    if signal.dim() == 1:
        signal = signal.unsqueeze(0)
    signal = signal.float()

    # Create the convolution kernel with equal weights
    kernel = torch.ones(window_size) / window_size

    # Reshape the kernel to match the shape required for F.conv1d
    kernel = kernel.view(1, 1, -1)

    # Apply padding to the tensor to maintain the output size
    padding = window_size // 2
    padded_tensor = F.pad(signal, (padding, padding), mode='reflect')

    # Apply the convolution
    moving_avg = F.conv1d(padded_tensor.unsqueeze(0), kernel).squeeze(0)

    return moving_avg


def prepare_data(data: torch.Tensor, downsample: int = 100, bits: int = 8, d_max: int = None):
    if d_max is None:
        data_max = torch.max(data)
        data_min = torch.min(data)
        data_max = np.sqrt(max(abs(data_max), abs(data_min)))
        data_min = -data_max
    else:
        data_max = d_max
        data_min = -d_max

    data = decimate(data, q=downsample)
    data = torch.from_numpy(data.copy()).float()
    data = torch.sqrt(torch.abs(data)) * torch.sign(data)
    data = normalize_11_torch(data, d_min=data_min, d_max=data_max)
    if bits > 0:
        data = quantize_encode(data, bits=bits)
    return data


def sash_condition(
        encoder: nn.Module,
        decoder: nn.Module,
        sashimi: Sashimi,
        context: torch.Tensor | np.ndarray | list,
        quantized: bool = False,
        rnn_mode: str = 'diagonal',
        device: str | torch.device = 'cpu'
):
    assert isinstance(sashimi, Sashimi)

    if isinstance(context, np.ndarray):
        context = torch.from_numpy(context).type(torch.float32)
    elif isinstance(context, list):
        context = torch.Tensor(context).type(torch.float32)
    elif isinstance(context, torch.Tensor):
        context = context.type(torch.float32)

    if context.dim() == 1:
        context = context.unsqueeze(0).unsqueeze(-1)

    sashimi = sashimi.to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    sashimi.eval()
    try:
        sashimi.setup_rnn(mode=rnn_mode)
    except:
        print(f'Could not setup RNN with mode {rnn_mode}')
        sashimi.setup_rnn()
    encoder.eval()
    decoder.eval()

    context = context.to(device)
    if quantized:
        context = context.long()

    state = sashimi.default_state(device=device)
    context_len = context.shape[1]
    context_output = []

    # inference in recurrent mode
    with torch.no_grad():
        # 'condition' the model by passing the context
        # teacher forcing approach, the output is saved for plotting but not fed back to model.
        pbar = tqdm(total=context_len)
        pbar.set_description('Processing context')
        for i in range(context_len):
            y = encoder(context[:, i])
            y, state = sashimi.step(y, state)
            y = decoder(y, state)
            if quantized:
                y = greedy_prediction(y)
            context_output.append(y.detach().cpu())
            pbar.update()
        pbar.close()
    return torch.stack(context_output, dim=1).cpu(), state


def auto_regressive_generation(
        encoder: nn.Module,
        decoder: nn.Module,
        sashimi: Sashimi,
        initial_state: list,
        initial_input: torch.Tensor,
        seq_len: int,
        num_predictions: int,
        quantized: bool = False,
        temperature: float = 1.0,
        sampling_mode: str = 'prob',
        k: int = 10,
        p: float = 0.0,
        rnn_mode: str = 'dense',
        device: str | torch.device = 'cpu'
):
    assert isinstance(sashimi, Sashimi)

    sashimi = sashimi.to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    sashimi.eval()
    try:
        sashimi.setup_rnn(mode=rnn_mode)
    except:
        print(f'Could not setup RNN with mode {rnn_mode}')
        sashimi.setup_rnn()
    encoder.eval()
    decoder.eval()

    initial_state_copy = copy.deepcopy(initial_state)
    if quantized:
        initial_input = initial_input.long()
    initial_input_copy = initial_input.clone()

    all_predictions = []

    # inference in recurrent mode
    with torch.no_grad():
        # Auto-regressive generation. The output of the model is used as the next input.
        if sampling_mode == 'greedy':
            num_predictions = 1

        for i in range(num_predictions):
            prediction_output = []
            y = initial_input_copy.clone().to(device)
            state = copy.deepcopy(initial_state_copy)

            pbar = tqdm(total=seq_len)
            pbar.set_description(f'Auto-regressive generation {i + 1} / {num_predictions}')
            for _ in range(seq_len):
                y = encoder(y)
                y, state = sashimi.step(y, state)
                y = decoder(y, state)
                if quantized:
                    if sampling_mode == 'greedy' or i == 0:
                        y = greedy_prediction(y)
                    elif sampling_mode == 'prob':
                        y = multinomial_prediction(y, temperature=temperature)
                    elif sampling_mode == 'top_k':
                        y = top_k_top_p_filtering(y, top_k=k, temperature=temperature)
                    elif sampling_mode == 'top_p':
                        y = top_k_top_p_filtering(y, top_p=p, temperature=temperature)
                    else:
                        print(f'Unknown sampling mode {sampling_mode}')
                prediction_output.append(y.detach().cpu())
                pbar.update()
            all_predictions.append(torch.stack(prediction_output, dim=1).cpu())
            pbar.close()

    return all_predictions


def sash_generate_with_context(
        encoder: nn.Module,
        decoder: nn.Module,
        sashimi: Sashimi,
        context: torch.Tensor | np.ndarray | list,
        seq_len: int,
        quantized: bool = False,
        rnn_mode: str = 'dense',
        sampling_strategy: str = 'greedy',
        num_predictions: int = 1,
        k: int = 10,
        p: float = 0.0,
        temperature: float = 1.0,
        device: str | torch.device = 'cpu'
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Inference using a Sashimi model.
    The context is used to condition the model. After that, the model generates a sequence of len seq_len
    in auto-regressive mode. Returns the predictions for the context and the auto-regressively generated sequence

    :param encoder: Encoder
    :param decoder: Decoder
    :param sashimi: Sashimi in rnn mode
    :param context: context for the generation
    :param seq_len: length of the sequence to generate after the context
    :param quantized: whether the model works with quantized data in a multiclass setting
    :param rnn_mode: S4 recurrence mode, e.g. 'diagonal', 'dense', 'linear'. 'diagonal' should be the fastest
            but might be unstable.
    :param sampling_strategy: sampling strategy for quantized data. One of ['greedy', 'prob', 'top_k', 'top_p']
    :param num_predictions: number of auto-regressive predictions.
    :param k: k for top_k sampling
    :param p: p for top_p sampling
    :param temperature: temperature for softmax sampling
    :param device: device (e.g. 'cpu' or 'cuda', 'mps' does not work)
    :return: context prediction, auto-regressive sequences. The first element of the predictions is greedy
    """

    # condition the model
    cp, conditioned_state = sash_condition(
        encoder=encoder,
        decoder=decoder,
        sashimi=sashimi,
        context=context,
        quantized=quantized,
        rnn_mode=rnn_mode,
        device=device
    )
    # auto-regressive generation
    predictions = auto_regressive_generation(
        encoder=encoder,
        decoder=decoder,
        sashimi=sashimi,
        initial_state=copy.deepcopy(conditioned_state),
        initial_input=cp[:, -1, :],
        seq_len=seq_len,
        quantized=quantized,
        num_predictions=num_predictions,
        temperature=temperature,
        sampling_mode=sampling_strategy,
        k=k,
        p=p,
        rnn_mode='dense',
        device=device
    )
    return cp.cpu(), predictions


def sash_generate_without_context(
        encoder: nn.Module,
        decoder: nn.Module,
        sashimi: Sashimi,
        seq_len: int,
        quantized: bool = False,
        rnn_mode: str = 'diagonal',
        device: str | torch.device = 'cpu'
):
    sashimi = sashimi.to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    sashimi.eval()
    sashimi.setup_rnn(mode=rnn_mode)
    encoder.eval()
    decoder.eval()

    # default state
    state = sashimi.default_state()

    # initial input
    if quantized:
        y = torch.zeros(1, 1).long()
    else:
        y = torch.zeros(1, 1)
    ys = []

    # auto-regressive generation
    for _ in tqdm(range(seq_len)):
        y = encoder(y)
        y, state = sashimi.step(y, state)
        y = decoder(y, state)
        if quantized:
            y = torch.argmax(y, dim=-1).unsqueeze(0)
        ys.append(y.detach().cpu())

    out = torch.stack(ys, dim=1)  # y.shape == x.shape
    return out


def plot_predictions(
        full_context: torch.Tensor | np.ndarray | list,
        predicted_context: torch.Tensor | np.ndarray | list,
        auto_reg_prediction: torch.Tensor | np.ndarray | list,
        len_context: int,
        title: str = '',
        fig_size: tuple = (10, 5),
        line_width: int = 3,
        save_path: str = None,
        show: bool = True,
):
    """
    Plot context and predictions from auto-regressive generation
    :param full_context: tensor, np.ndarray or list containing the context and ground truth
            len(full_context) >= len(predicted_context) + len(prediction)
    :param predicted_context: predictions form conditioning the model
    :param auto_reg_prediction: predictions from auto-regressive generation
    :param len_context: length of the context used to condition the model
    :param title: title of plot
    :param fig_size: size of figure
    :param line_width: width of lines in plot, default 3
    :param save_path: if this is not none, the plot will be saved in save_path with the given title
    :param show: if true, plot.show() is called
    """
    if isinstance(full_context, torch.Tensor):
        full_context = full_context.detach().cpu().squeeze().numpy()
    if isinstance(predicted_context, torch.Tensor):
        predicted_context = predicted_context.detach().cpu().squeeze().numpy()
    if isinstance(auto_reg_prediction, torch.Tensor):
        auto_reg_prediction = [auto_reg_prediction]

    provided_context = full_context[:len_context]
    len_prediction = len(auto_reg_prediction[0].squeeze())

    # x axis
    x = np.arange(len_context + len_prediction)

    # plotting
    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(
        x[:len_context + len_prediction],
        full_context[1:len_context + len_prediction + 1],
        label='actual context',
        linewidth=line_width,
    )
    ax.plot(
        x[:len(provided_context)],
        predicted_context,
        label='predicted context',
        linewidth=line_width,
    )
    for i in range(len(auto_reg_prediction) - 1):
        ax.plot(
            x[len(provided_context):],
            auto_reg_prediction[i + 1].detach().cpu().squeeze().numpy(),
            color='green',
            alpha=0.2,
            linewidth=line_width,
        )
    ax.plot(
        x[len(provided_context):],
        auto_reg_prediction[0].detach().cpu().squeeze().numpy(),
        color='green',
        label='auto-regressive',
        linewidth=line_width,
    )
    plt.xlabel('sample')
    plt.legend()
    plt.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, '{}.png'.format(title)))
    if show:
        plt.show()


def plot_multiple_predictions(
        full_context: torch.Tensor | np.ndarray | list,
        predicted_context: torch.Tensor | np.ndarray | list,
        auto_reg_prediction: list,
        len_context: int,
        title: str = '',
        fig_size: tuple = (10, 5),
        line_width: int = 3,
        save_path: str = None,
        show: bool = True,
):
    if isinstance(full_context, torch.Tensor):
        full_context = full_context.detach().cpu().squeeze().numpy()
    if isinstance(predicted_context, torch.Tensor):
        predicted_context = predicted_context.detach().cpu().squeeze().numpy()

    greedy_pred = auto_reg_prediction[0].squeeze().numpy()
    preds_np = [p.detach().cpu().squeeze().unsqueeze(0).numpy() for p in auto_reg_prediction]
    len_prediction = len(greedy_pred)

    x = np.arange(len_context + len_prediction)

    fig, ax = plt.subplots(figsize=fig_size)
    # plot context
    ax.plot(
        x[:len_context + len_prediction],
        full_context[1: len_context + len_prediction + 1],
        label='Context',
        linewidth=line_width,
    )
    # plot next sample predictions for context
    ax.plot(
        x[:len_context],
        predicted_context,
        label='next sample predictions',
        linewidth=line_width,
        color='orange'
    )
    # plot mean of multiple autoregressive predictions
    ax.plot(
        x[len_context:],
        np.mean(preds_np[1:], axis=0)[0],
        label='mean prediction',
        linewidth=line_width,
        color='green'
    )
    # plot greedy prediction
    ax.plot(
        x[len_context:],
        greedy_pred,
        label='greedy prediction',
        linewidth=line_width,
        color='red'
    )
    # plot confidence interval
    ax.fill_between(
        x[len_context:],
        np.percentile(preds_np[1:], 5, axis=0)[0],
        np.percentile(preds_np[1:], 95, axis=0)[0],
        alpha=0.5,
        label='5% and 95% confidence interval',
        color='green'
    )
    plt.legend()
    plt.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, '{}.png'.format(title)))
        plt.close()
    if show:
        plt.show()


def plot_batched_multiple_predictions(
        full_context: torch.Tensor,
        predicted_context: torch.Tensor | np.ndarray | list,
        auto_reg_prediction: list,
        len_context: int,
        title: str = '',
        fig_size: tuple = (10, 5),
        line_width: int = 3,
        save_path: str = None,
        show: bool = True,
):
    batch_size = full_context.shape[0]
    for i in range(batch_size):
        plot_multiple_predictions(
            full_context=full_context[i],
            predicted_context=predicted_context[i],
            auto_reg_prediction=[p[i] for p in auto_reg_prediction],
            len_context=len_context,
            title=title + f'_{i + 1:02}_of_{batch_size}',
            fig_size=fig_size,
            line_width=line_width,
            save_path=save_path,
            show=show
        )


def sashimi_eval_test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    path = '../wandb_logs/MA/2024-07-22__15_29_38/checkpoints/epoch=219-step=15620.ckpt'
    pl_module, hparams = load_checkpoint(checkpoint_path=path)
    encoder, decoder, sash = get_pipeline_components(pl_module=pl_module)

    t = np.linspace(0, 2, 32000)
    context_len = 256
    prediction_len = 200
    f = 200
    a = 0.8

    context_np = a * np.sin(2 * np.pi * f * t)[:context_len]
    context_list = [a * np.sin(2 * np.pi * f * t_i) for t_i in t[:context_len]]
    context_torch = torch.from_numpy(context_np)[:context_len]

    cp_np, pred_np = sash_generate_with_context(encoder, decoder, sash, context_np, prediction_len, device)
    cp_list, pred_list = sash_generate_with_context(encoder, decoder, sash, context_list, prediction_len)
    cp_torch, pred_torch = sash_generate_with_context(encoder, decoder, sash, context_torch, prediction_len)

    full_context = torch.from_numpy(a * np.sin(2 * np.pi * f * t))

    plot_predictions(full_context.numpy(), cp_np, pred_np, context_len, title='prediction np')
    plot_predictions(full_context.tolist(), cp_list, pred_list, context_len, title='prediction list')
    plot_predictions(full_context, cp_torch, pred_torch, context_len, title='prediction torch')

    print('done!')


def top_k_test():
    pred = torch.tensor([100, -2.5, 3, 11.5, 99, 0.1])
    for i in range(10):
        out = top_k_prediction(pred, k=3)
        print(out)


if __name__ == '__main__':
    # sashimi_eval_test()
    top_k_test()
