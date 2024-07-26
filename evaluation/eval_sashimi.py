import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from models.sashimi.sashimi_standalone import Sashimi
from evaluation.eval_utils import load_checkpoint, get_pipeline_components
import matplotlib.pyplot as plt


def sash_generate_with_context(
        encoder: nn.Module,
        decoder: nn.Module,
        sashimi: Sashimi,
        context: torch.Tensor | np.ndarray | list,
        seq_len: int,
        quantized: bool = False,
        device: str | torch.device = 'cpu'
) -> tuple[torch.Tensor, torch.Tensor]:
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
    :param device: device (e.g. 'cpu' or 'cuda', 'mps' does not work)
    :return: context prediction, auto-regressive sequence
    """

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
    sashimi.setup_rnn()
    encoder.eval()
    decoder.eval()

    context = context.to(device)
    if quantized:
        context = context.long()

    state = sashimi.default_state(device=device)
    context_len = context.shape[1]
    context_output = []
    prediction_output = []

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
                y = torch.argmax(y, dim=-1).unsqueeze(0)
            context_output.append(y.detach().cpu())
            pbar.update()
        pbar.close()

        # Auto-regressive generation. The output of the model is used as the next input.
        pbar = tqdm(total=seq_len)
        pbar.set_description('Auto-regressive generation')
        for i in range(seq_len):
            y = encoder(y)
            y, state = sashimi.step(y, state)
            y = decoder(y, state)
            if quantized:
                y = torch.argmax(y, dim=-1).unsqueeze(0)
            prediction_output.append(y.detach().cpu())
            pbar.update()
        pbar.close()

    context_output = torch.stack(context_output, dim=1)
    prediction_output = torch.stack(prediction_output, dim=1)
    return context_output.cpu(), prediction_output.cpu()


def sash_generate_without_context(
        encoder: nn.Module,
        decoder: nn.Module,
        sashimi: Sashimi,
        seq_len: int,
        quantized: bool = False,
        device: str | torch.device = 'cpu'
):
    sashimi = sashimi.to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    sashimi.eval()
    sashimi.setup_rnn()
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
        auto_reg_prediction = auto_reg_prediction.detach().cpu().squeeze().numpy()

    provided_context = full_context[:len_context]
    len_prediction = len(auto_reg_prediction)

    # x axis
    x = np.arange(len(provided_context) + len(auto_reg_prediction))

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
    ax.plot(
        x[len(provided_context):],
        auto_reg_prediction,
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


if __name__ == '__main__':
    sashimi_eval_test()
