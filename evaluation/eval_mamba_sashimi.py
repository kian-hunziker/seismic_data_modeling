import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from tqdm import tqdm
from models.sashimi.sashimi_mamba import MambaSashimi
from models.mamba_inference_params import InferenceParams
from evaluation.multiclass_sampling import greedy_prediction, multinomial_prediction, top_k_prediction, top_k_top_p_filtering

supported_sampling_modes = ['greedy', 'prob', 'top_k', 'top_p']


def free_generation_from_default_state(sash, encoder, decoder, length=1024, num_predictions: int = 10,
                                       temperature: float = 1.0):
    initial_inference_params = InferenceParams(max_seqlen=length + 10, max_batch_size=num_predictions)
    default_state = sash.default_state(device='cuda')

    state = copy.deepcopy(default_state)
    inference_params = copy.deepcopy(initial_inference_params)
    y = torch.zeros(num_predictions, 1).long().to("cuda")
    ys = []
    with torch.no_grad():
        for _ in tqdm(range(length)):
            y = encoder(y)
            y, state = sash.step(y, state=state, inference_params=inference_params)
            y = decoder(y, state)
            y = multinomial_prediction(y, temperature=temperature)
            ys.append(y.detach().cpu())
            inference_params.seqlen_offset += 1

    all_predictions = torch.stack(ys, dim=1)
    return all_predictions


def condition(
        encoder: nn.Module,
        decoder: nn.Module,
        sashimi: MambaSashimi,
        context: torch.Tensor | np.ndarray | list,
        seq_len: int,
        quantized: bool = True,
        device: str | torch.device = 'cuda'
):
    """
    Condition the model with provided context. Start from the default state and returns the greedy next sample
    predictions, Sashimi state list and Mamba inference params
    :param encoder: Encoder
    :param decoder: Decoder
    :param sashimi: Mamba Sashimi Model to condition
    :param context: Context to condition
    :param seq_len: approximate length of sequences to generate to initialize inference params
    :param quantized: Whether the context is quantized
    :param device: must be 'cuda' for Mamba to work
    :return: predicted context, state, inference params
    """
    assert isinstance(sashimi, MambaSashimi)
    assert device == 'cuda'

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
    encoder.eval()
    decoder.eval()

    context = context.to(device)
    if quantized:
        context = context.long()
    else:
        context = context.float()
        if context.dim() == 2:
            context = context.unsqueeze(-1)

    context_len = context.shape[1]
    inference_params = InferenceParams(max_seqlen=seq_len + context_len + 10, max_batch_size=context.shape[0])
    state = sashimi.default_state(device=device)

    context_output = []

    # inference in recurrent mode
    with torch.no_grad():
        # 'condition' the model by passing the context
        # teacher forcing approach, the output is saved for plotting but not fed back to model.
        pbar = tqdm(total=context_len)
        pbar.set_description('Processing context')
        for i in range(context_len):
            y = encoder(context[:, i])
            y, state = sashimi.step(y, state=state, inference_params=inference_params)
            y = decoder(y, state)
            if quantized:
                y = greedy_prediction(y)
            context_output.append(y.detach().cpu())
            inference_params.seqlen_offset += 1
            pbar.update()
        pbar.close()

        context_output = torch.stack(context_output, dim=1)

    return context_output, state, inference_params


def auto_regressive_generation(
        encoder: nn.Module,
        decoder: nn.Module,
        sashimi: MambaSashimi,
        state: list,
        inference_params: InferenceParams,
        initial_input: torch.Tensor,
        seq_len: int,
        quantized: bool = True,
        num_predictions: int = 10,
        temperature: float = 1.0,
        sampling_mode: str = 'prob',
        k: int = 10,
        p: float = 0.0,
        device: str | torch.device = 'cuda'
) -> list[torch.Tensor]:
    """
    Auto-regressive generation starting from a conditioned state. Returns a list of predictions with length num_predictions.
    The first prediction uses greedy sampling.
    :param encoder: Encoder
    :param decoder: Decoder
    :param sashimi: Mamba Sashimi
    :param state: Sashimi state list that has been conditioned or default state: sashimi.default_state(device=device)
    :param inference_params: Conditioned Mamba inference parameters or fresh ones (InferenceParams(max_seqlen=seq_len + context_len + 10, max_batch_size=1))
    :param initial_input: Initial input [1, 1]
    :param seq_len: Length of sequences to generate
    :param quantized: Whether multiclass classification setting is used
    :param num_predictions: Number of independent predictions e.g. number of sequences to generate
    :param temperature: Temperature for softmax sampling
    :param sampling_mode: one of ['greedy', 'prob', 'top_k']
    :param k: k for top_k sampling
    :param p: p for top_p sampling
    :param device: must be 'cuda'
    :return: list of predictions. The first prediction uses greedy sampling
    """
    sashimi = sashimi.to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    sashimi.eval()
    encoder.eval()
    decoder.eval()

    initial_state = copy.deepcopy(state)
    initial_inference_params = copy.deepcopy(inference_params)

    all_predictions = []

    # inference in recurrent mode
    with torch.no_grad():
        for p_idx in range(num_predictions):
            state = copy.deepcopy(initial_state)
            inference_params = copy.deepcopy(initial_inference_params)
            y = initial_input.to(device)
            if quantized:
                y = y.long()
            prediction_output = []

            # Auto-regressive generation. The output of the model is used as the next input.
            pbar = tqdm(total=seq_len)
            pbar.set_description(f'Auto-regressive generation {p_idx + 1} / {num_predictions}')
            for i in range(seq_len):
                # pass input through encoder-model-decoder pipeline
                y = encoder(y)
                y, state = sashimi.step(y, state=state, inference_params=inference_params)
                y = decoder(y, state)

                # sample from output. Output dim should be [1, num_classes] in a multiclass setting
                if sampling_mode == 'greedy' or p_idx == 0:
                    y = greedy_prediction(y)
                elif sampling_mode == 'prob':
                    y = multinomial_prediction(y, temperature=temperature)
                elif sampling_mode == 'top_k':
                    y = top_k_top_p_filtering(y, top_k=k, temperature=temperature)
                elif sampling_mode == 'top_p':
                    y = top_k_top_p_filtering(y, top_p=p, temperature=temperature)
                else:
                    print(f'Unknown sampling mode: {sampling_mode}')
                prediction_output.append(y.detach().cpu())
                inference_params.seqlen_offset += 1
                pbar.update()
            all_predictions.append(torch.stack(prediction_output, dim=1).cpu())
            pbar.close()

    return all_predictions


def mamba_generate_with_context(
        encoder: nn.Module,
        decoder: nn.Module,
        sashimi: MambaSashimi,
        context: torch.Tensor | np.ndarray | list,
        seq_len: int,
        quantized: bool = True,
        num_predictions: int = 10,
        temperature: float = 1.0,
        sampling_mode: str = 'prob',
        k: int = 10,
        device: str | torch.device = 'cuda'
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """
    Inference using a Sashimi model.
    The context is used to condition the model. After that, the model generates a sequence of len seq_len
    in auto-regressive mode. Returns the predictions for the context and the auto-regressively generated sequence

    :param encoder: Encoder
    :param decoder: Decoder
    :param sashimi: Mamba Sashimi
    :param context: context for the generation
    :param seq_len: length of the sequence to generate after the context
    :param quantized: whether the model works with quantized data in a multiclass setting
    :param num_predictions: number of predictions after conditioning the model on context
    :param temperature: temperature for softmax sampling
    :param sampling_mode: one of ['greedy', 'prob', top_k']
    :param k: k for top_k sampling
    :param device: device must be 'cuda' for mamba
    :return: context prediction, auto-regressive sequences. The first entry of auto-regressive
            sequences is greedy sampling
    """

    assert isinstance(sashimi, MambaSashimi)
    assert device == 'cuda'
    assert sampling_mode in supported_sampling_modes

    context_output, conditioned_state, conditioned_inference_params = condition(
        encoder=encoder,
        decoder=decoder,
        sashimi=sashimi,
        context=context,
        seq_len=seq_len,
        quantized=quantized,
        device=device
    )
    predictions = auto_regressive_generation(
        encoder=encoder,
        decoder=decoder,
        sashimi=sashimi,
        state=copy.deepcopy(conditioned_state),  # should not be changed. use deepcopy just to make sure
        inference_params=copy.deepcopy(conditioned_inference_params),  # deepcopy to make sure it is not changed
        initial_input=context_output[-1],
        seq_len=seq_len,
        num_predictions=num_predictions,
        temperature=temperature,
        sampling_mode=sampling_mode,
        k=k,
        device=device
    )
    return context_output, predictions
