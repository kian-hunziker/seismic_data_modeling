import os
import yaml
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from models.sashimi.sashimi_standalone import Sashimi
import evaluation.eval_utils as u
from dataloaders.data_utils.signal_encoding import quantize_encode, decode_dequantize, normalize_11
from evaluation.eval_utils import load_checkpoint, get_pipeline_components
import matplotlib.pyplot as plt


def sashimi_beam_search(
        sashimi: Sashimi,
        encoder: nn.Module,
        decoder: nn.Module,
        initial_state: list,
        initial_input: torch.Tensor,
        beam_width: int,
        sequence_length: int,
        temperature: float = 1.0,
        device: str | torch.device = 'cpu',
):
    sashimi = sashimi.to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    sashimi.eval()
    sashimi.setup_rnn()
    encoder.eval()
    decoder.eval()

    new_state = None
    with torch.no_grad():
        y = encoder(initial_input)
        y, new_state = sashimi.step(y, initial_state)
        y = decoder(y, new_state)
        y = y / temperature
        y_softmax = F.softmax(y, dim=-1)
        top_k_predictions = torch.topk(y_softmax, beam_width)

    states = [copy.deepcopy(new_state) for _ in range(beam_width)]
    inputs = [(torch.log(v), i) for v, i in torch.cat(top_k_predictions).T]

    out_sequences = [[] for _ in range(beam_width)]
    with torch.no_grad():
        pbar = tqdm(total=sequence_length)
        pbar.set_description('Beam search')
        for step in range(sequence_length):
            y_candidates = []
            v_candidates = []
            state_candidates = []
            for i in range(beam_width):
                # generate new outputs
                y = encoder(inputs[i][1].unsqueeze(-1).unsqueeze(-1).long())
                y, new_state = sashimi.step(y, states[i])
                y = decoder(y, new_state)

                # softmax & top_k prediction
                y = y / temperature
                y_softmax = torch.log(F.softmax(y, dim=-1)) + inputs[i][0]
                top_k_predictions = torch.topk(y_softmax, beam_width)

                # store state for future predictions
                state_candidates.append(copy.deepcopy(new_state))
                # store probabilities and prediction values
                for v, idx in torch.cat(top_k_predictions).T:
                    y_candidates.append(idx)
                    v_candidates.append(v)
            # select top candidates from candidate list
            y_candidates = torch.tensor(y_candidates)
            top_k_probabilities, top_k_indices = torch.topk(torch.tensor(v_candidates), beam_width)
            temp_sequences = copy.deepcopy(out_sequences)
            for idx in range(beam_width):
                selected_output = y_candidates[int(top_k_indices[idx])]
                current_prob = top_k_probabilities[idx]
                inputs[idx] = (current_prob, selected_output)
                states[idx] = copy.deepcopy(state_candidates[int(top_k_indices[idx] // beam_width)])
                out_sequences[idx] = copy.deepcopy(temp_sequences[int(top_k_indices[idx] // beam_width)])
                out_sequences[idx].append(selected_output)
            pbar.update()
    return out_sequences


def beam_search_test():
    ckpt_path = '../wandb_logs/MA/2024-07-24__16_56_19'
    pl_module, hparams = u.load_checkpoint(ckpt_path)
    encoder, decoder, model = u.get_pipeline_components(pl_module)
    u.print_hparams(hparams)

    t = np.linspace(0, 2, 32000)
    context_len = 10
    prediction_len = 100
    f = 50
    a = .5
    p = 0
    noise = 0.0
    full_context = np.sin(2 * np.pi * f * t + p) * a + noise * np.random.randn(len(t))

    short_context = torch.from_numpy(full_context)[:context_len].type(torch.float32)
    short_context_quantized = quantize_encode(short_context).unsqueeze(0).unsqueeze(-1)
    state = model.default_state()
    model.eval()
    model.setup_rnn()

    with torch.no_grad():
        # 'condition' the model by passing the context
        # teacher forcing approach, the output is saved for plotting but not fed back to model.
        pbar = tqdm(total=context_len)
        pbar.set_description('Processing context')
        context_output = []
        for i in range(context_len):
            y = encoder(short_context_quantized[:, i])
            y, state = model.step(y, state)
            y = decoder(y, state)
            y = torch.argmax(y, dim=-1).unsqueeze(0)
            context_output.append(y.detach().cpu())
            pbar.update()
        pbar.close()

    out = sashimi_beam_search(
        sashimi=model,
        encoder=encoder,
        decoder=decoder,
        initial_state=state,
        initial_input=context_output[-1],
        beam_width=3,
        sequence_length=prediction_len,
        temperature=2,
    )

    x = torch.arange(0, context_len + prediction_len)
    context_output = torch.stack(context_output, dim=1)
    plt.plot(x, quantize_encode(torch.from_numpy(full_context))[:len(x)], label='context')
    plt.plot(x[:context_len], context_output.squeeze().numpy(), label='greedy')
    for i, o in enumerate(out):
        plt.plot(x[context_len:], torch.tensor(o).numpy(), label=f'prediction {i}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    beam_search_test()
