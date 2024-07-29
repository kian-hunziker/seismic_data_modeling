import os
import argparse
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from evaluation.eval_sashimi import sash_generate_with_context, sash_generate_without_context, plot_predictions
from evaluation.eval_utils import load_checkpoint, get_pipeline_components, print_hparams, get_model_summary
from train import LightningSequenceModel
import matplotlib.pyplot as plt


def setup_evaluation(path: str):
    # load checkpoint
    pl_module, hparams = load_checkpoint(path)
    # print hparams
    print_hparams(hparams)

    # check if quantization was used
    quantize = hparams['dataset']['quantize']
    print('Quantization is: ', quantize)

    # create directory for plots
    save_dir = os.path.join(path, 'eval_plots')

    # Check if the 'plots' directory exists
    if not os.path.exists(save_dir):
        # If it does not exist, create it
        os.makedirs(save_dir)
        print(f"Directory 'eval_plots' created at {save_dir}")
    else:
        print(f"Directory 'eval_plots' already exists at {save_dir}")

    # print model summary
    print_hparams(hparams['model'])
    print(get_model_summary(pl_module, max_depth=1))

    return pl_module, hparams, quantize, save_dir


def free_auto_regressive_generation(
        pl_module: LightningSequenceModel,
        seq_len: int,
        save_path: str,
        quantize: bool = False,
        device: str = 'cpu'
):
    print('\nAUTO-REGRESSIVE GENERATION\n')
    encoder, decoder, model = get_pipeline_components(pl_module)
    out = sash_generate_without_context(encoder, decoder, model, seq_len, quantize, device)

    title = 'Autoregressive_generation_no_context'

    plt.plot(out.detach().squeeze().numpy())
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, '{}.png'.format(title)))


def generate_with_context(
        pl_module: LightningSequenceModel,
        data_loader: DataLoader,
        save_path: str,
        show: bool = True,
        description: str = '',
        context_split: float = 0.8,
        quantize: bool = False,
        device: str = 'cpu'
):
    encoder, decoder, model = get_pipeline_components(pl_module)
    x, _ = next(iter(data_loader))

    num_examples = x.shape[0]
    data_len = x.shape[1]
    data_dim = x.shape[-1]
    context_len = int(context_split * data_len)
    generation_len = data_len - context_len - 1

    print('\nGENERATING WITH CONTEXT')
    print(f'num_examples: {num_examples}')
    print(f'example_len: {data_len}')
    print(f'context_len: {context_len}')
    print(f'generation_len: {generation_len}')

    for i in range(num_examples):
        print(f'example: {i + 1} / {num_examples}')
        context = x[i].squeeze()

        cp, pred = sash_generate_with_context(
            encoder=encoder,
            decoder=decoder,
            sashimi=model,
            context=context[:context_len],
            seq_len=generation_len,
            quantized=quantize,
            device=device
        )

        plot_predictions(
            full_context=context,
            predicted_context=cp,
            auto_reg_prediction=pred,
            len_context=context_len,
            title=f'{description}__Example_{i}',
            fig_size=(10, 5),
            line_width=1,
            save_path=save_path,
            show=show
        )
        plt.close()


def eval_train_and_test(
        path: str,
        context_len: int,
        generation_len: int,
        num_train_examples: int,
        show: bool = True
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pl_module, hparams, quantize, save_dir = setup_evaluation(path)

    train_dataset = pl_module.train_dataloader().dataset
    test_dataset = pl_module.test_dataloader().dataset

    total_len = context_len + generation_len + 1
    print(f'Total length: {total_len}')
    context_split = float(context_len) / float(total_len)

    train_dataset.dataset.sample_len = total_len
    test_dataset.sample_len = total_len

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=num_train_examples, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    '''free_auto_regressive_generation(
        pl_module=pl_module,
        seq_len=hparams['dataset']['sample_len'],
        save_path=save_dir,
        quantize=quantize,
        device=device
    )'''

    generate_with_context(
        pl_module=pl_module,
        data_loader=train_dataloader,
        save_path=save_dir,
        show=show,
        description='Train',
        context_split=context_split,
        quantize=quantize,
        device=device
    )
    generate_with_context(
        pl_module=pl_module,
        data_loader=test_dataloader,
        save_path=save_dir,
        show=show,
        description='Test',
        context_split=context_split,
        quantize=quantize,
        device=device
    )


def manual_test():
    # sash quantized
    path = 'wandb_logs/MA/2024-07-25__10_33_35'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pl_module, hparams, quantize, save_dir = setup_evaluation(path)

    train_dataset = pl_module.train_dataloader().dataset
    test_dataset = pl_module.test_dataloader().dataset

    num_examples = 3
    total_len = 4000
    context_split = 0.5

    train_dataset.sample_len = total_len
    train_loader = DataLoader(dataset=train_dataset, batch_size=num_examples, shuffle=False)

    free_auto_regressive_generation(pl_module, 1024, save_dir, quantize, device)

    generate_with_context(
        pl_module=pl_module,
        data_loader=train_loader,
        save_path=save_dir,
        description='Test',
        context_split=context_split,
        quantize=quantize,
        device=device
    )


def main(path, context_len, generation_len, num_training_examples):
    # Print the extracted arguments
    print(f"path: {path}")
    print(f"context_len: {context_len}")
    print(f"generation_len: {generation_len}")
    print(f"num_training_examples: {num_training_examples}")


if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Extract and print keyword arguments.")

    # Add arguments
    parser.add_argument('--path', type=str, required=True, help='Path to the data.')
    parser.add_argument('--c_len', type=int, required=True, help='Context length.')
    parser.add_argument('--g_len', type=int, required=True, help='Generation length.')
    parser.add_argument('--num_ex', type=int, required=True, help='Number of training examples.')
    parser.add_argument('--show', type=bool, default=False, help='Show plots')

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    print(args.show)
    eval_train_and_test(args.path, args.c_len, args.g_len, args.num_ex, args.show)
