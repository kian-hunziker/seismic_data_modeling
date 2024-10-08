import argparse
import evaluation.pick_eval as pe
from models.evaluation_wrapper import PhasePickerLit

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate evaluation targets. See the docstring for details."
    )
    parser.add_argument(
        "checkpoint", type=str, help="Path to model checkpoint"
    )
    parser.add_argument(
        '--target_dataset', type=str, default='ETHZ', help="Name of the target dataset e.g. ETHZ"
    )
    parser.add_argument(
        '--num_workers', type=int, default=0, help="Number of workers for dataloader"
    )
    args = parser.parse_args()

    sets = 'train,dev,test'
    ckpt_path = args.checkpoint

    model = PhasePickerLit(ckpt_path)

    if '.ckpt' in ckpt_path:
        ckpt_path = '/'.join(ckpt_path.split('/')[:-2])

    target_path = 'evaluation/eval_tasks/' + args.target_dataset
    sample_lengths = [3072, 4096, 6144, 8192, 10240, 12288]
    for sample_length in sample_lengths:
        print('*' * 32, '\n')
        print(f'Sample length: {sample_length}')
        print('*' * 32, '\n')

        # set sample length. When the evaluation augmentations are requested, this sample length
        # is used to sample the steered window
        model.sample_len = sample_length

        pe.save_pick_predictions(
            model=model,
            target_path=target_path,
            ckpt_path=ckpt_path,
            sets=sets,
            save_tag=f'sample_len_{sample_length}',
            batch_size=64,
            num_workers=args.num_workers
        )
