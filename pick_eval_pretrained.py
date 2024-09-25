import argparse
import evaluation.pick_eval as pe
from models.evaluation_wrapper import PhasePickerLit

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate evaluation targets. See the docstring for details."
    )
    parser.add_argument(
        "model_name", type=str, default='PhaseNet', help="Name of pretrained model"
    )
    parser.add_argument(
        'train_dataset', type=str, default='ethz', help="Name of the dataset used to pretrain the model"
    )
    parser.add_argument(
        '--target_dataset', type=str, default='ETHZ', help="Name of the dataset to evaluate"
    )
    parser.add_argument(
        '--num_workers', type=int, default=0, help="Number of workers for dataloader"
    )
    args = parser.parse_args()

    sets = 'train,dev,test'
    target_path = 'evaluation/eval_tasks/' + args.target_dataset

    model_name = args.model_name
    training_dataset = args.train_dataset

    model = PhasePickerLit(ckpt_path=None, pretrained_name=model_name, pretrained_dataset=training_dataset)
    pe.save_pick_predictions(
        model=model,
        target_path=target_path,
        ckpt_path=f'pretrained_benchmark/{model_name}',
        sets=sets,
        save_tag='eval',
        batch_size=64,
        num_workers=args.num_workers
    )
