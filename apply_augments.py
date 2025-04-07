import argparse
import os
from pathlib import Path

from datasets import load_dataset, Features
import numpy as np
import submitit

from recitations_segmenter.train.process_data import save_to_disk_split
from recitations_segmenter.train.augment import augment_ds_split, DS_FEATURES_AUGMNETED
from recitations_segmenter.utils import overwrite_readme_yaml


def write_redmme(splits: list[str], dataset_path, features: Features):
    """Write metadata to yaml section of readme:
    EX:

    ---
    configs:
    - config_name: default
    data_files:
    - path: data/recitation_6/train/*.parquet
        split: recitation_6
    ---
    """
    os.makedirs(dataset_path, exist_ok=True)

    metadata_items = []
    for split in splits:
        metadata_items.append(
            {'split': split,
             'path': f'data/{split}/train/*.parquet'
             }
        )
    metadata = {
        'dataset_info': {'featrues': features._to_yaml_list()},
        'configs': [{
            'config_name': 'default',
            'data_files': metadata_items,
        }]
    }
    overwrite_readme_yaml(Path(dataset_path) / 'README.md', metadata)


def process_ds(
    args,
    split,
    seed
) -> None:
    ds = load_dataset(args.dataset_path, streaming=True, split=split)
    ds = ds.shuffle(seed=seed)
    out_ds_split = augment_ds_split(
        ds,
        seed=int(seed),
        stretch_ragne=[args.min_stretch_ratio, args.max_stretch_ratio],
        augment_prob=args.augment_prob,
        batch_size=args.batch_size,
    )
    save_to_disk_split(
        out_ds_split,
        split_name=split,
        out_path=args.out_path,
        samples_per_shard=args.samples_per_shard,
    )


def main(args):

    ds_dict = load_dataset(args.dataset_path, streaming=True)

    # Writing out dataset metadata
    splits = [split for split in ds_dict]
    write_redmme(splits, args.out_path, features=DS_FEATURES_AUGMNETED)

    # generating reandom seeeds for every split
    rng = np.random.default_rng(seed=args.seed)
    seeds = rng.integers(low=0, high=512, size=(len(splits),))

    # Configure Slurm
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        slurm_account="shams035",
        slurm_partition="cpu",
        slurm_time="2-01:00:00",
        slurm_ntasks_per_node=1,
        cpus_per_task=16,
    )

    for split, seed in zip(splits, seeds):
        print(f'split={split}, seed={seed}')
        executor.update_parameters(
            slurm_job_name=f"{split}",
            slurm_additional_parameters={
                # "output": f"QVADcpu_{split}_%j.out"  # %j = Slurm job ID
            }
        )
        job = executor.submit(
            process_ds,
            args=args,
            split=split,
            seed=seed,
        )

        print(job.job_id)
        # process_ds(args, split, int(seed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare Dataset for VAD training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='../segment-ds-processed.hf',
        help='Path to input Hugging Face dataset'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for VAD processing'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Initial seed'
    )
    parser.add_argument(
        '--min-stretch-ratio',
        type=float,
        default=0.8,
        help='The mininum value for stretching the audio array',
    )
    parser.add_argument(
        '--max-stretch-ratio',
        type=float,
        default=1.5,
        help='The maximux value for stretching the audio array',
    )
    parser.add_argument(
        '--augment-prob',
        type=float,
        default=0.4,
        help='The Augmentatinon probability for the dataset (The precentage of augmented samples)',
    )
    parser.add_argument(
        '--samples-per-shard',
        type=int,
        default=128,
        help='Number of samples per output shard'
    )
    parser.add_argument(
        '--out-path',
        type=str,
        default='../segment-ds-augmented.hf',
        help='Output path for processed dataset'
    )

    args = parser.parse_args()

    main(args)
