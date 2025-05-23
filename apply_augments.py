import argparse
import os
from pathlib import Path

from datasets import load_dataset, Features
import numpy as np
import submitit

from recitations_segmenter.train.process_data import save_to_disk_split
from recitations_segmenter.train.augment import (
    augment_ds_split,
    DS_FEATURES_TRAIN,
    AugmentConfig,
    extract_features_for_ds,
)
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
    config: AugmentConfig,
    split,
    seed
) -> None:
    ds = load_dataset(args.dataset_path, streaming=True, split=split)

    aug_ds_split = augment_ds_split(
        ds,
        seed=int(seed),
        stretch_ragne=[config.min_stretch_ratio, config.max_stretch_ratio],
        augment_prob=config.augment_prob,
        batch_size=config.batch_size,
    )

    out_ds_split = extract_features_for_ds(aug_ds_split, config)

    out_ds_split = out_ds_split.shuffle(seed=seed)

    save_to_disk_split(
        out_ds_split,
        split_name=split,
        out_path=args.out_path,
        samples_per_shard=config.samples_per_shard,
    )


def main(args):

    config = AugmentConfig.from_yaml('./augment_config.yml')
    ds_dict = load_dataset(args.dataset_path, streaming=True)

    # Writing out dataset metadata
    splits = [split for split in ds_dict]
    write_redmme(splits, args.out_path, features=DS_FEATURES_TRAIN)

    # generating reandom seeeds for every split
    rng = np.random.default_rng(seed=config.seed)
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
            config=config,
            split=split,
            seed=seed,
        )
        print(job.job_id)

        # process_ds(args=args, split=split, config=config, seed=int(seed))


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
        '--out-path',
        type=str,
        default='../segment-ds-augmented.hf',
        help='Output path for processed dataset'
    )

    args = parser.parse_args()

    main(args)
