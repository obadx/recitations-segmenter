import argparse
from pathlib import Path
import subprocess
import os

import yaml
import submitit

from recitations_segmenter.utils import overwrite_readme_yaml
from recitations_segmenter.train.process_data import DS_FEATURES_PROCESSED


def load_splits(recitations_file_path) -> list[str]:
    splits = []
    with open(recitations_file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)['recitations']
    for rec in data:
        splits.append(f'recitation_{rec["id"]}')

    return splits


def write_redmme(splits: list[str], dataset_path):
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
        'dataset_info': {'featrues': DS_FEATURES_PROCESSED._to_yaml_list},
        'configs': [{
            'config_name': 'default',
            'data_files': metadata_items,
        }]
    }
    overwrite_readme_yaml(Path(dataset_path) / 'README.md', metadata)


def process_split(
    split,
    dataset_path,
    batch_size,
    out_path,
    device='cpu',
    samples_per_shard=1024,
):

    # Build the command with environment setup and Python execution
    cmd = f"""
    echo 'Extracting Speech intervals #########'
    bash /cluster/users/shams035u1/.bashrc
    source /cluster/users/shams035u1/data/miniconda3/bin/activate
    conda activate segment
    cd /cluster/users/shams035u1/data/codes/recitations-segmenter

    python extract_speech_intervals.py \\
        --dataset-path {dataset_path} \\
        --split {split} \\
        --device {device} \\
        --batch-size {batch_size} \\
        --samples-per-shard {samples_per_shard} \\
        --out-path {out_path}

    """
    print(cmd)

    # Execute the command in a single shell session
    subprocess.run(cmd, shell=True, check=True)


def main(args):
    # Loading dataset splits
    splits = load_splits('./recitations.yml')

    # Write yaml metadata to README.md
    write_redmme(splits, args.out_path)

    # Configure Slurm
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        slurm_account="shams035",
        slurm_partition="cpu",
        slurm_time="2-01:00:00",
        slurm_ntasks_per_node=1,
        cpus_per_task=16,
    )

    # Submit jobs
    for split in splits:
        executor.update_parameters(
            slurm_job_name=f"{split}",
            slurm_additional_parameters={
                "output": f"QVADcpu_{split}_%j.out"  # %j = Slurm job ID
            }
        )
        job = executor.submit(
            process_split,
            split=split,
            dataset_path=args.dataset_path,
            batch_size=args.batch_size,
            out_path=args.out_path,
            device=args.device,
            samples_per_shard=args.samples_per_shard,
        )

        print(job.job_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process audio dataset with voice activity detection',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='../segment-ds.hf',
        help='Path to input Hugging Face dataset'
    )
    parser.add_argument(
        '--recitations-file',
        type=str,
        default='./recitations.yml',
        help='Path to recitations metadata YAML file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cuda', 'cpu'],
        help='Device to use for VAD processing'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for VAD processing'
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
        default='../segment-ds-processed.hf',
        help='Output path for processed dataset'
    )

    args = parser.parse_args()

    main(args)
