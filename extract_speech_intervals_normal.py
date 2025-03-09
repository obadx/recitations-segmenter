import argparse

from datasets import load_dataset

from recitations_segmenter.train.process_data import extract_speech_interval_from_ds_normal, save_to_disk_normal
from recitations_segmenter.train.vad_utils import load_vad_model


def main(args):
    device = 'cpu'

    # Load and process dataset
    print(f"Loading dataset from {args.dataset_path}...")
    ds = load_dataset(args.dataset_path, streaming=False)

    print("Processing speech intervals...")
    processed_ds = extract_speech_interval_from_ds_normal(
        ds,
        args.recitations_file,
        device=device,
        num_proc=args.num_proc,
    )
    print(f"Processed dataset:\n{processed_ds}")

    # Save results
    print(f"Saving processed dataset to {args.out_path}...")
    save_to_disk_normal(
        processed_ds,
        out_path=args.out_path,
        samples_per_shard=args.samples_per_shard,
    )
    print("Processing complete!")


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
        '--num-proc',
        type=int,
        default=7,
        help='Num of parallel processes'
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
