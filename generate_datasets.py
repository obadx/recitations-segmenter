import argparse

from recitations_segmenter.train.process_data import to_huggingface_16k_dataset, save_to_disk


def main(args):
    # Process and create dataset
    ds = to_huggingface_16k_dataset(
        args.recitations_file, base_dir=args.download_dir)
    print(f'Processed dataset:\n{ds}')

    # Save dataset
    save_to_disk(ds, out_path=args.out_path,
                 samples_per_shard=args.samples_per_shard)
    print(f'Dataset saved to {args.out_path} with {
          args.samples_per_shard} samples per shard')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process Quran recitations and create a Hugging Face dataset'
    )
    parser.add_argument(
        '--download-dir',
        type=str,
        default='./data',
        help='Directory to download audio files (default: ./data)'
    )
    parser.add_argument(
        '--recitations-file',
        type=str,
        default='./recitations.yml',
        help='Path to YAML file containing recitation metadata (default: ./recitations.yml)'
    )
    parser.add_argument(
        '--out-path',
        type=str,
        default='../segment-ds.hf',
        help='Output path for the generated dataset (default: ../segment-ds.hf)'
    )
    parser.add_argument(
        '--samples-per-shard',
        type=int,
        default=128,
        help='Number of samples per dataset shard (default: 128)'
    )

    args = parser.parse_args()

    main(args)
