import argparse

from datasets import load_dataset

from recitations_segmenter.train.process_data import extract_speech_interval_from_ds, save_to_disk
from recitations_segmenter.train.vad_utils import load_vad_model


def main(args):
    # Load VAD model
    print(f"Loading VAD model on {args.device}...")
    model = load_vad_model().to(args.device)

    # Load and process dataset
    print(f"Loading dataset from {args.dataset_path}...")
    ds = load_dataset(args.dataset_path, streaming=True)

    print("Processing speech intervals...")
    processed_ds = extract_speech_interval_from_ds(
        ds,
        args.recitations_file,
        vad_model=model,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"Processed dataset:\n{processed_ds}")

    # Save results
    print(f"Saving processed dataset to {args.out_path}...")
    save_to_disk(processed_ds,
                 out_path=args.out_path,
                 samples_per_shard=args.samples_per_shard)
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
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for VAD processing'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
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
