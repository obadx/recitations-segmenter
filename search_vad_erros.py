import gc

from datasets import load_dataset, Dataset
import torch

from recitations_segmenter.train.process_data import extract_speech_interval_from_ds, save_to_disk
from recitations_segmenter.train.vad_utils import load_vad_model, quran_split_by_silence_batch


def main():
    device = 'cuda'

    # Load VAD model
    print(f"Loading VAD model on {device}...")
    model = load_vad_model().to(device)

    # Load and process dataset
    ds = load_dataset('parquet', data_files='./shard_00007.parquet')['train']

    for item in ds:
        waves = [torch.tensor(item['audio']['array'], dtype=torch.float32)]
        print(f'Aya name: {item["aya_name"]}')
        outs = quran_split_by_silence_batch(
            waves,
            model=model,
            window_size_samples=1536,
            threshold=0.3,
            min_silence_duration_ms=400,
            min_speech_duration_ms=1000,
        )

        # clean gpu memory
        for out in outs:
            out.clean_gpu()

        # call garbage collection
        gc.collect()

        # clean GPU cache
        torch.cuda.empty_cache()


if __name__ == '__main__':

    main()
