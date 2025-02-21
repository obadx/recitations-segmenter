from datasets import load_dataset

from recitations_segmenter.train.process_data import extract_speech_interval_from_ds, save_to_disk
from recitations_segmenter.train.vad_utils import load_vad_model

if __name__ == '__main__':
    dataset_path = '../segment-ds.hf'
    recitations_file = './recitations.yml'
    device = 'cuda'
    batch_size = 32
    samples_per_shard = 128
    out_path = '../segment-ds-processed.hf'

    model = load_vad_model().to(device)

    ds = load_dataset(dataset_path, streaming=True)
    ds = extract_speech_interval_from_ds(
        ds,
        recitations_file,
        vad_model=model,
        batch_size=batch_size,
        device=device,
    )
    print(ds)

    save_to_disk(ds, out_path=out_path,
                 samples_per_shard=samples_per_shard)
