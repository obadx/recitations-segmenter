from pathlib import Path
import warnings

import yaml
from datasets import Dataset, DatasetDict, load_dataset, Audio, Array2D, IterableDatasetDict, Features, Value, IterableDataset
from dataclasses import dataclass
import os
import torch
import torchaudio
import librosa
import gc
from tqdm import tqdm

from ..utils import (
    download_file_fast,
    get_audiofiles,
    save_jsonl,
    SURA_TO_AYA_COUNT,
    overwrite_readme_yaml,
    downlaod_recitation_iterative,
    deduce_filename,
)
from .vad_utils import quran_split_by_silence_batch, load_vad_model

DS_FEATURES = Features({
    'aya_name': Value(dtype='string'),
    'reciter_name': Value(dtype='string'),
    'recitation_id': Value(dtype='int32'),
    'url': Value(dtype='string'),
    'audio': Audio(),
    'duration': Value(dtype='float32'),
    'speech_intervals': Array2D(shape=(None, 2), dtype="float32"),
    'is_interval_complete': Value(dtype='bool'),
})

DS_FEATURES_PROCESSED = Features({
    'aya_name': Value(dtype='string'),
    'reciter_name': Value(dtype='string'),
    'recitation_id': Value(dtype='int32'),
    'url': Value(dtype='string'),
    'audio': Audio(sampling_rate=16000),
    'duration': Value(dtype='float32'),
    'speech_intervals': Array2D(shape=(None, 2), dtype="float32"),
    'is_interval_complete': Value(dtype='bool'),
})


@dataclass
class Recitation:
    reciter_name: str
    id: int
    url: str
    dataset: Dataset = None
    download_path: Path = None
    window_size_samples: int = 1536
    threshold: float = 0.3
    min_silence_duration_ms: float = 300
    min_speech_duration_ms: float = 700
    pad_duration_ms: float = 30


def valid_aya_format(p: Path) -> bool:
    name = p.name.split('.')[0]
    if name in ['audhubillah', 'bismillah']:
        return True
    try:
        if len(name) != 6:
            return False
        int_name = int(name)
        sura_idx = int_name // 1000
        aya_name = int_name % 1000
        if sura_idx >= 1 and sura_idx <= 114:
            # 0 idx represnet bismillah
            if aya_name <= SURA_TO_AYA_COUNT[sura_idx]:
                return True
    except Exception as e:
        pass

    return False


def download_recitations(recitation: Recitation, base_dir) -> Path:
    p = Path(base_dir) / f'{recitation.id}'
    os.makedirs(p, exist_ok=True)

    # download the zip file form url
    try:
        out_path = download_file_fast(recitation.url, p, extract_zip=True)
    except Exception as e:
        warnings.warn(
            f'An Error happened while processing trying to download each file independently. Error: {e} ')
        zip_name = recitation.url.split('/')[-1]
        base_url = '/'.join(recitation.url.split('/')[:-1])
        downlaod_recitation_iterative(p / zip_name, base_url=base_url)

    return p

# 2. Custom decoder with mono conversion


def mono_decoder(batch):
    """Read mono channel (single) channel from aduio file
    """
    audio_data = []
    for audio in batch["audio"]:
        audio_path = audio['path']
        try:
            # Load audio (supports MP3, WAV, etc.)
            waveform, sample_rate = torchaudio.load(audio_path)

            # Force mono conversion (3 methods to choose from)
            if waveform.shape[0] > 1:  # Multi-channel audio
                # Method 1: Average channels (best for general use)
                mono_waveform = waveform.mean(dim=0)

                # Method 2: Select first channel (if left channel preferred)
                # mono_waveform = waveform[0]

                # Method 3: FFmpeg-style mix (requires custom weights)
                # weights = torch.tensor([0.8, 0.2])  # Custom channel weights
                # mono_waveform = (waveform * weights.view(-1, 1)).sum(dim=0)
            else:
                mono_waveform = waveform.squeeze()

            # see: https://huggingface.co/docs/datasets/v3.3.0/en/package_reference/main_classes#datasets.Audio
            audio_data.append({
                "array": mono_waveform.numpy(),
                "sampling_rate": sample_rate,
                "path": audio_path,
            })
        except Exception as e:
            print(f"⚠️ Failed {audio_path}: {str(e)}")
            raise e

    return {"audio": audio_data}


def librosa_mono_decoder(batch):
    audio_data = []
    durations: list[float] = []
    for audio in batch["audio"]:
        audio_path = audio['path']
        try:
            # Load as mono with original sample rate
            waveform, sample_rate = librosa.core.load(
                audio_path,
                sr=None,  # Keep native rate
                mono=True  # Force mono conversion
            )

            audio_data.append({
                "array": waveform,
                "sampling_rate": sample_rate,
                "path": audio_path,
                "bytes": None,  # solving bug for new dataset version 3.2.2
            })
            durations.append(len(waveform) / sample_rate)
        except Exception as e:
            print(f"⚠️ Failed {audio_path}: {str(e)}")
            raise e

    return {"audio": audio_data, "duration": durations}


def generate_ds(recitation: Recitation, ds_path: Path) -> Dataset:
    """
    Generating an audio dataset from folder with a metadata.jsonl file that contains:
        - audio_file
        - aya_name
        - reciter_name
        - reciter_id
        - url

    See: https://huggingface.co/docs/datasets/audio_dataset#audiofolder
    """
    metadata = []
    audio_pathes = get_audiofiles(
        ds_path,
        condition_callback=valid_aya_format,
        delete_audiofile_on_false_cond=True
    )

    audio_pathes = sorted(audio_pathes)
    for p in audio_pathes:
        metadata.append({
            'file_name': str(p.relative_to(ds_path)),
            'aya_name': p.name.split('.')[0],
            'reciter_name': recitation.reciter_name,
            'recitation_id': recitation.id,
            'url': recitation.url,
        })
    save_jsonl(metadata, ds_path / 'metadata.jsonl')
    ds = load_dataset(
        'audiofolder', data_dir=ds_path,
        split='train', streaming=True,
        features=DS_FEATURES,
    )

    # custom loading method for audio files
    ds = ds.cast_column("audio", Audio(decode=False))  # Keep raw paths
    ds = ds.map(
        librosa_mono_decoder,
        # mono_decoder,
        batched=True,
        batch_size=10,
        features=DS_FEATURES,
    )
    # resample to 16000
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    return ds


def intervals_map(batch, idx_to_recitation: dict[int, Recitation], device='cpu', model=None,):
    waves = [torch.tensor(i['array'], dtype=torch.float32)
             for i in batch['audio']]
    # make sure that the batch is the same recitation
    assert (
        idx_to_recitation[batch['recitation_id'][0]] == idx_to_recitation[batch['recitation_id'][-1]])
    recitation = idx_to_recitation[batch['recitation_id'][0]]
    outs = quran_split_by_silence_batch(
        waves,
        model=model,
        window_size_samples=recitation.window_size_samples,
        min_silence_duration_ms=recitation.min_silence_duration_ms,
        min_speech_duration_ms=recitation.min_speech_duration_ms,
        pad_duration_ms=recitation.pad_duration_ms,
        threshold=recitation.threshold,
        sample_rate=16000,
        device=device,
    )
    completes = []
    speech_intervals = []
    for out in outs:
        is_complete = out.clean_intervals.view(-1,)[-1] != float('inf')
        completes.append(is_complete)
        speech_intervals.append(out.clean_intervals.cpu().numpy())

        # clean gpu memory
        out.clean_gpu()

    # call garbage collection
    gc.collect()

    # clean GPU cache
    torch.cuda.empty_cache()

    return {
        'speech_intervals': speech_intervals,
        'is_interval_complete': completes,
    }


def to_huggingface_16k_dataset(
    recitations_file: str | Path,
    base_dir='data',
) -> IterableDatasetDict:
    """Converting Audio files to hugginface audio dataset and downsample to 16k

    Args:
        num_proc (int): number of parallel tasks to process the dataset
        limit (int): for testing only take out number of `limit` samples
    """

    base_dir = Path(base_dir)
    # Load the YAML file
    recitations = []
    idx_to_recitation = {}
    with open(recitations_file, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)['recitations']
    for rec in data:
        recitations.append(Recitation(**rec))
        idx_to_recitation[rec['id']] = Recitation(**rec)

    # Downloading and extracting all rectitations
    for idx in range(len(recitations)):
        recitation = recitations[idx]
        p = download_recitations(recitation, base_dir)
        recitations[idx].download_path = p

    # Generating dataset for every rectiation
    for idx in range(len(recitations)):
        recitation = recitations[idx]
        ds = generate_ds(recitation, recitation.download_path)
        recitations[idx].dataset = ds

    # concatenated dataset as datasetdict with key is the  reciter_id
    dataset_dict = IterableDatasetDict()
    for rec in recitations:
        dataset_dict[f'recitation_{rec.id}'] = rec.dataset

    return dataset_dict


def extract_speech_interval_from_ds(
    dataset_dict: IterableDatasetDict,
    recitations_file: str | Path,
    vad_model,
    device='cuda',
    batch_size=256,
) -> IterableDatasetDict:
    assert isinstance(dataset_dict, IterableDatasetDict)

    recitations = []
    idx_to_recitation = {}
    with open(recitations_file, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)['recitations']
    for rec in data:
        recitations.append(Recitation(**rec))
        idx_to_recitation[rec['id']] = Recitation(**rec)

    # the map loops over splits separtely
    dataset_dict = dataset_dict.map(
        intervals_map,
        batched=True,
        batch_size=batch_size,
        fn_kwargs={'model': vad_model, 'device': device,
                   'idx_to_recitation': idx_to_recitation},
    )
    dataset_dict = dataset_dict.cast(DS_FEATURES_PROCESSED)

    return dataset_dict


def save_to_disk(
    dataset: IterableDatasetDict,
    out_path: str | Path,
    samples_per_shard: int = 128,
):
    """save an Iterable hugginfce dataset dict onto disk
    """
    assert isinstance(dataset, IterableDatasetDict), (
        f'We only support IterableDatsetDict we got {type(dataset)}')

    out_path = Path(out_path)

    # create directory structure
    os.makedirs(out_path, exist_ok=True)

    # loop to save as parquet format
    metadata_items = []
    for split in dataset:
        cache = []
        shard_idx = 0
        metadata_items.append(
            {'split': split,
             'path': f'data/{split}/train/*.parquet'
             }
        )

        for idx, item in tqdm(enumerate(dataset[split])):
            cache.append(item)

            if idx % samples_per_shard == 0:
                shard_ds = Dataset.from_list(cache)
                shard_ds.to_parquet(
                    out_path / f'data/{split}/train/shard_{shard_idx:0{5}}.parquet')
                del shard_ds
                del cache
                gc.collect()
                cache = []
                shard_idx += 1

        # rest of the items
        if cache:
            shard_ds = Dataset.from_list(cache)
            shard_ds.to_parquet(
                out_path / f'data/{split}/train/shard_{shard_idx:0{5}}.parquet')
            del shard_ds
            del cache
            gc.collect()
            cache = []
            shard_idx += 1

    # create metadata yaml on top of the readme
    metadata = {
        'configs': [{
            'config_name': 'default',
            'data_files': metadata_items,
        }]
    }
    overwrite_readme_yaml(out_path / 'README.md', metadata)
