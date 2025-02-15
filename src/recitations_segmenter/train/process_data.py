from pathlib import Path
import yaml
from datasets import Dataset, DatasetDict, load_dataset, Audio, Array2D
from dataclasses import dataclass
import os
import torch
import numpy as np

from ..utils import (
    download_file_fast,
    get_audiofiles,
    save_jsonl,
    SURA_TO_AYA_COUNT,
)
from .vad_utils import quran_split_by_silence_batch, load_vad_model


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
    out_path = download_file_fast(recitation.url, p, extract_zip=True)
    return p


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
    ds = load_dataset('audiofolder', data_dir=ds_path)
    return ds['train']


def intervals_map(batch, recitation: Recitation, device='cpu', model=None,):
    waves = [torch.tensor(i['array'], dtype=torch.float32)
             for i in batch['audio']]
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

    return {
        'speech_intervals': speech_intervals,
        'is_interval_complete': completes,
    }


def extarct_speech_intervals(
    dataset: Dataset,
    recitation: Recitation,
    batch_size=256,
    device='cuda',
    model=None,
) -> Dataset:

    ds = dataset.map(
        intervals_map,
        batched=True,
        batch_size=batch_size,
        fn_kwargs={'model': model, 'device': device, 'recitation': recitation},
    )
    return ds


def to_huggingface_dataset(
    recitations_file: str | Path,
    base_dir='data',
    batch_size=256,
    device='cuda',
    limit: int = None,
) -> DatasetDict:
    """Converting Audio files to hugginface audio dataset

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

    # Generating datgaset for every rectiation
    for idx in range(len(recitations)):
        recitation = recitations[idx]
        ds = generate_ds(recitation, recitation.download_path)
        recitations[idx].dataset = ds

    # concatenated dataset as datasetdict with key is the  reciter_id
    dataset_dict = DatasetDict()
    for rec in recitations:
        dataset_dict[f'recitation_{rec.id}'] = rec.dataset

    # cast dataset to sampling_rate 16000
    dataset_dict = dataset_dict.cast_column(
        'audio', Audio(sampling_rate=16000))

    print(dataset_dict)

    # extract speech intervals
    model = load_vad_model().to(device)
    for rec_id in dataset_dict:
        id = int(rec_id.split('_')[-1])
        ds = dataset_dict[rec_id]
        if limit:
            ds = ds.select(range(limit))

        dataset_dict[rec_id] = extarct_speech_intervals(
            ds, idx_to_recitation[id],
            device=device,
            batch_size=batch_size,
            model=model,
        )
    dataset_dict = dataset_dict.cast_column(
        'speech_intervals', Array2D(shape=(None, 2), dtype="float32"))

    return dataset_dict
