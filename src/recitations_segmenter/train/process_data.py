from pathlib import Path
import yaml
from datasets import Dataset, DatasetDict, load_dataset, Audio
from dataclasses import dataclass
import os

from ..utils import download_file_fast, get_audiofiles, save_jsonl, SURA_TO_AYA_COUNT


@dataclass
class Recitation:
    reciter_name: str
    id: int
    url: str
    dataset: Dataset = None
    download_path: Path = None


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
            'reciter_id': recitation.id,
            'url': recitation.url,
        })
    save_jsonl(metadata, ds_path / 'metadata.jsonl')
    ds = load_dataset('audiofolder', data_dir=ds_path)
    return ds['train']


def to_huggingface_dataset(recitations_file: str | Path, base_dir='data') -> DatasetDict:
    """Converting Audio files to hugginface audio dataset
    """

    base_dir = Path(base_dir)
    # Load the YAML file
    recitations = []
    with open(recitations_file, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)['recitations']
    for rec in data:
        recitations.append(Recitation(**rec))

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
        dataset_dict[str(rec.id)] = rec.dataset

    # cast dataset to sampling_rate 16000
    dataset_dict = dataset_dict.cast_column(
        'audio', Audio(sampling_rate=16000))

    return dataset_dict
