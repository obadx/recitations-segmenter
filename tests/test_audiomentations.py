from pathlib import Path
from random import randint

from librosa.core import load
import soundfile as sf
from numpy.typing import NDArray
import numpy as np

from recitations_segmenter.train.augment import build_audiomentations_augs

if __name__ == '__main__':
    file_path = Path('/home/abdullah/Downloads/002091.wav')
    wav, sr = load(file_path, sr=16000)
    if isinstance(wav, np.ndarray):
        print('true')
    seed = randint(1, 23523)
    print(f'Seed={seed}')
    augs = build_audiomentations_augs(p=1, all=False, seed=seed)
    out_wav = augs(wav, sample_rate=sr)
    sf.write(file_path.parent / '002091_processed.wav', out_wav, sr)
