import os
from pathlib import Path
import shutil

from datasets import load_dataset, Dataset, disable_caching
import soundfile as sf

if __name__ == '__main__':
    disable_caching()
    orig_ds = load_dataset(
        '../segment-ds-processed.hf')['recitation_6']
    ds = load_dataset(
        '../segment-ds-augmented.hf',
        keep_in_memory=True,
        download_mode='force_redownload',
    )['recitation_6']

    idx = 9

    print(ds)
    print(f"Stretch Ratio: {ds[idx]['speed']}")
    print(f"Original intervals: {orig_ds[idx]['speech_intervals']}")
    print(f"Accelerated Intervals: {ds[idx]['speech_intervals']}")

    # sf.write('out.wav', ds[idx]['audio']['array'], 16000)
    #
    # augmented_ds = ds.filter(
    #     lambda x: x['is_augmented'] == True and x['speed'] != 1)
    # print(augmented_ds)
    # sf.write('augmented_out.wav', augmented_ds[1]['audio']['array'], 16000)

    shutil.rmtree('out-waves')
    os.makedirs('out-waves', exist_ok=True)

    for item in ds:
        sf.write(f"out-waves/{item['aya_id']}.wav",
                 item['audio']['array'], 16000)
        del item['audio']
        del item['input_features']
        print(item)
        print('-' * 40)
