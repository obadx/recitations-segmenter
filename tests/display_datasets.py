from datasets import load_dataset, Dataset
import soundfile as sf

if __name__ == '__main__':
    orig_ds = load_dataset(
        '../segment-ds-processed.hf', download_mode='force_redownload')['recitation_6']
    ds = load_dataset(
        '../segment-ds-augmented.hf', download_mode='force_redownload')['recitation_6']

    idx = 9

    print(ds)
    print(f"Stretch Ratio: {ds[idx]['speed']}")
    print(f"Original intervals: {orig_ds[idx]['speech_intervals']}")
    print(f"Accelerated Intervals: {ds[idx]['speech_intervals']}")

    sf.write('out.wav', ds[idx]['audio']['array'], 16000)

    augmented_ds = ds.filter(
        lambda x: x['is_augmented'] == True and x['speed'] != 1)
    print(augmented_ds)
    sf.write('augmented_out.wav', augmented_ds[1]['audio']['array'], 16000)
