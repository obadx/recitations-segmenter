from pathlib import Path

from recitations_segmenter import segment_recitations, read_audio, clean_speech_intervals
from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification
import torch

if __name__ == '__main__':
    device = torch.device('cuda')
    dtype = torch.bfloat16
    cach_dir = './q_cache'

    processor = AutoFeatureExtractor.from_pretrained(
        "obadx/recitation-segmenter-v2")
    model = AutoModelForAudioFrameClassification.from_pretrained(
        "obadx/recitation-segmenter-v2",
    )

    model.to(device, dtype=dtype)

    file_pathes = list(Path('./assets').glob('*.mp3'))
    waves = [read_audio(p) for p in file_pathes]

    # Extracting speech inervals in samples according to 16000 Sample rate
    sampled_outputs = segment_recitations(
        waves,
        model,
        processor,
        device=device,
        dtype=dtype,
        batch_size=4,
        cache_dir=cach_dir,
        overwrite_cache=False,
    )

    # Clean The speech intervals by:
    # * merging small silence durations
    # * remove small speech durations
    # * add padding to each speech duration
    # Raises:
    # * NoSpeechIntervals: if the wav is complete silence
    # * TooHighMinSpeechDruation: if `min_speech_duration` is too high which
    # resuls for deleting all speech intervals
    for out, path in zip(sampled_outputs, file_pathes):
        clean_out = clean_speech_intervals(
            out.speech_intervals,
            out.is_complete,
            min_silence_duration_ms=30,
            min_speech_duration_ms=30,
            pad_duration_ms=30,
            return_seconds=True,
        )

        print(f'Speech Intervals of: {path.name}: ')
        print(clean_out.clean_speech_intervals)
        print(clean_out.is_complete)
        print('-' * 40)
