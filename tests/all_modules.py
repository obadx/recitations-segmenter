from recitations_segmenter import segment_recitations, read_audio, clean_speech_intervals
from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification
import torch

if __name__ == '__main__':
    device = torch.device('cuda')
    dtype = torch.bfloat16
    processor = AutoFeatureExtractor.from_pretrained(
        "obadx/recitation-segmenter-v2")
    model = AutoModelForAudioFrameClassification.from_pretrained(
        "obadx/recitation-segmenter-v2",
    )

    model.to(device, dtype=dtype)

    file_path = './assets/dussary_002282.mp3'
    wav = read_audio(file_path)
    print(wav.shape)

    # Extracting speech inervals in samples according to 16000 Sample rate
    sampled_outputs = segment_recitations(
        [wav],
        model,
        processor,
        device=device,
        dtype=dtype,
        batch_size=4,
    )

    clean_outputs = []
    # Clean The speech intervals by:
    # * merging small silence durations
    # * remove small speech durations
    # * add padding to each speech duration
    for out in sampled_outputs:
        clean_out = clean_speech_intervals(
            out.speech_intervals,
            out.is_complete,
            min_silence_duration_ms=30,
            min_speech_duration_ms=30,
            pad_duration_ms=30,
            return_seconds=True,
        )
        clean_outputs.append(clean_out)

    print(len(clean_outputs))
    print(clean_outputs[0].clean_speech_intervals)
