from recitations_segmenter import segment_recitations, read_audio
from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification
import torch

if __name__ == '__main__':
    device = torch.device('cuda')
    dtype = torch.bfloat16
    processor = AutoFeatureExtractor.from_pretrained(
        "obadx/recitation-segmenter-v2")
    model = AutoModelForAudioFrameClassification.from_pretrained(
        "obadx/recitation-segmenter-v2",
        torch_dtype=dtype,
        device_map=device,
    )

    file_path = '/home/abdullah/Downloads/002282.mp3'
    wav = read_audio(file_path)
    print(wav.shape)

    output = segment_recitations(
        [wav],
        model,
        processor,
        return_seconds=True,
        device=device,
        dtype=dtype,
        min_silence_duration_ms=30,
        min_speech_duration_ms=30,
        pad_duration_ms=30,
        batch_size=4,
    )

    print(len(output))
    print(output[0].clean_speech_intervals)
