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

    file_path = '/home/abdullah/Downloads/002091.wav'
    wav = read_audio(file_path)

    output = segment_recitations(
        [wav],
        model,
        processor,
        return_seconds=True,
        device=device,
        dtype=dtype,
    )

    print(len(output))
    print(output[0].clean_speech_intervals)
