from datasets import load_dataset
import torch
from recitations_segmenter.train.vad_utils import read_audio, quran_split_by_silence_batch


if __name__ == '__main__':
    # waves = []
    # file_path = '/home/abdullah/Downloads/002002.mp3'
    # wav = read_audio(file_path)
    # waves.append(wav)
    #
    # file_path = '/home/abdullah/Downloads/001.adts'
    # wav = read_audio(file_path)
    # waves.append(wav)
    #
    # file_path = '/home/abdullah/Downloads/054.mp3'
    # wav = read_audio(file_path)
    # waves.append(wav)

    ds = load_dataset('abdullahaml/test')['train']
    waves = []
    for item in ds:
        wav = item['audio']['array']
        wav = torch.tensor(wav, dtype=torch.float32)
        waves.append(wav)

    waves = [
        torch.tensor([1, 2, 3]),
        torch.tensor([3, 4, 5, 5]),
        torch.tensor([3, 4, 5, 5, 8]),
    ]
    wav[-4000:] = 0.0
    waves = [wav for _ in range(512)]

    out = quran_split_by_silence_batch(
        waves,
        sample_rate=16000,
        window_size_samples=1536,
        min_silence_duration_ms=400,
        min_speech_duration_ms=700,
        pad_duration_ms=30,
        threshold=0.3,
        device='cuda',
    )
    print(out[0].clean_intervals.shape)
    print(out[0].clean_intervals)
