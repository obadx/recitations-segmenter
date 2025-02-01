from recitations_segmenter.train .vad_utils import read_audio, quran_split_by_silence


if __name__ == '__main__':
    file_path = '/home/abdullah/Downloads/001.adts'
    wav = read_audio(file_path)
    out = quran_split_by_silence(
        wav,
        sample_rate=16000,
        window_size_samples=1536,
        min_silence_duration_ms=400,
        pad_duration_ms=30,
        threshold=0.3,
        device='cuda',
    )
    print(out.clean_intervals.shape)
    print(out.clean_intervals)
