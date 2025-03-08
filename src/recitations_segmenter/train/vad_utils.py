from pathlib import Path
from dataclasses import dataclass

import torchaudio
import torch
from torch.nn.utils.rnn import pad_sequence

"""
The code for this file is developed using this notebook:
https://colab.research.google.com/drive/114cbKnaMXgrERug7otodEi1HgappW4dw?usp=sharing
"""

SILERO_VAD_PATH = Path(__file__).parent / '../data/silero_vad_v4.0.jit'


def read_audio(path: str,
               sampling_rate: int = 16000):
    list_backends = torchaudio.list_audio_backends()

    assert len(list_backends) > 0, 'The list of available backends is empty, please install backend manually. \
                                    \n Recommendations: \n \tSox (UNIX OS) \n \tSoundfile (Windows OS, UNIX OS) \n \tffmpeg (Windows OS, UNIX OS)'

    try:
        effects = [
            ['channels', '1'],
            ['rate', str(sampling_rate)]
        ]

        wav, sr = torchaudio.sox_effects.apply_effects_file(
            path, effects=effects)
    except:
        wav, sr = torchaudio.load(path)

        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != sampling_rate:
            transform = torchaudio.transforms.Resample(orig_freq=sr,
                                                       new_freq=sampling_rate)
            wav = transform(wav)
            sr = sampling_rate

    assert sr == sampling_rate
    return wav.squeeze(0)


def init_jit_model(model_path: str,
                   device=torch.device('cpu')):
    torch.set_grad_enabled(False)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


def load_vad_model():
    return init_jit_model(SILERO_VAD_PATH)


class NoSpeechIntervals(Exception):
    pass


class TooHighMinSpeechDuration(Exception):
    pass


@dataclass
class SegmentationOutput:
    """
    Attrubutes:
        intervlas (torch.FloatTensor): the actual speech intervlas of the model without any cleaning (in seconds)
        pobs: (torch.FloatTensor): the average probabilty for every speech segment for `intervals` without cleaning. Same shape as `intervlas`
        clean_intervlas (torch.FloatTensor): the speech intervlas after merging short silecne intervals (< min_silence_duration_ms) in seconds
    """

    clean_intervals: torch.FloatTensor
    intervals: torch.FloatTensor
    probs: torch.FloatTensor

    def clean_gpu(self):
        del self.clean_intervals
        del self.intervals
        del self.probs


def remove_small_speech_intervals(
    intervals: torch.tensor, min_speech_duration_samples,
) -> torch.tensor:
    """Remove speech segments (< min_speech_duration_samples)  to speech segments
    Example: speech
    """
    intervals = intervals.view(-1)
    intrval_diffs = torch.diff(intervals)
    speech_intervals = intrval_diffs[0: len(intrval_diffs): 2]
    speech_mask = speech_intervals >= min_speech_duration_samples
    mask = speech_mask.view(-1, 1).repeat(1, 2).reshape(-1)
    intervals = intervals[mask].view(-1, 2)
    return intervals


def remove_silence_intervals(
    intervals: torch.tensor,
    min_silence_duration_samples,
) -> torch.tensor:
    """Merging slilecne segments (< min_silence_duration_samples)  to speech segments
    Example: speech
    """
    device = intervals.device
    # remove silence intervals
    intervals = intervals.view(-1)
    intrval_diffs = torch.diff(intervals)
    silence_intervals = intrval_diffs[1: len(intrval_diffs): 2]
    silence_mask = silence_intervals >= min_silence_duration_samples
    mask = silence_mask.view(-1, 1).repeat(1, 2).reshape(-1)
    mask = torch.cat([torch.tensor([True], device=device),
                     mask, torch.tensor([True], device=device)], dim=0)
    intervals = intervals[mask].view(-1, 2)
    return intervals


@torch.no_grad()
def quran_split_by_silence(
    wav: torch.FloatTensor,
    sample_rate=16000,
    model=load_vad_model(),
    window_size_samples=1536,
    threshold=0.3,
    min_silence_duration_ms=30,
    min_speech_duration_ms=30,
    pad_duration_ms=30,
    device=torch.device('cpu'),
    return_probabilities=False,
) -> SegmentationOutput:
    """Extractes Speech Intervals from input `wav`

    Extractes speech Intervals using https://github.com/snakers4/silero-vad/tree/v4.0stable v4.0 model
    The model is located in: https://github.com/snakers4/silero-vad/blob/v4.0stable/files/silero_vad.jit
    with winodw size 1536

    Args:
        wav (torch.FloatTensor): Input audio waveform as a PyTorch tensor.
        sample_rate (int, optional): Sampling rate of the audio. Defaults to 16000.
        model: (torch.nn.Module): silero VAD model to use for segmentation. Defaults is  snakers4/silero-vad v4.0 model.
        window_size_samples (int, optional):  Window size in samples used for VAD processing. Defaults to 1536.
        threshold (float, optional): Probability threshold for speech detection. Defaults to 0.3.
        min_silence_duration_ms (int, optional): Minimum duration of silence in milliseconds to be considered a segment boundary. Defaults to 30.
        min_speech_duration_ms (int, optional): The Minimum speech duration in milliseconds will be removed and marked as silence.
        pad_duration_ms (int, optional): Duration of padding in milliseconds to add to the beginning and end of each speech segment. Defaults to 30.
        device (torch.device, optional): Device to run the model on (e.g., 'cpu' or 'cuda'). Defaults to torch.device('cpu').
        return_probabilities (bool, optional): If True, return the average probabilities for each speech segment. Defaults to False.

    Returns:
        SegmentationOutput: with:
            * clean_intervlas (torch.FloatTensor): the speech intervlas after merging short silecne intervals (< min_silence_duration_ms) in seconds
            * intervlas (torch.FloatTensor): the actual speech intervlas of the model without any cleaning (in seconds)
            * pobs: (torch.FloatTensor): the average probabilty for every speech segment for `intervals` without cleaning. Same shape as `intervlas`.
                If `return_probabilities` is `True` else return `None`
    """
    assert isinstance(wav, torch.Tensor), (
        f'`wav` should be tensor got `{type(wav)}`')
    # paddign wav
    pad_len = window_size_samples - (wav.shape[0] % window_size_samples)
    wav_input = torch.nn.functional.pad(
        input=wav, pad=(0, pad_len), mode='constant', value=0)
    wav_input = wav_input.view(-1, window_size_samples)

    # inference step
    model.reset_states()
    model.to(device)
    model.eval()
    wav_input = wav_input.to(device)

    probs = torch.zeros(wav_input.shape[0], device=device)
    for idx, wav_window in enumerate(wav_input):
        probs[idx] = model(wav_window, sample_rate)

    # extracting intervals
    diffs = torch.diff(probs > threshold,
                       prepend=torch.tensor([False], device=device))
    intervals = torch.arange(probs.shape[0], device=device)[diffs]

    if intervals.shape[0] == 0:
        raise NoSpeechIntervals(
            'No speech intervals found. May be `threshold` is too high or the input `wav` is complete silence')

    # no silence at the end of the track
    if intervals.shape[0] % 2 != 0:
        intervals = torch.cat(
            [intervals, torch.tensor([float('inf')], device=device)])

    # scaling to frames instead of mulitple of window_size_samples
    intervals = intervals.view(-1, 2) * window_size_samples

    # remove small silence duration
    min_silence_duration_samples = int(
        min_silence_duration_ms * sample_rate / 1000)
    clean_intervals = remove_silence_intervals(
        intervals, min_silence_duration_samples)

    # remove small speech durations
    min_speech_duration_samples = int(
        min_speech_duration_ms * sample_rate / 1000)
    clean_intervals = remove_small_speech_intervals(
        clean_intervals, min_speech_duration_samples)

    if clean_intervals.shape[0] == 0:
        raise TooHighMinSpeechDuration(
            'No speech intervals found Please Lower the `min_speech_duration_ms`')

    # add padding
    padding_samples = int(pad_duration_ms * sample_rate / 1000)
    padding = torch.ones_like(clean_intervals) * padding_samples
    padding[:, 0] *= -1
    clean_intervals += padding
    if clean_intervals[0, 0] < 0:
        clean_intervals[0, 0] = 0

    # Extracting probability for each interval
    if return_probabilities:
        start = 0
        intervals_probs = []
        for idx in clean_intervals.view(-1,).to(torch.long) // window_size_samples:
            if idx < 0:
                idx = probs.shape[0]
            p = probs[start: idx].mean().item()
            intervals_probs.append(p)
            start = idx
        if clean_intervals[-1, -1] != float('inf'):
            intervals_probs.append(probs[start:].mean().item())
        intervals_probs = torch.tensor(intervals_probs)

    # convert it to seconds
    clean_intervals = clean_intervals / sample_rate
    intervals = intervals / sample_rate

    return SegmentationOutput(
        clean_intervals=clean_intervals,
        intervals=intervals.cpu(),
        probs=intervals_probs.cpu() if return_probabilities else None,
    )


@torch.no_grad()
def quran_split_by_silence_batch(
    waves: list[torch.FloatTensor],
    sample_rate=16000,
    model=load_vad_model(),
    window_size_samples=1536,
    threshold=0.3,
    min_silence_duration_ms=30,
    min_speech_duration_ms=30,
    pad_duration_ms=30,
    device=torch.device('cpu'),
    return_probabilities=False,
) -> list[SegmentationOutput]:
    """Extractes Speech Intervals from input `wav`

    Extractes speech Intervals using https://github.com/snakers4/silero-vad/tree/v4.0stable v4.0 model
    The model is located in: https://github.com/snakers4/silero-vad/blob/v4.0stable/files/silero_vad.jit
    with winodw size 1536

    Args:
        waves (list[torch.FloatTensor]): Input audio waveform as a list PyTorch tensors.
        sample_rate (int, optional): Sampling rate of the audio. Defaults to 16000.
        model: (torch.nn.Module): silero VAD model to use for segmentation. Defaults is  snakers4/silero-vad v4.0 model.
        window_size_samples (int, optional):  Window size in samples used for VAD processing. Defaults to 1536.
        threshold (float, optional): Probability threshold for speech detection. Defaults to 0.3.
        min_silence_duration_ms (int, optional): Minimum duration of silence in milliseconds to be considered a segment boundary. Defaults to 30.
        min_speech_duration_ms (int, optional): The Minimum speech duration in milliseconds will be removed and marked as silence.
        pad_duration_ms (int, optional): Duration of padding in milliseconds to add to the beginning and end of each speech segment. Defaults to 30.
        device (torch.device, optional): Device to run the model on (e.g., 'cpu' or 'cuda'). Defaults to torch.device('cpu').
        return_probabilities (bool, optional): If True, return the average probabilities for each speech segment. Defaults to False.

    Returns:
        list[SegmentationOutput]: with:
            * clean_intervlas (torch.FloatTensor): the speech intervlas after merging short silecne intervals (< min_silence_duration_ms) in seconds
            * intervlas (torch.FloatTensor): the actual speech intervlas of the model without any cleaning (in seconds)
            * pobs: (torch.FloatTensor): the average probabilty for every speech segment for `intervals` without cleaning. Same shape as `intervlas`.
                If `return_probabilities` is `True` else return `None`
    """
    # assert isinstance(wav, torch.Tensor), (
    #     f'`wav` should be tensor got `{type(wav)}`')

    # collecting waves to be a single batch
    lengths = torch.tensor([len(w) for w in waves])
    # of shape batchsize, max(len)
    padded_waves = pad_sequence(waves, batch_first=True, padding_value=0)

    # paddign wav
    pad_len = window_size_samples - \
        (padded_waves.shape[1] % window_size_samples)
    # batch_size, max_len
    wav_input = torch.nn.functional.pad(
        input=padded_waves, pad=(0, pad_len), mode='constant', value=0)
    # batch_sisze, num_frames, window_size_samples
    batch_size, _ = wav_input.shape
    wav_input = wav_input.view(batch_size, -1, window_size_samples)

    # # inference step
    model.reset_states()
    model.to(device)
    model.eval()
    wav_input = wav_input.to(device)

    # batch_size, num_frames, windw_size_samples
    batch_size, num_frames, _ = wav_input.shape
    probs = torch.zeros(batch_size, num_frames, device=device)
    for idx in range(num_frames):
        wav_window = wav_input[:, idx, :]  # batch_size, window_size_samples
        probs[:, idx] = model(wav_window, sample_rate).squeeze(dim=-1)

    # # extracting intervals
    outputs = []
    for idx in range(batch_size):
        diffs = torch.diff(
            probs[idx] > threshold,
            prepend=torch.tensor([False], device=device))
        intervals = torch.arange(num_frames, device=device)[diffs]

        if intervals.shape[0] == 0:
            raise NoSpeechIntervals(
                'No speech intervals found. May be `threshold` is too high or the input `wav` is complete silence')

        # no silence at the end of the track
        if intervals.shape[0] % 2 != 0:
            intervals = torch.cat(
                [intervals, torch.tensor([float('inf')], device=device)])

        # scaling to frames instead of mulitple of window_size_samples
        intervals = intervals.view(-1, 2) * window_size_samples

        # remove small silence duration
        min_silence_duration_samples = int(
            min_silence_duration_ms * sample_rate / 1000)
        clean_intervals = remove_silence_intervals(
            intervals, min_silence_duration_samples)

        # remove small speech durations
        min_speech_duration_samples = int(
            min_speech_duration_ms * sample_rate / 1000)
        clean_intervals = remove_small_speech_intervals(
            clean_intervals, min_speech_duration_samples)

        if clean_intervals.shape[0] == 0:
            raise TooHighMinSpeechDuration(
                'No speech intervals found Please Lower the `min_speech_duration_ms`')

        # add padding
        padding_samples = int(pad_duration_ms * sample_rate / 1000)
        padding = torch.ones_like(clean_intervals) * padding_samples
        padding[:, 0] *= -1
        clean_intervals += padding
        if clean_intervals[0, 0] < 0:
            clean_intervals[0, 0] = 0

        # Extracting probability for each interval
        if return_probabilities:
            start = 0
            intervals_probs = []
            for idx in clean_intervals.view(-1,).to(torch.long) // window_size_samples:
                if idx < 0:
                    idx = probs[idx].shape[0]
                p = probs[idx][start: idx].mean().item()
                intervals_probs.append(p)
                start = idx
            if clean_intervals[-1, -1] != float('inf'):
                intervals_probs.append(probs[start:].mean().item())
            intervals_probs = torch.tensor(intervals_probs)

        # convert it to seconds
        clean_intervals = clean_intervals / sample_rate
        intervals = intervals / sample_rate

        outputs.append(SegmentationOutput(
            clean_intervals=clean_intervals,
            intervals=intervals,
            probs=intervals_probs if return_probabilities else None,
        ))

    return outputs
