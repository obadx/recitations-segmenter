from pathlib import Path
from dataclasses import dataclass

import torchaudio
import torch

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


@dataclass
class SegmentationOutput:
    clean_intervals: torch.FloatTensor
    intervals: torch.FloatTensor
    probs: torch.FloatTensor

    """
    Attrubutes:
        intervlas (torch.FloatTensor): the actual speech intervlas of the model without any cleaning (in seconds)
        pobs: (torch.FloatTensor): the average probabilty for every speech segment for `intervals` without cleaning. Same shape as `intervlas`
        clean_intervlas (torch.FloatTensor): the speech intervlas after merging short silecne intervals (< min_silence_duration_ms) in seconds
    """


def remove_silence_intervals(
    intervals: torch.tensor,
    min_silence_duration_samples,
) -> torch.tensor:
    """Merging slilecne segments (< min_silence_duration_samples)  to speech segments
    Example: speech
    """
    # remove silence intervals
    intervals = intervals.view(-1)
    intrval_diffs = torch.diff(intervals)
    silence_intervals = intrval_diffs[1: len(intrval_diffs): 2]
    silence_mask = silence_intervals >= min_silence_duration_samples
    mask = silence_mask.view(-1, 1).repeat(1, 2).reshape(-1)
    mask = torch.cat([torch.tensor([True]), mask, torch.tensor([True])], dim=0)
    intervals = intervals[mask].view(-1, 2)
    return intervals


@torch.no_grad()
def quran_split_by_silence(
    wav: torch.FloatTensor,
    sample_rate=16000,
    model=init_jit_model(SILERO_VAD_PATH),
    window_size_samples=1536,
    threshold=0.3,
    min_silence_duration_ms=30,
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

    probs = []
    for wav_window in wav_input:
        probs.append(model(wav_window, sample_rate).cpu().item())
    probs = torch.tensor(probs)

    # extracting intervals
    diffs = torch.diff(probs > threshold, prepend=torch.tensor([False]))
    intervals = torch.arange(probs.shape[0], device=device)[diffs]

    # no silence at the end of the track
    if intervals.shape[0] % 2 != 0:
        intervals = torch.cat([intervals, torch.tensor([float('inf')])])

    # scaling to frames instead of mulitple of window_size_samples
    intervals = intervals.view(-1, 2) * window_size_samples

    # remove small silence duration
    min_silence_duration_samples = int(
        min_silence_duration_ms * sample_rate / 1000)
    clean_intervals = remove_silence_intervals(
        intervals, min_silence_duration_samples)

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
