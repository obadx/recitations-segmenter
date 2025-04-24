from dataclasses import dataclass
from typing import Sequence

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from transformers.models.wav2vec2_bert import Wav2Vec2BertForAudioFrameClassification
from transformers import Wav2Vec2BertProcessor
from tqdm import tqdm

from .train.augment import calc_frames
# TODO:
# batchify
# inferece
# augmenting results


class NoSpeechIntervals(Exception):
    pass


class TooHighMinSpeechDuration(Exception):
    pass


@dataclass
class W2vBSegmentationOutput:
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


@dataclass
class WavInfo:
    wav_len: int
    batch_start: int  # inclusive
    batch_end: int  # execlusive
    idx_in_batch_start: int  # inclusive
    idx_in_batch_end: int  # execlusive


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


def extract_intervals(
    logits: torch.Tensor,
    min_silence_duration_ms=30,
    min_speech_duration_ms=30,
    pad_duration_ms=30,
    speech_label=1,
    silence_label=0,
    sample_rate=16000,
    return_probabilities=False,

) -> W2vBSegmentationOutput:
    device = logits.device
    labels = logits.argmax(dim=-1)
    # TODO: select best probabilities only
    probs = torch.nn.functional.softmax(logits)
    window_size_samples = None

    # extracting intervals
    # TODO: totally different algorithm
    diffs = torch.diff(labels == speech_label,
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

    # TODO: configre device
    return W2vBSegmentationOutput(
        clean_intervals=clean_intervals,
        intervals=intervals.cpu(),
        probs=intervals_probs.cpu() if return_probabilities else None,
    )


def batchify_input(
    waves: list[torch.FloatTensor],
    max_len_samples: int,
    max_batch_size=64,
) -> tuple[list[WavInfo], Sequence[torch.FloatTensor]]:
    """
    Spliting input waves into batches to utlize GPU memory most at inference
    """
    batches = []
    wav_infos = []
    for wav in waves:
        assert len(wav) > 0, 'wav length should be > 0 got zerolength tensor'
        wav = wav.to('cpu')  # insure that every wav is on cpu
        pad = torch.zeros(
            max_len_samples - len(wav) % max_len_samples if len(wav) % max_len_samples != 0 else 0)
        padded_wav = torch.cat((wav, pad), dim=0)
        idx_start = len(batches) % max_batch_size
        batch_start = len(batches) // max_batch_size

        wav_chunks = list(padded_wav.split(max_len_samples))
        batches += wav_chunks

        idx_end = len(batches) % max_batch_size
        batch_end = (len(batches) - 1) // max_batch_size + 1

        # the case that the len of new wav chunck is the same as max_batch_size
        # and idx_start == 0 (it will bug if idx_end is 0 it has to be max_batch_size
        if idx_start == 0 and idx_end == 0:
            idx_end = max_batch_size

        wav_infos.append(WavInfo(
            wav_len=len(wav),
            batch_start=batch_start,
            idx_in_batch_start=idx_start,
            batch_end=batch_end,
            idx_in_batch_end=idx_end,
        ))

    if batches:
        batches = torch.stack(batches, dim=0).split(max_batch_size, dim=0)
    return wav_infos, batches


def collect_results(
    wav_infos: list[WavInfo],
    batches_logits: Sequence[torch.FloatTensor],
    processor_window=400,
    prcoessor_hop=160,
    processor_stride=2,
):
    out_logits: list[torch.FloatTensor] = []
    for wav_info in wav_infos:
        start = wav_info.idx_in_batch_start
        logits: list[torch.FloatTensor] = []
        loop_len = wav_info.batch_end - wav_info.batch_start
        # every batches_logits[idx] is of shape batch_size, sequence_len, 2
        for idx in range(loop_len):
            # last loop
            if (loop_len - 1) == idx:
                selected_logits = batches_logits[wav_info.batch_start +
                                                 idx][start: wav_info.idx_in_batch_end]
            else:
                selected_logits = batches_logits[wav_info.batch_start + idx][start:]

            logits += [t.squeeze(0) for t in selected_logits.split(1)]
            start = 0

        # aggrecating results after loop
        logits = torch.cat(logits, dim=0)

        # removing extra outputs from output due to padding for batching
        num_frames = calc_frames(
            wav_info.wav_len,
            W=processor_window,
            H=prcoessor_hop,
            S=processor_stride)
        logits = logits[: num_frames]
        out_logits.append(logits)

    return out_logits


@torch.no_grad()
def segment_recitations(
    waves: list[torch.FloatTensor],
    model: Wav2Vec2BertForAudioFrameClassification,
    processor: Wav2Vec2BertProcessor,
    batch_size=64,
    sample_rate=16000,
    processor_window=400,
    processor_hop=160,
    processor_stride=2,
    max_duration_ms=20000,
    min_silence_duration_ms=30,
    min_speech_duration_ms=30,
    pad_duration_ms=30,
    speech_label=1,
    silence_label=0,
    device=torch.device('cpu'),
    dtype=torch.bfloat16,
    return_probabilities=False,
) -> list[W2vBSegmentationOutput]:
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

    # TODO: assert if the max_duration_ms does not produce %2 samples

    assert processor_hop == 160, 'This a pre-defined  value for the Wav2Vec2BertProcessor processor Do not change it'
    assert processor_window == 400, 'This a pre-defined  value for the Wav2Vec2BertProcessor processor Do not change it'
    assert processor_stride == 2, 'This a pre-defined  value for the Wav2Vec2BertProcessor processor Do not change it'
    assert sample_rate == 16000, 'This a pre-defined  value for the Wav2Vec2BertProcessor processor Do not change it'
    assert max_duration_ms <= 20000 and max_duration_ms >= 2, 'We fine-tune W2vecBert on max duration of 20 secnds during training'

    max_duration_sampels = int(max_duration_ms * sample_rate / 1000)

    max_frames = int(
        1 + np.floor((max_duration_sampels - processor_window) / processor_hop))

    # conveting input to batches
    wav_infos, wav_batches = batchify_input(
        waves,
        int(max_duration_ms * sample_rate),
        batch_size,
    )

    # Run infernce on batches
    batches_logits: list[torch.FloatTensor] = []
    batches_attention_mask: list[torch.LongTensor] = []
    for batch in tqdm(wav_batches):
        model_inputs = processor(
            batch,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        model_inputs = model_inputs.to(dtype).to(device)
        logits = model(
            **model_inputs
        )
        # Going back to cpu
        batches_logits.append(logits.cpu().to(torch.float32))

    # Aggeregate batches
    collected_logits: list[torch.FloatTensor] = collect_results(
        wav_infos,
        batches_logits,
        processor_window=processor_window,
        prcoessor_hop=processor_hop,
        processor_stride=processor_stride
    )

    # Extract speech intervals for every input
    outputs = list[W2vBSegmentationOutput] = []
    for logits in collected_logits:
        out = extract_intervals(
            logits,
            min_silence_duration_ms=min_silence_duration_ms,
            min_speech_duration_ms=min_speech_duration_ms,
            pad_duration_ms=pad_duration_ms,
            speech_label=speech_label,
            silence_label=silence_label,
            sample_rate=sample_rate,
            return_probabilities=return_probabilities,

        )
        outputs.append(out)

    return outputs
