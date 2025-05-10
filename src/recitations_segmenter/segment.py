from dataclasses import dataclass
from typing import Sequence
import warnings

import torch
import torchaudio
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from transformers.models.wav2vec2_bert import Wav2Vec2BertForAudioFrameClassification
from transformers import Wav2Vec2BertProcessor
from tqdm import tqdm


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


def is_dtype_supported(dtype, device):
    """
    Checks if the specified torch dtype is supported by the currently available GPU.

    Args:
        dtype (torch.dtype): The data type to check for GPU support.

    Returns:
        bool: True if the dtype is supported by the GPU, False otherwise.
    """
    if not torch.cuda.is_available():
        return True
    try:
        # Attempt to create an empty tensor of the given dtype on the GPU
        torch.empty(0, dtype=dtype, device=device)
        return True
    except RuntimeError as e:
        raise RuntimeError(f'{dtype} is not supported on this GPU as: {e}')


class NoSpeechIntervals(Exception):
    pass


class TooHighMinSpeechDuration(Exception):
    pass


@dataclass
class W2vBSegmentationOutput:
    """
    Attrubutes:
        - clean_speech_intervals: Tensor of shape (N, 2) containing speech intervals after filtering.
                Format: `[[speech_start, speech_end], [speech_start, speech_end], ...]` in samples.
        - speech_intervals: Tensor of shape (N, 2) containing raw speech intervals before filtering.
                Format: `[[speech_start, speech_end], [speech_start, speech_end], ...]` in samples.
        - probs: Class probabilities (None if not requested)
        - is_complete: Whether audio processing completed normally
    """

    clean_speech_intervals: torch.FloatTensor
    speech_intervals: torch.FloatTensor
    probs: torch.FloatTensor
    is_complete: bool

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
    speech_intervals: torch.LongTensor,
    min_speech_duration_samples: int,
) -> torch.LongTensor:
    """Removes speech segments shorter than the specified minimum duration.

    Args:
        speech_intervals: Tensor of shape (N, 2) containing speech intervals.
            Format: `[[speech_start, speech_end], [speech_start, speech_end], ...]` in samples.
        min_speech_duration_samples: Minimum allowed duration (in samples) for a speech segment.
            speech intervals durations < `min_speech_duration_ms` will be removed

    Returns:
        torch.Tensor: Filtered speech intervals tensor of shape (M, 2), where M <= N.
    """
    intervals = speech_intervals.view(-1)
    interval_diffs = torch.diff(intervals)
    speech_intervals = interval_diffs[0: len(interval_diffs): 2]
    speech_mask = speech_intervals >= min_speech_duration_samples
    mask = speech_mask.view(-1, 1).repeat(1, 2).reshape(-1)
    intervals = intervals[mask].view(-1, 2)
    return intervals


def remove_silence_intervals(
    intervals: torch.tensor,
    min_silence_duration_samples,
) -> torch.tensor:
    """Merges adjacent speech segments if the silence between them is shorter than the specified minimum.

    Args:
        speech_intervals: Tensor of shape (N, 2) containing speech intervals.
            Format: `[[speech_start, speech_end], [speech_start, speech_end], ...]` in samples.
        min_silence_duration_samples: Minimum allowed silence duration (in samples).
            silence durations < `min_silence_duration_ms` will be merged into speech durations

    Returns:
        torch.Tensor: Filtered speech intervals tensor of shape (M, 2), where M <= N.

    Note:
        - Input intervals are expected to alternate between speech and silence segments.
        - Only operates on silence segments (odd-indexed intervals when flattened).
        - Preserves leading/trailing silences by default.
    """
    device = intervals.device
    # remove silence intervals
    intervals = intervals.view(-1)
    interval_diffs = torch.diff(intervals)
    silence_intervals = interval_diffs[1: len(interval_diffs): 2]
    silence_mask = silence_intervals >= min_silence_duration_samples
    mask = silence_mask.view(-1, 1).repeat(1, 2).reshape(-1)
    mask = torch.cat([torch.tensor([True], device=device),
                     mask, torch.tensor([True], device=device)], dim=0)
    intervals = intervals[mask].view(-1, 2)
    return intervals


# TODO:
# * add return prbabilities
def extract_speech_intervals(
    logits: torch.Tensor,
    time_stamps: torch.LongTensor,
    min_silence_duration_ms=30,
    min_speech_duration_ms=30,
    pad_duration_ms=30,
    speech_label=1,
    silence_label=0,
    sample_rate=16000,
    hop=160,
    stride=2,
    return_probabilities=False,
    return_seconds=False,
    wav_index=None,
) -> W2vBSegmentationOutput:
    """Extracts and processes speech intervals from model logits.

    Args:
        logits: Model output tensor of shape (T, num_classes)
        time_stamps: Tensor of shape (T,) containing frame timestamps in samples
        min_silence_duration_ms: Minimum silence duration (ms) between speech segments.
            silence durations < `min_silence_duration_ms` will be merged into speech durations
        min_speech_duration_ms: Minimum duration (ms) for a valid speech segment
            speech intervals durations < `min_speech_duration_ms` will be removed
        pad_duration_ms: Padding duration (ms) to add around speech segments
        speech_label: Class index representing speech
        silence_label: Class index representing silence
        sample_rate: Audio sample rate in Hz
        hop: Hop length of Wav2Vec2BertProcessor used in feature extraction
        stride: Stride factor of `Wav2Vec2BertProcessor` for timestamp calculation
        return_probabilities: Whether to return class probabilities
        return_seconds: Whether to return intervals in seconds instead of samples

    Returns:
        W2vBSegmentationOutput:
            - clean_speech_intervals: Tensor of shape (N, 2) containing speech intervals after filtering.
                Format: `[[speech_start, speech_end], [speech_start, speech_end], ...]` in samples.
            - speech_intervals: Tensor of shape (N, 2) containing raw speech intervals before filtering.
                Format: `[[speech_start, speech_end], [speech_start, speech_end], ...]` in samples.
            - probs: Class probabilities (None if not requested)
            - is_complete: Whether audio processing completed normally

    Raises:
        NoSpeechIntervals: If no speech segments are detected
        TooHighMinSpeechDuration: If filtering removes all speech segments

    Note:
        - Intervals are clamped to prevent negative starts or exceeding audio length
        - Final interval end is clamped to (audio_length + hop*stride) if not provided
    """
    min_silence_duration_samples = int(
        min_silence_duration_ms * sample_rate / 1000)

    min_speech_duration_samples = int(
        min_speech_duration_ms * sample_rate / 1000)

    is_complete = True
    labels = logits.argmax(dim=-1)
    # TODO: returning probabilities
    probs = torch.nn.functional.softmax(
        logits, dim=-1)[torch.arange(len(labels)), labels]

    # extracting intervals
    diffs = torch.diff(labels == speech_label,
                       prepend=torch.tensor([False]))
    intervals = time_stamps[diffs]

    if intervals.shape[0] == 0:
        raise NoSpeechIntervals(
            'No speech intervals found. May be `threshold` is too high or the input `wav` is complete silence')

    # no silence at the end of the track
    if intervals.shape[0] % 2 != 0:
        is_complete = False
        intervals = torch.cat(
            [intervals, time_stamps[-1:] + hop * stride])

    intervals = intervals.view(-1, 2)

    # remove small silence duration
    clean_intervals = remove_silence_intervals(
        intervals, min_silence_duration_samples)

    # remove small speech durations
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
    # avoiding negative samples
    clean_intervals[:, 0] = torch.clamp(clean_intervals[:, 0], min=0)

    # convert it to seconds
    if return_seconds:
        clean_intervals = clean_intervals / sample_rate
        intervals = intervals / sample_rate

    return W2vBSegmentationOutput(
        clean_speech_intervals=clean_intervals,
        speech_intervals=intervals.cpu(),
        probs=None,
        is_complete=is_complete,
    )


def batchify_input(
    waves: list[torch.FloatTensor],
    max_len_samples: int,
    max_batch_size=64,
) -> tuple[list[WavInfo], Sequence[torch.FloatTensor]]:
    """
    Spliting input waves into batches to utlize GPU memory most at inference
    """
    segments_len = 0
    for w in waves:
        assert w.device.type == 'cpu', 'All wav inputs has to be on `cpu`'
        assert len(w) > 0, 'wav length should be > 0 got zerolength tensor'
        segments_len += int(np.ceil(len(w) / max_len_samples))

    batches = torch.zeros(segments_len, max_len_samples, dtype=torch.float32)
    occupied_len = 0
    wav_infos = []
    for wav in waves:
        pad_len = (max_len_samples -
                   len(wav) % max_len_samples if len(wav) % max_len_samples != 0 else 0)
        padded_wav = torch.nn.functional.pad(wav, (0, pad_len))
        idx_start = occupied_len % max_batch_size
        batch_start = occupied_len // max_batch_size

        wav_chunks = padded_wav.view(-1, max_len_samples)
        batches[occupied_len: occupied_len +
                wav_chunks.shape[0], :] = wav_chunks
        occupied_len += wav_chunks.shape[0]

        idx_end = occupied_len % max_batch_size
        batch_end = (occupied_len - 1) // max_batch_size + 1

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

    if occupied_len:
        batches = batches.split(max_batch_size, dim=0)
    return wav_infos, batches


def collect_results(
    wav_infos: list[WavInfo],
    batches_logits: Sequence[torch.FloatTensor],
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

            logits.append(selected_logits)
            start = 0

        # aggrecating results after loop
        batch_size, seq_len, num_classes = logits[0].shape
        logits = torch.cat([l.view(-1, num_classes) for l in logits], dim=0)

        out_logits.append(logits)

    return out_logits


def generate_time_stamps(
    features_len,
    max_duration_samples=320000,
    max_featrues_len=998,
    window=400,
    hop=160,
    stride=2,
) -> torch.LongTensor:
    """Generate timestampls for the labels as every label represents its timestamps

    Generating timestampls for every label as input is split into multiple batches
    we have different time stamp for every label

    Ensuring that every segment with duration (max_duration_samples) begins with multiple of
    `max_duration_sample` thus representing the exact sample timeframe

    Example:
        time_stamps = generate_time_stamps(
            2* 2 + 1,
            max_duration_samples=1000,
            max_featrues_len=2,
            window=400,
            hop=160,
            stride=2,
        )

    time_stamps = [0, 320, 1000, 1320, 2000]
                   ^   ^     ^    ^
                   ^   ^     ^    ^     last seg
                 first seg  second seg

    """
    time_stamps = torch.arange(
        features_len, dtype=torch.long) * stride * hop

    for batch_idx in range(1, int(np.ceil(features_len / max_featrues_len))):
        idx = batch_idx * max_featrues_len
        time_stamps[idx:] += batch_idx * max_duration_samples - \
            time_stamps[idx] if idx < len(time_stamps) else 0

    return time_stamps


def check_devices(device1, device2) -> bool:
    devices = torch.device(device1), torch.device(device2)
    devices_ids = [0, 0]
    for idx, device in enumerate(devices):
        if device.index:
            devices_ids[idx] = device.index
    return (devices_ids[0] == devices_ids[1]) and (devices[0].type == devices[1].type)


@torch.no_grad()
def segment_recitations(
    waves: list[torch.FloatTensor],
    model: Wav2Vec2BertForAudioFrameClassification,
    processor: Wav2Vec2BertProcessor,
    batch_size=64,
    min_silence_duration_ms=30,
    min_speech_duration_ms=30,
    pad_duration_ms=30,
    device=torch.device('cpu'),
    dtype=torch.bfloat16,
    return_probabilities=False,
    return_seconds=False,
    sample_rate=16000,
    processor_window=400,
    processor_hop=160,
    processor_stride=2,
    max_duration_ms=19995,
    speech_label=1,
    silence_label=0,
) -> list[W2vBSegmentationOutput]:
    """Segment The Holy Quran rectiations into speech intervals based on وقف using Wav2Vec2Bert model.

    Args:
        waves: List of audio waveforms to process (each as FloatTensor)
        model: Loaded Wav2Vec2BertForAudioFrameClassification model
        processor: Wav2Vec2BertProcessor for feature extraction
        batch_size: Number of samples per batch
        sample_rate: Input audio sampling rate (must be 16000)
        processor_window: Processor window size (fixed at 400 samples)
        processor_hop: Processor hop length (fixed at 160 samples)
        processor_stride: Processor stride (fixed at 2)
        max_duration_ms: Maximum chunk duration in ms for processing (2-20000)
        min_silence_duration_ms: Minimum silence duration (ms) between speech segments.
            silence durations < `min_silence_duration_ms` will be merged into speech durations
        min_speech_duration_ms: Minimum duration (ms) for a valid speech segment
            speech intervals durations < `min_speech_duration_ms` will be removed
        pad_duration_ms: Padding to add around detected speech segments
        speech_label: Class index for speech segments
        silence_label: Class index for silence segments
        device: Torch device for inference
        dtype: Data type for model computation only. for post processing we use `torch.float32`
        return_probabilities: Whether to return class probabilities
        return_probabilities: Whether to return class probabilities
        return_probabilities: Whether to return seconds or samples for speech durations

    Returns:
        list[W2vBSegmentationOutput]:
        - clean_speech_intervals: Tensor of shape (N, 2) containing speech intervals after filtering.
            Format: `[[speech_start, speech_end], [speech_start, speech_end], ...]` in samples.
        - speech_intervals: Tensor of shape (N, 2) containing raw speech intervals before filtering.
            Format: `[[speech_start, speech_end], [speech_start, speech_end], ...]` in samples.
        - probs: Class probabilities (None if not requested)
        - is_complete: Whether audio processing completed normally

    Raises:
        NoSpeechIntervals: If no speech segments detected in one of the audio
        TooHighMinSpeechDuration: If filtering removes all speech segments

    Note:
        - Processes audio in chunks of max_duration_ms for GPU memory efficiency
        - Input waveforms are automatically padded and batched
        - Final interval end is clamped to (audio_length + hop*stride) if not provided
    """

    assert processor_hop == 160, 'This a pre-defined  value for the Wav2Vec2BertProcessor processor Do not change it'
    assert processor_window == 400, 'This a pre-defined  value for the Wav2Vec2BertProcessor processor Do not change it'
    assert processor_stride == 2, 'This a pre-defined  value for the Wav2Vec2BertProcessor processor Do not change it'
    assert sample_rate == 16000, 'This a pre-defined  value for the Wav2Vec2BertProcessor processor Do not change it'
    assert max_duration_ms <= 20000 and max_duration_ms >= 2, 'We fine-tune W2vecBert on max duration of 20 secnds during training'

    if max_duration_ms != 19995:
        warnings.warn(
            'To get best resutls we recommend using `max_duration_ms` with 19995', UserWarning)

    # checking if the dtype is supported by the GPU or not
    is_dtype_supported(dtype, device)

    model_device = next(model.parameters()).device
    assert check_devices(model_device, device), (
        f"Device mismatch!. Model Device: {model_device}, Device: {device}")
    assert next(model.parameters()
                ).dtype == dtype, "Model not in target dtype!"

    max_duration_sampels = int(max_duration_ms * sample_rate / 1000)

    # conveting input to batches
    wav_infos, wav_batches = batchify_input(
        waves,
        max_duration_sampels,
        batch_size,
    )

    # Run infernce on batches
    model.eval()
    batches_logits: list[torch.FloatTensor] = []
    for idx, batch in enumerate(tqdm(wav_batches)):
        model_inputs = processor(
            [b for b in batch],
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features = model_inputs['input_features'].to(device, dtype=dtype)
        attention_mask = model_inputs['attention_mask'].to(device, dtype=dtype)
        model_out = model(
            input_features=input_features,
            attention_mask=attention_mask,
            return_dict=False,
        )
        # Going back to cpu
        logits = model_out[0]
        batches_logits.append(logits.cpu().to(torch.float32))

    # Aggeregate batches
    collected_logits: list[torch.FloatTensor] = collect_results(
        wav_infos,
        batches_logits,
    )

    # Knowing the max feature lens for every input
    max_features_len = input_features.shape[1]

    # Extract speech intervals for every input
    outputs: list[W2vBSegmentationOutput] = []
    for logits in collected_logits:

        time_stamps = generate_time_stamps(
            len(logits),
            max_duration_samples=max_duration_sampels,
            max_featrues_len=max_features_len,
            window=processor_window,
            hop=processor_hop,
            stride=processor_stride,
        )
        out = extract_speech_intervals(
            logits,
            time_stamps,
            min_silence_duration_ms=min_silence_duration_ms,
            min_speech_duration_ms=min_speech_duration_ms,
            pad_duration_ms=pad_duration_ms,
            speech_label=speech_label,
            silence_label=silence_label,
            sample_rate=sample_rate,
            hop=processor_hop,
            stride=processor_stride,
            return_probabilities=return_probabilities,
            return_seconds=return_seconds,
        )
        outputs.append(out)

    return outputs
