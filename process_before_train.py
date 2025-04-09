# this code with build on top of this colab notebook: https://colab.research.google.com/drive/1q3q4xkFNhYpYfrcSENR6LrS2eJdNKg0X?usp=sharing

from typing import Any
from dataclasses import dataclass

from transformers import AutoFeatureExtractor, Wav2Vec2BertProcessor
from datasets import Features, Audio, Array2D, Value, Dataset, DatasetDict

import numpy as np

DS_FEATURES_TRAIN = Features({
    'aya_name': Value(dtype='string'),
    'reciter_name': Value(dtype='string'),
    'recitation_id': Value(dtype='int32'),
    'url': Value(dtype='string'),
    'audio': Audio(sampling_rate=16000, decode=False),
    'duration': Value(dtype='float32'),
    'speed': Value(dtype='float32'),
    'speech_intervals': Array2D(shape=(None, 2), dtype="float32"),
    'is_interval_complete': Value(dtype='bool'),
    'is_augmented': Value(dtype='bool'),
    'input_features': Array2D(shape=(None, 2), dtype="float32"),
    'attention_mask': Array2D(shape=(None, 1), dtype="int32"),
    'labels': Array2D(shape=(None, 1), dtype="int32"),
})


@dataclass
class TruncateOutput:
    audio: list[dict[str, Any]]
    speech_intervals_sec: list[np.ndarray]
    speech_intervals_samples: list[np.ndarray]

    """
    audio: list({'array': np.ndarray, 'sampling_rate': int})
    """


def truncate(
    wav: np.ndarray,
    speech_intervals_sec: np.ndarray,
    sampling_rate=16000,
    truncate_window_overlap_length=16000,
    max_size_samples=480000,
) -> TruncateOutput:
    """Moving winodw truncatation arlogrith where the window size is `max_size_samples`
    Note:
    * speech_inatevals are inclusive EX intv = [1, 5] sor [1, 2, 3, 4, ,5] are speech
    """

    assert max_size_samples > truncate_window_overlap_length, '`max_size_samples` should be > `truncate_window_overlap_length` '
    speech_intervals_samples = np.array(speech_intervals_sec) * sampling_rate
    speech_intervals_samples = speech_intervals_samples.astype(np.longlong)

    # edge case last interval end should be < total waves length
    if speech_intervals_samples.shape[0] > 0:
        if speech_intervals_samples[-1][1] >= len(wav):
            speech_intervals_samples[-1][1] = len(wav) - 1

    out = TruncateOutput([], [], [])
    overlap = truncate_window_overlap_length
    window = max_size_samples
    step = window - overlap
    num_items = int(np.ceil((len(wav) - window) / (window - overlap))) + 1
    if len(wav) == 0:
        num_items = 0

    start = 0
    intv_start_idx = 0
    for idx in range(num_items):
        end = start + window
        out.audio.append(
            {'array': wav[start: end],
             'sampling_rate': sampling_rate})

        chosen_idx = intv_start_idx
        frgmented_intv = None
        intv_idx = 0
        for intv_idx in range(intv_start_idx, len(speech_intervals_samples)):
            # iterval end is smaller than the winodw size
            if speech_intervals_samples[intv_idx][1] < end - overlap:
                chosen_idx += 1

            elif speech_intervals_samples[intv_idx][0] < end:
                frgmented_intv = np.zeros(2, dtype=np.longlong)
                frgmented_intv[0] = speech_intervals_samples[intv_idx][0]
                frgmented_intv[1] = min(
                    end - 1, int(speech_intervals_samples[intv_idx][1]))

                speech_intervals_samples[intv_idx][0] = max(
                    end - overlap, int(speech_intervals_samples[intv_idx][0]))
                break
            else:
                break

        if frgmented_intv is None:
            out.speech_intervals_samples.append(
                speech_intervals_samples[intv_start_idx: chosen_idx])
        else:
            out.speech_intervals_samples.append(
                np.concatenate(
                    (speech_intervals_samples[intv_start_idx: chosen_idx], np.expand_dims(frgmented_intv, 0)), axis=0),
            )

        # making intervals relative to each audio frame not the entire audio
        out.speech_intervals_samples[-1] -= start

        # end of the loop
        out.speech_intervals_sec.append(
            out.speech_intervals_samples[-1] / sampling_rate)
        start += step
        intv_start_idx = intv_idx

    assert (len(out.audio) == len(out.speech_intervals_samples))

    return out


def calculate_overlap(
    intervals: np.ndarray,
    window_start: int,
    window_end: int,
) -> int:
    """Calcualutes the overlap between window and speech_intervals
    Args:
        intervals (np.ndarray): intervals are 2D array with eatch row represnts 
            (intervals_start, intervals_end).
            Note: the interval_end are inclusive unlike python indexing

    Returns:
        the overlap between the winodw and the intervals:
        * as integer > 0 if there exisits an overlap
        * 0 of ther is no overlap
    """
    start = np.empty_like(intervals)
    start[:, 0] = window_start
    start[:, 1] = intervals[:, 0]
    start = start.max(axis=1)

    end = np.empty_like(intervals)
    end[:, 0] = window_end
    end[:, 1] = intervals[:, 1] + 1
    end = end.min(axis=1)

    overlap = end - start
    return overlap[overlap > 0].sum()


def calc_frames(L, W=400, H=160, S=2):
    return max(0, int(1 + np.floor((L - W) / H)) // S)


def annotate(
    wav: np.ndarray,
    speech_intervals_samples: np.ndarray,
    attention_mask: np.ndarray,
    window_length_samples=400,
    hop_length_samples=160,
    stride=2,
    speech_label=1,
    silence_label=0,
    ignored_idx=-100,
) -> np.ndarray:
    """Annotates frame level as a `speech`, `silence` and `ignored` if attention_mask==0
    Args:
        speech_intervals_samples (np.narray): 2D array and earch row indicates the
            start and the end indices of speech intervals:
            NOTE: both start and end are inclusive unlike python indexing
        attention_mask (np.narrayl): a single dimention vector with type np.int64 with 1s ns 0s.
            Note: len(attention_mask) >= floor(floor(len(wav) - window_size_samples) / hop_length_samples) + 1) / stride)
    Returns the labels as 1s and 0s and ignored index for masked inputs (i.e mask=0) as single dimention np array
    """
    num_frames = attention_mask.sum()
    labels = np.ones(attention_mask.shape, dtype=np.longlong) * ignored_idx
    window = window_length_samples + (stride - 1) * hop_length_samples
    start = 0
    end = 0
    for frame_idx in range(num_frames):
        end = start + window
        overlap = calculate_overlap(speech_intervals_samples, start, end)
        if overlap / window > 0.5:
            labels[frame_idx] = speech_label
        else:
            labels[frame_idx] = silence_label

        start += stride * hop_length_samples

    # checkng
    max_frames = calc_frames(end, window_length_samples,
                             hop_length_samples, stride)
    assert max_frames == num_frames, 'There exists missing frames'

    return labels


def extract_features_and_labels(
    batch: dict[str, list[Any]],
    processor: Wav2Vec2BertProcessor,
    min_size_samples=32000,
    max_size_samples=480000,
    truncate_window_overlap_length=16000,
    window_length_samples=400,
    hop_length_samples=160,
    sampling_rate=16000,
    stride=2,
    speech_label=1,
    silence_label=0,
    ignored_idx=-100,
    model_id='facebook/w2v-bert-2.0',
) -> dict[str, list[Any]]:

    # truncate samples
    speech_intervals_samples = []
    new_batch = {'audio': [], 'speech_intervals': []}
    for idx in range(len(batch['audio'])):
        trunc_outs = truncate(
            batch['audio'][idx]['array'],
            batch['speech_intervals'][idx],
            sampling_rate=batch['audio'][idx]['sampling_rate'],
            truncate_window_overlap_length=truncate_window_overlap_length,
            max_size_samples=max_size_samples,
        )
        new_batch['audio'] += trunc_outs.audio
        new_batch['speech_intervals'] += trunc_outs.speech_intervals_sec
        speech_intervals_samples += trunc_outs.speech_intervals_samples
    batch['audio'] = new_batch['audio']
    batch['speech_intervals'] = new_batch['speech_intervals']

    # remove short samples < min_size_samples
    to_del_ids = []
    for idx in range(len(batch['audio'])):
        if len(batch['audio'][idx]['array']) < min_size_samples:
            to_del_ids.append(idx)
    # avoid index shefting (i.e remove woring index)
    for idx in sorted(to_del_ids, reverse=True):
        del speech_intervals_samples[idx]
        for key in batch:
            del batch[key][idx]

    assert len(speech_intervals_samples) == len(batch['audio'])

    # extract features
    # taken from https://github.com/huggingface/transformers/blob/main/src/transformers/audio_utils.py#L589
    # the total number of max frames will be max_frames / stride
    max_frames = int(
        1 + np.floor((max_size_samples - window_length_samples) / hop_length_samples))
    processor: Wav2Vec2BertProcessor = AutoFeatureExtractor.from_pretrained(
        model_id)
    waves = [batch['audio'][idx]['array']
             for idx in range(len(batch['audio']))]
    model_inputs = processor(
        waves,
        sampling_rate=16000,
        return_tensors="np",  # TODO:` numpy or pt
        max_length=max_frames,
        padding='max_length',
    )
    batch['input_features'] = model_inputs['input_features']
    batch['attention_mask'] = model_inputs['attention_mask']

    # get labels
    batch['labels'] = []
    for idx in range(len(batch['audio'])):
        labels = annotate(
            batch['audio'][idx]['array'],
            speech_intervals_samples[idx],
            batch['attention_mask'][idx],
            window_length_samples=window_length_samples,
            hop_length_samples=hop_length_samples,
            stride=stride,
            speech_label=speech_label,
            silence_label=silence_label,
            ignored_idx=ignored_idx,
        )
        batch['labels'].append(labels)

    return batch


def extract_features_for_ds(
    ds: Dataset | DatasetDict,
    batch_size=32,
    num_proc=128,
    **kwargs,
) -> Dataset | DatasetDict:
    ds = ds.map(
        extract_features_and_labels,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        features=DS_FEATURES_TRAIN,
        fn_kwargs=kwargs,
    )
    return ds
