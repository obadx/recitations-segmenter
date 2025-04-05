# this code with build on top of this colab notebook: https://colab.research.google.com/drive/1q3q4xkFNhYpYfrcSENR6LrS2eJdNKg0X?usp=sharing

from typing import Any
from dataclasses import dataclass
from transformers import AutoFeatureExtractor, Wav2Vec2BertProcessor

import numpy as np


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
    """
    Returns the labels as 1s and 0s as single dimention np array
    """
    num_frames = int(
        np.floor(wav.shape[0] / (hop_length_samples * stride)) + 1)
    labels = np.ones(num_frames, dtype=np.longlong) * silence_label

    interval_idx = 0
    for idx in range(num_frames):
        ...

    return labels


def extract_featrues_and_labels(
    batch: dict[str, list[Any]],
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
            batch['audio'][idx]['sampling_rate'],
            batch['speech_intervals'][idx],
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
    for idx in to_del_ids:
        for key in batch:
            del batch[key][idx]

    # extract features
    processor = AutoFeatureExtractor.from_pretrained(model_id)
    waves = [batch['audio'][idx]['array'] for idx in range(batch['audio'])]
    model_inputs = processor(
        waves,
        sampling_rate=16000,
        return_tensors="np",  # TODO:` numpy or pt
        max_length=max_size_samples,
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
            batch['model_inputs'][idx]['attention_mask'],
            window_length_samples=window_length_samples,
            hop_length_samples=hop_length_samples,
            stride=stride,
            speech_label=speech_label,
            silence_label=silence_label,
            ignored_idx=ignored_idx,
        )
        batch['labels'].append(labels)

    return batch
