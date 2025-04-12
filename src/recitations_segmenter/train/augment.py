# this code with build on top of this colab notebook: https://colab.research.google.com/drive/1q3q4xkFNhYpYfrcSENR6LrS2eJdNKg0X?usp=sharing

from typing import Any
from dataclasses import dataclass
import random

from transformers import AutoFeatureExtractor, Wav2Vec2BertProcessor

from audiomentations import TimeStretch
import numpy as np
from numpy.typing import NDArray
from datasets import Features, Value, Audio, Array2D, IterableDataset
from pydantic import BaseModel, Field
import yaml


class AugmentConfig(BaseModel):
    min_size_samples: int = Field(
        32000,
        description="Minimum audio length in samples (2 seconds at 16kHz sampling rate)"
    )
    max_size_samples: int = Field(
        320000,
        description="Maximum audio length in samples (20 seconds at 16kHz sampling rate)"
    )
    truncate_window_overlap_length: int = Field(
        16000,
        description="Overlap length in samples when splitting long audio segments as sliding windowd alogrithm"
    )
    window_length_samples: int = Field(
        400,
        description="Window length for spectrogram feature extraction (in samples) taken from Wav2VecBertProcessor"
    )
    hop_length_samples: int = Field(
        160,
        description="Hop length between spectrogram frames (in samples) taken from Wav2VecBertProcessor"
    )
    sampling_rate: int = Field(
        16000,
        description="Audio sampling rate in Hz"
    )
    stride: int = Field(
        2,
        description="Stride value for feature extraction convolution operations taken from Wav2VecBertProcessor"
    )
    speech_label: int = Field(
        1,
        description="Label value for speech segments"
    )
    silence_label: int = Field(
        0,
        description="Label value for silence/non-speech segments"
    )
    ignored_idx: int = Field(
        -100,
        description="Index value to ignore in loss calculations (for padding/masking)"
    )
    model_id: str = Field(
        'facebook/w2v-bert-2.0',
        description="HuggingFace model identifier for Wav2Vec2-BERT"
    )
    batch_size: int = Field(
        32,
        alias='batch-size',
        description="Batch size for processing augmented samples"
    )
    samples_per_shard: int = Field(
        1024,
        alias='samples-per-shard',
        description="Number of samples per shard in Parquet output to save"
    )
    seed: int = Field(
        1,
        description="Random seed for reproducibility"
    )
    min_stretch_ratio: float = Field(
        0.8,
        alias='min-stretch-ratio',
        description="Minimum time stretch ratio (0.8x speed)"
    )
    max_stretch_ratio: float = Field(
        1.5,
        alias='max-stretch-ratio',
        description="Maximum time stretch ratio (1.5x speed)"
    )
    augment_prob: float = Field(
        0.4,
        alias='augment-prob',
        description="Probability of applying augmentation to samples"
    )

    @classmethod
    def from_yaml(cls, yaml_file: str) -> 'AugmentConfig':
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)


DS_FEATURES_AUGMNETED = Features({
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
})


DS_FEATURES_TRAIN = Features({
    'aya_name': Value(dtype='string'),
    'aya_id': Value(dtype='string'),
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


def build_audiomentations_augs(p=0.4, seed=42, all=False):
    """taken form: https://github.com/snakers4/silero-vad/blob/master/tuning/utils.py#L37
    """
    # audiomentations usesd python random for its calculations
    random.seed(seed)
    np.random.seed(seed)

    from audiomentations import (
        SomeOf,
        AirAbsorption,
        BandPassFilter,
        BandStopFilter,
        ClippingDistortion,
        HighPassFilter,
        HighShelfFilter,
        LowPassFilter,
        LowShelfFilter,
        Mp3Compression,
        PeakingFilter,
        PitchShift,
        RoomSimulator,
        SevenBandParametricEQ,
        Aliasing,
        AddGaussianNoise,
        GainTransition,
        Compose,
    )
    transforms = [
        Aliasing(p=1),
        AddGaussianNoise(p=1),
        AirAbsorption(p=1),
        BandPassFilter(p=1),
        BandStopFilter(p=1),
        ClippingDistortion(p=1),
        HighPassFilter(p=1),
        HighShelfFilter(p=1),
        LowPassFilter(p=1),
        LowShelfFilter(p=1),
        Mp3Compression(p=1),
        PeakingFilter(p=1),
        PitchShift(p=1),
        RoomSimulator(p=1, leave_length_unchanged=True),
        SevenBandParametricEQ(p=1),
        GainTransition(p=1, min_gain_db=-17),
    ]
    if all:
        return Compose(transforms, p=p)
    return SomeOf((1, 3), transforms=transforms, p=p)


class StrechAugment(object):
    def __init__(
        self,
        seed=77,
        stretch_ragne=[0.8, 1.25],
        augment_prob=0.4,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.stretch_range = stretch_ragne
        self.augment_prob = augment_prob
        self.augment = build_audiomentations_augs(
            1, seed=seed)

    def _apply_stretching(
        self,
        wav: NDArray[np.float32],
        sampling_rate=16000,
    ) -> tuple[NDArray[np.float32], float]:

        if not isinstance(wav, np.ndarray):
            wav = np.array(wav, dtype=np.float32)

        # No stretching
        if self.rng.random() > self.augment_prob:
            return np.array(wav), 1

        speed = self.rng.uniform(
            self.stretch_range[0], self.stretch_range[1])
        augment = TimeStretch(
            min_rate=speed,
            max_rate=speed,
            p=1,
            leave_length_unchanged=False,
        )
        return augment(wav, sampling_rate), speed

    def _apply_augmentations(
        self,
        wav: NDArray[np.float32],
        sampling_rate=16000,
    ) -> tuple[NDArray[np.float32], bool]:
        """
        Returns:
            (new_wav, is_augmented)
        """

        if not isinstance(wav, np.ndarray):
            wav = np.array(wav, dtype=np.float32)

        # No stretching
        if self.rng.random() > self.augment_prob:
            return np.array(wav), False

        new_wav = self.augment(wav, sampling_rate)
        return new_wav, True

    def __call__(
        self,
        batch
    ) -> dict[str, list]:

        batch['speed'] = []
        batch['is_augmented'] = []
        for idx in range(len(batch['audio'])):

            # Apply stetching
            new_wav, speed = self._apply_stretching(
                batch['audio'][idx]['array'],
                batch['audio'][idx]['sampling_rate'])

            batch['audio'][idx]['array'] = new_wav
            batch['duration'][idx] = len(
                new_wav) / batch['audio'][idx]['sampling_rate']
            batch['speech_intervals'][idx] = (
                np.array(batch['speech_intervals'][idx]) / speed)
            batch['speed'].append(speed)

            # Applying augmentations
            # NOTE: we are applying augmentations for both stretched and
            # not stretched samples
            augmented_wav, is_augmented = self._apply_augmentations(
                batch['audio'][idx]['array'],
                batch['audio'][idx]['sampling_rate'],
            )
            batch['audio'][idx]['array'] = augmented_wav
            batch['is_augmented'].append(is_augmented)

        return batch


def augment_ds_split(
    ds: IterableDataset,
    seed=77,
    stretch_ragne=[0.8, 1.25],
    augment_prob=0.4,
    batch_size=32,
) -> IterableDataset:

    assert isinstance(ds, IterableDataset), (
        f'We only support `IterableDataset` we got: {type(ds)}')
    mapping_func = StrechAugment(
        seed=seed,
        stretch_ragne=stretch_ragne,
        augment_prob=augment_prob,
    )
    out_ds = ds.map(
        mapping_func,
        features=DS_FEATURES_AUGMNETED,
        batched=True,
        batch_size=batch_size,
    )

    return out_ds


# ---------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------

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
    verbose=False,
) -> TruncateOutput:
    """Moving winodw truncatation arlogrith where the window size is `max_size_samples`
    Note:
    * speech_inatevals are execlusive EX intv = [1, 6] so [1, 2, 3, 4, ,5] are speech
    * speech_intervals are not overlapped
    """

    assert max_size_samples > truncate_window_overlap_length, '`max_size_samples` should be > `truncate_window_overlap_length` '
    speech_intervals_samples = np.array(speech_intervals_sec) * sampling_rate
    speech_intervals_samples = speech_intervals_samples.astype(np.longlong)

    # edge case last interval end should be < total waves length &interval end with inf
    if speech_intervals_samples.shape[0] > 0:
        if speech_intervals_samples[-1][1] > len(wav) or speech_intervals_sec[-1][-1] == float('inf'):
            speech_intervals_samples[-1][1] = len(wav)

    out = TruncateOutput([], [], [])
    overlap = truncate_window_overlap_length
    window = max_size_samples
    step = window - overlap
    num_items = int(
        np.ceil(max(0, len(wav) - window) / (window - overlap))) + 1
    if len(wav) == 0:
        num_items = 0

    if verbose:
        print(f'num of items: {num_items}')

    # if verbose:
    #     print(f'before intervals:\n{speech_intervals_samples}')
    #     print(f'before seconds:\n{speech_intervals_sec}')
    #     print(f'len of wav: {len(wav)}')
    #     print(f'num of items: {num_items}')

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
            # print(f' speech_intervals:\n {speech_intervals_samples}')
            # start >= interval end (because of speech iterval end are execlusive)
            if start >= speech_intervals_samples[intv_idx][1]:
                break

            # interval end is smaller than the winodw size
            # ( <=because of speech iterval end are execlusive)
            elif speech_intervals_samples[intv_idx][1] <= end - overlap:
                chosen_idx += 1

            # deviding the speech interval in two parts
            # part to be added to the currect frame(idx)
            # and the other one for the next frame
            elif speech_intervals_samples[intv_idx][0] < end:
                frgmented_intv = np.zeros(2, dtype=np.longlong)
                # in case of overlapping winodws
                frgmented_intv[0] = speech_intervals_samples[intv_idx][0]
                frgmented_intv[1] = min(
                    end, int(speech_intervals_samples[intv_idx][1]))

                # new start for the next iteration
                # if start of speech interval between end and (end -overlap)
                speech_intervals_samples[intv_idx][0] = max(
                    end - overlap, int(speech_intervals_samples[intv_idx][0]))
                break

            # TODO: non reachable case
            else:
                break

        if frgmented_intv is None:
            out.speech_intervals_samples.append(
                speech_intervals_samples[intv_start_idx: chosen_idx].copy())
        else:
            out.speech_intervals_samples.append(
                np.concatenate(
                    (speech_intervals_samples[intv_start_idx: chosen_idx].copy(), np.expand_dims(frgmented_intv, 0)), axis=0),
            )

        # print('before')
        # print(f'{idx}:\n{np.concatenate(out.speech_intervals_samples, 0)}')
        # print(f'intv idx: {intv_idx}')

        # making intervals relative to each audio frame not the entire audio
        out.speech_intervals_samples[-1] -= start

        # print('after')
        # print(np.concatenate(out.speech_intervals_samples, 0))
        # print('-' * 50)

        # end of the loop
        out.speech_intervals_sec.append(
            out.speech_intervals_samples[-1] / sampling_rate)
        start += step
        intv_start_idx = intv_idx

    # if (num_items == 10) and verbose:
    #     print(out.speech_intervals_sec)
    #     print(out.speech_intervals_samples)
    #     print('\n\n\n')

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
            Note: the interval_end are exlusive exacly like python indexing

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
    end[:, 1] = intervals[:, 1]
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
            NOTE: both start and end are execlusive exaclly python indexing
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
    new_batch = {'audio': [], 'speech_intervals': [], 'aya_id': []}
    for key in batch.keys():
        new_batch[key] = []

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
        for trunc_idx in range(len(trunc_outs.audio)):
            new_batch['aya_id'].append(f'{batch['aya_name'][idx]}_{trunc_idx}')
            new_batch['duration'].append(
                len(trunc_outs.audio[trunc_idx]['array']) / trunc_outs.audio[trunc_idx]['sampling_rate'])

        for key in set(batch.keys()) - {'audio', 'speech_intervals', 'duration'}:
            new_batch[key] += [batch[key][idx]] * len(trunc_outs.audio)

    # back to new batch
    for key in new_batch.keys():
        batch[key] = new_batch[key]

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
        sampling_rate=sampling_rate,
        return_tensors="np",
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
    ds: IterableDataset,
    config: AugmentConfig,
) -> IterableDataset:
    assert isinstance(ds, IterableDataset), (
        f'We only support `IterableDataset` we got: {type(ds)}')

    out_ds = ds.map(
        extract_features_and_labels,
        batched=True,
        batch_size=config.batch_size,
        features=DS_FEATURES_TRAIN,
        fn_kwargs={
            'min_size_samples': config.min_size_samples,
            'max_size_samples': config.max_size_samples,
            'truncate_window_overlap_length': config.truncate_window_overlap_length,
            'window_length_samples': config.window_length_samples,
            'hop_length_samples': config.hop_length_samples,
            'sampling_rate': config.sampling_rate,
            'stride': config.stride,
            'speech_label': config.speech_label,
            'silence_label': config.silence_label,
            'ignored_idx': config.ignored_idx,
            'model_id': config.model_id,
        },
    )
    return out_ds
