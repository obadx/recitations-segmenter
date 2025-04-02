import random

from audiomentations import TimeStretch
import numpy as np
from numpy.typing import NDArray
from datasets import Features, Value, Audio, Array2D, IterableDataset


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

# TODO:
# * stresh [DONE]
# * Augment
# Training:
# * Truncate
# * Extract Features


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


class ExtractFeatures(object):
    ...
