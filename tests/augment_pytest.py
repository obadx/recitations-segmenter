from typing import Dict

import numpy as np
import pytest
from datasets import Dataset

from recitations_segmenter.train.augment import (
    calculate_overlap,
    annotate,
    extract_features_and_labels,
)


@pytest.mark.parametrize(
    "intervals, window_start, window_end, expected",
    [
        # No overlap: intervals before window
        ([[6, 10]], 0, 5, 0),
        # No overlap: intervals after window
        ([[1, 5]], 6, 10, 0),
        # Single interval fully inside window
        ([[3, 7]], 2, 8, 5),
        # Single interval fully covering window
        ([[2, 10]], 5, 8, 3),
        # Partial overlap at the start of the window
        ([[1, 4]], 3, 6, 2),
        # Partial overlap at the end of the window
        ([[5, 9]], 3, 7, 2),
        # Interval adjacent to window start (no overlap)
        ([[3, 4]], 5, 10, 0),
        # Interval adjacent to window end (no overlap)
        ([[10, 15]], 5, 10, 0),
        # Multiple intervals with partial and full overlaps
        ([[1, 3], [4, 6], [7, 9]], 2, 8, 6),
        # Interval ends exactly at window start (no overlap)
        ([[3, 4]], 5, 8, 0),
        # Interval starts exactly at window end (no overlap)
        ([[8, 10]], 5, 8, 0),
        # Zero-length interval inside window
        ([[5, 5]], 5, 6, 1),
        # Zero-length interval outside window
        ([[4, 4]], 5, 6, 0),
        # Multiple intervals with some overlaps
        ([[2, 3], [5, 7], [9, 10]], 4, 8, 3),
        # Invalid interval (start > end) results in no overlap
        ([[7, 5]], 4, 8, 0),
        # Zero-length window (no overlap)
        ([[5, 5]], 5, 5, 0),
    ],
)
def test_calculate_overlap(intervals, window_start, window_end, expected):
    intervals_np = np.array(intervals)
    result = calculate_overlap(intervals_np, window_start, window_end)
    assert result == expected


def calc_frames(L, W=400, H=160, S=2):
    return int(1 + np.floor((L - W) / H)) // S


@pytest.mark.parametrize(
    "wav, speech_intervals, attention_mask, window_length, hop_length, stride, speech_label, silence_label, expected_labels",
    [
        # Test Case 1: Full overlap → speech
        (
            np.random.rand(16000),
            [[0, 399]],
            np.ones(calc_frames(16000)),
            400,
            160,
            2,
            1,
            0,
            [1] + [0] * (calc_frames(16000) - 1),
        ),
        # Test Case 1.1: Full overlap → speech partilly maskes
        (
            np.random.rand(16000),
            [[0, 399]],
            [1] * 10 + [0] * (calc_frames(16000) - 10),
            400,
            160,
            2,
            1,
            0,
            [1] + [0] * 9 + [-100] * (calc_frames(16000) - 10),
        ),

        # Test Case 2: No overlap → silence (exactly 50% overlap)
        (
            np.random.rand(16000),
            [[0, 279]],
            np.ones(calc_frames(16000)),
            400,
            160,
            2,
            1,
            0,
            [0] + [0] * (calc_frames(16000) - 1),
        ),
        # Test Case 2.2: No overlap → silence (exactly 50% overlap) + all masked
        (
            np.random.rand(16000),
            [[0, 279]],
            np.zeros(calc_frames(16000), dtype=np.longlong),
            400,
            160,
            2,
            1,
            0,
            [-100] * (calc_frames(16000)),
        ),
        # # Test Case 3: Exactly 50% overlap → silence (the other end)
        (
            np.random.rand(16000),
            [[200, 479]],
            np.ones(calc_frames(16000), dtype=np.longlong),
            400,
            160,
            2,
            1,
            0,
            [0] * (calc_frames(16000)),
        ),
        # Test Case 4: Multiple frames with varying overlaps
        (
            np.random.rand(16000),
            [[0, 399], [600, 820]],
            np.ones(calc_frames(16000), dtype=np.longlong),
            400,
            160,
            2,
            1,
            0,
            [1, 1] + [0] * (calc_frames(16000) - 2),
        ),
        # # Test Case 5: Zero-length speech intervals → all silence
        (
            np.random.rand(16000),
            [],
            np.ones(calc_frames(16000), dtype=np.longlong),
            400,
            160,
            2,
            1,
            0,
            [0] * (calc_frames(16000)),
        ),
        # Test Case 8: Custom labels
        (
            np.random.rand(16000),
            [[80, 400], [920, 1500]],
            np.ones(calc_frames(16000), dtype=np.longlong),
            400,
            160,
            2,
            2,
            3,
            [2, 3, 3, 2] + [3] * (calc_frames(16000) - 4),
        ),
        # Test Case 10: Stride 1 and hop length 160 → check window positions
        (
            np.random.rand(16000),
            [[0, 500]],
            np.ones(calc_frames(16000), dtype=np.longlong),
            400,
            160,
            1,
            1,
            0,
            [1, 1] + [0] * (calc_frames(16000) - 2),
        ),
        # Test Case 11: Overlap from multiple intervals → summed correctly, stride = 1
        (
            np.random.rand(16000),
            [[0, 199], [300, 499]],
            np.ones(calc_frames(16000), dtype=np.longlong),
            400,
            160,
            1,
            1,
            0,
            [1, 1] + [0] * (calc_frames(16000) - 2),
        ),
    ],
)
def test_annotate(
    wav,
    speech_intervals,
    attention_mask,
    window_length,
    hop_length,
    stride,
    speech_label,
    silence_label,
    expected_labels,
):
    # Convert inputs to numpy arrays
    speech_intervals_np = (
        np.array(speech_intervals, dtype=np.int64)
        if speech_intervals
        else np.empty((0, 2), dtype=np.int64)
    )
    attention_mask_np = np.array(attention_mask, dtype=np.longlong)

    # Generate labels using the annotate function
    result = annotate(
        wav=wav,
        speech_intervals_samples=speech_intervals_np,
        attention_mask=attention_mask_np,
        window_length_samples=window_length,
        hop_length_samples=hop_length,
        stride=stride,
        speech_label=speech_label,
        silence_label=silence_label,
    )

    # Convert expected labels to numpy array
    expected_np = np.array(expected_labels, dtype=np.longlong)

    # Verify the output matches expected labels
    np.testing.assert_array_equal(result, expected_np)


@pytest.fixture
def sample_batch() -> Dict[str, list]:
    return {
        'audio': [{
            'array': np.random.rand(480000).astype(np.float32),
            'sampling_rate': 16000,
        }],
        'speech_intervals': [np.array([[0.0, 30.0]], dtype=np.float32)],
        'aya_name': ['sample_aya'],
        'reciter_name': ['sample_reciter'],
        'recitation_id': [1],
        'url': ['http://example.com'],
        'duration': [30.0],
        'speed': [1.0],
        'is_interval_complete': [True],
        'is_augmented': [False],
    }


def test_extract_features_and_labels_basic(sample_batch):
    # Test basic processing with one sample
    processed_batch = extract_features_and_labels(
        sample_batch,
        min_size_samples=32000,
        max_size_samples=480000,
        truncate_window_overlap_length=16000,
    )

    assert 'input_features' in processed_batch
    assert 'attention_mask' in processed_batch
    assert 'labels' in processed_batch
    assert 'aya_id' in processed_batch
    assert len(processed_batch['audio']) == 1
    assert len(processed_batch['labels'][0]) == calc_frames(
        len(sample_batch['audio'][0]['array']))
    assert len(processed_batch['labels'][0][processed_batch['labels'][0] != -100]) == calc_frames(
        len(sample_batch['audio'][0]['array']))


# def test_truncate_removes_short_samples(sample_batch):
#     # Add a short sample to the batch
#     sample_batch['audio'].append({
#         'array': np.random.rand(16000).astype(np.float32),
#         'sampling_rate': 16000,
#     })
#     sample_batch['speech_intervals'].append(
#         np.array([[0.0, 1.0]], dtype=np.float32))
#     # Other fields are extended for consistency
#
#     processed_batch = extract_features_and_labels(
#         sample_batch,
#         min_size_samples=32000,
#     )
#
#     # The second sample (16000 samples) should be removed
#     assert len(processed_batch['audio']) == 1


# def test_extract_features_for_ds(sample_batch, mock_processor):
#     # Create a Dataset from the sample batch
#     ds = Dataset.from_dict(sample_batch)
#     processed_ds = extract_features_for_ds(
#         ds,
#         batch_size=1,
#         num_proc=1,
#         model_id='mock/model',
#     )
#
#     assert 'input_features' in processed_ds.features
#     assert 'attention_mask' in processed_ds.features
#     assert 'labels' in processed_ds.features
#     assert len(processed_ds) == 1
#
#
# def test_dataset_dict_processing(sample_batch, mock_processor):
#     # Create a DatasetDict with train and test splits
#     train_ds = Dataset.from_dict(sample_batch)
#     test_ds = Dataset.from_dict(sample_batch)
#     ds_dict = DatasetDict({'train': train_ds, 'test': test_ds})
#
#     processed_ds = extract_features_for_ds(
#         ds_dict,
#         batch_size=1,
#         num_proc=1,
#     )
#
#     assert 'train' in processed_ds
#     assert 'test' in processed_ds
#     assert all(
#         feat in processed_ds['train'].features for feat in DS_FEATURES_TRAIN)
