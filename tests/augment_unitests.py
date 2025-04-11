import unittest

import numpy as np

from recitations_segmenter.train.augment import (
    truncate,
    TruncateOutput,
    calculate_overlap,
)


def trunc_print(out: TruncateOutput):
    for idx in range(len(out.audio)):
        print(f'Audio:             {out.audio[idx]['array']}')
        print(f'Intervals Samples: {out.speech_intervals_samples[idx]}')
        print(f'Intervals Seconds: {out.speech_intervals_sec[idx]}')
        print('-' * 30)


class TestReciterPool(unittest.TestCase):
    def setUp(self):
        ...

    def tearDown(self):
        ...

    def test_failed_case_turncate(self):
        wav = np.random.rand(593920)
        speech_intervals = np.array(
            [[0.37400001,  5.38600016],
             [8.72599983, 33.51399994]]
        )
        out = truncate(
            wav=wav,
            speech_intervals_sec=speech_intervals,
            sampling_rate=16000,
            truncate_window_overlap_length=16000,
            max_size_samples=80000,
            verbose=True,
        )

        intervals = np.concatenate(out.speech_intervals_samples, 0)
        np.testing.assert_equal((intervals >= 0).all(), True)

    def test_truncate(self):
        wav = np.arange(10, dtype=np.float32)
        speec_intervals = np.arange(
            12, step=2, dtype=np.float32).reshape(-1, 2)

        # overlap=0
        print('overlap=0')
        out = truncate(
            wav=wav,
            speech_intervals_sec=speec_intervals,
            sampling_rate=1,
            truncate_window_overlap_length=0,
            max_size_samples=3,
        )
        print(f'Intervals:\n{speec_intervals}')
        trunc_print(out)
        out_waves = np.concatenate([w['array'] for w in out.audio], axis=0)
        out_intervals = np.concatenate(
            [i for i in out.speech_intervals_samples], 0)
        expected_intervals = np.array(
            [
                [0, 2],
                [1, 2],
                [0, 0],
                [2, 2],
                [0, 0],
            ]
        )
        self.assertEqual((wav == out_waves).all(), True)
        self.assertEqual((out_intervals == expected_intervals).all(), True)
        print('PASSED')

        # overlap=1
        print('\n\noverlap=1')
        out = truncate(
            wav=wav,
            speech_intervals_sec=speec_intervals,
            sampling_rate=1,
            truncate_window_overlap_length=1,
            max_size_samples=3,
        )
        print(f'Intervals:\n{speec_intervals}')
        trunc_print(out)
        out_waves = np.concatenate([w['array'] for w in out.audio], axis=0)
        exeptected_waves = np.array([
            0, 1, 2,
            2, 3, 4,
            4, 5, 6,
            6, 7, 8,
            8, 9
        ])
        out_intervals = np.concatenate(
            [i for i in out.speech_intervals_samples], 0)
        expected_intervals = np.array(
            [
                [0, 2],
                [0, 0],
                [2, 2],
                [0, 2],
                [0, 0],
                [2, 2],
                [0, 1],
            ]
        )
        self.assertEqual((exeptected_waves == out_waves).all(), True)
        self.assertEqual((out_intervals == expected_intervals).all(), True)
        print('PASSED')

        # overlap=2
        print('\n\noverlap=2')
        out = truncate(
            wav=wav,
            speech_intervals_sec=speec_intervals,
            sampling_rate=1,
            truncate_window_overlap_length=2,
            max_size_samples=3,
        )
        print(f'Intervals:\n{speec_intervals}')
        trunc_print(out)
        out_waves = np.concatenate([w['array'] for w in out.audio], axis=0)
        exeptected_waves = np.array([
            0, 1, 2,
            1, 2, 3,
            2, 3, 4,
            3, 4, 5,
            4, 5, 6,
            5, 6, 7,
            6, 7, 8,
            7, 8, 9,
        ])
        out_intervals = np.concatenate(
            [i for i in out.speech_intervals_samples], 0)
        expected_intervals = np.array(
            [
                [0, 2],
                [0, 1],
                [0, 0],
                [2, 2],
                [1, 2],
                [0, 2],
                [0, 1],
                [0, 0],
                [2, 2],
                [1, 2],
            ]
        )
        self.assertEqual((exeptected_waves == out_waves).all(), True)
        self.assertEqual((out_intervals == expected_intervals).all(), True)
        print('PASSED')

    def test_truncate_empty_intervals(self):
        wav = np.arange(5, dtype=np.float32)
        speech_intervals_sec = np.empty((0, 2), dtype=np.float32)
        out = truncate(
            wav=wav,
            speech_intervals_sec=speech_intervals_sec,
            sampling_rate=1,
            truncate_window_overlap_length=1,
            max_size_samples=3,
        )
        # Two windows expected
        self.assertEqual(len(out.audio), 2)
        # Each window's speech intervals should be empty
        for intervals in out.speech_intervals_samples:
            self.assertEqual(len(intervals), 0)
        # Check audio chunks
        expected_audio = [
            np.array([0, 1, 2], dtype=np.float32),
            np.array([2, 3, 4], dtype=np.float32),
        ]
        for i in range(2):
            np.testing.assert_array_equal(
                out.audio[i]['array'], expected_audio[i])

    def test_truncate_entire_audio_speech(self):
        wav = np.arange(10, dtype=np.float32)
        speech_intervals_sec = np.array([[0.0, 9.0]], dtype=np.float32)
        out = truncate(
            wav=wav,
            speech_intervals_sec=speech_intervals_sec,
            sampling_rate=1,
            truncate_window_overlap_length=1,
            max_size_samples=3,
        )
        self.assertEqual(len(out.audio), 5)
        expected_intervals = [
            np.array([[0, 2]]),
            np.array([[0, 2]]),
            np.array([[0, 2]]),
            np.array([[0, 2]]),
            np.array([[0, 1]]),
        ]
        for i in range(5):
            np.testing.assert_array_equal(
                out.speech_intervals_samples[i], expected_intervals[i])

    def test_truncate_small_audio(self):
        wav = np.arange(10, dtype=np.float32)
        speech_intervals_sec = np.array(
            [[0.0, 2.0], [4.0, 9.0]], dtype=np.float32)
        out = truncate(
            wav=wav,
            speech_intervals_sec=speech_intervals_sec,
            sampling_rate=1,
            truncate_window_overlap_length=2,
            max_size_samples=20,
        )
        self.assertEqual(len(out.audio), 1)
        np.testing.assert_array_equal(out.audio[0]['array'], wav)
        intervals = np.concatenate(out.speech_intervals_samples, 0)
        np.testing.assert_array_equal(
            intervals, speech_intervals_sec.astype(np.longlong))

    def test_assert_overlap_smaller_than_window(self):
        with self.assertRaises(AssertionError):
            truncate(
                wav=np.array([0], dtype=np.float32),
                speech_intervals_sec=np.empty((0, 2)),
                sampling_rate=1,
                truncate_window_overlap_length=3,
                max_size_samples=3,
            )

    def test_speech_interval_after_last_window(self):
        wav = np.arange(5, dtype=np.float32)
        speech_intervals_sec = np.array([[4.0, 4.0]], dtype=np.float32)
        out = truncate(
            wav=wav,
            speech_intervals_sec=speech_intervals_sec,
            sampling_rate=1,
            truncate_window_overlap_length=1,
            max_size_samples=3,
        )
        self.assertEqual(len(out.audio), 2)
        self.assertEqual(len(out.speech_intervals_samples[0]), 0)
        np.testing.assert_array_equal(
            out.speech_intervals_samples[1], np.array([[2, 2]]))

    def test_truncate_inteval_end_is_inf(self):
        wav = np.arange(10, dtype=np.float32)
        speech_intervals_sec = np.array(
            [[0.0, 2.0], [4.0, float('inf')]], dtype=np.float32)
        out = truncate(
            wav=wav,
            speech_intervals_sec=speech_intervals_sec,
            sampling_rate=1,
            truncate_window_overlap_length=2,
            max_size_samples=20,
            verbose=True
        )
        self.assertEqual(len(out.audio), 1)
        np.testing.assert_array_equal(out.audio[0]['array'], wav)
        intervals = np.concatenate(out.speech_intervals_samples, 0)
        np.testing.assert_array_equal(
            intervals, np.array([[0, 2], [4, 9]], dtype=np.longlong))

    def test_calculte_overlap(self):
        intervals = np.array([
            [4, 8],
            [10, 19],
            [30, 35],
        ], dtype=np.longlong)
        overlap = calculate_overlap(intervals, window_start=0, window_end=5)
        self.assertEqual(overlap, 1)


if __name__ == '__main__':
    unittest.main()
