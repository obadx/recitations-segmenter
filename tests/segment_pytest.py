import torch
import pytest
import subprocess
import shutil
from pathlib import Path
import json


from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification
import numpy as np


from recitations_segmenter.segment import (
    batchify_input,
    WavInfo,
    collect_results,
    generate_time_stamps,
    extract_speech_intervals,
    NoSpeechIntervals,
    TooHighMinSpeechDuration,
    read_audio,
    segment_recitations,
    clean_speech_intervals,
)


class TestBatchify:
    def test_single_waveform_no_padding(self):
        waves = [torch.ones(5)]
        max_len = 5
        batch_size = 1
        wav_infos, batches = batchify_input(waves, max_len, batch_size)

        assert len(wav_infos) == 1
        info = wav_infos[0]
        assert info == WavInfo(
            wav_len=5,
            batch_start=0,
            batch_end=1,
            idx_in_batch_start=0,
            idx_in_batch_end=1
        )
        assert len(batches) == 1
        assert batches[0].shape == (1, 5)
        assert torch.all(batches[0] == 1)

    def test_single_waveform_with_padding(self):
        waves = [torch.ones(3)]
        max_len = 5
        batch_size = 1
        wav_infos, batches = batchify_input(waves, max_len, batch_size)

        info = wav_infos[0]
        padded_wav = torch.cat((torch.ones(3), torch.zeros(2)))
        assert info == WavInfo(
            wav_len=3,
            batch_start=0,
            batch_end=1,
            idx_in_batch_start=0,
            idx_in_batch_end=1
        )
        assert batches[0].shape == (1, 5)
        assert torch.all(batches[0][0] == padded_wav)

    def test_exact_batch_fill(self):
        waves = [torch.ones(10)]
        max_len = 5
        batch_size = 2
        wav_infos, batches = batchify_input(waves, max_len, batch_size)

        info = wav_infos[0]
        assert info == WavInfo(
            wav_len=10,
            batch_start=0,
            batch_end=1,
            idx_in_batch_start=0,
            idx_in_batch_end=2  # Expected failure: code returns 0
        )
        assert len(batches) == 1
        assert batches[0].shape == (2, 5)

    def test_chunks_span_multiple_batches(self):
        waves = [torch.ones(15)]
        max_len = 5
        batch_size = 2
        wav_infos, batches = batchify_input(waves, max_len, batch_size)

        info = wav_infos[0]
        assert info == WavInfo(
            wav_len=15,
            batch_start=0,
            batch_end=2,
            idx_in_batch_start=0,
            idx_in_batch_end=1
        )
        assert len(batches) == 2
        assert batches[0].shape == (2, 5)
        assert batches[1].shape == (1, 5)

    def test_multiple_waveforms(self):
        wave1 = torch.ones(5)
        wave2 = torch.ones(8)
        waves = [wave1, wave2]
        max_len = 5
        batch_size = 2
        wav_infos, batches = batchify_input(waves, max_len, batch_size)

        # Validate first waveform
        info1 = wav_infos[0]
        assert info1 == WavInfo(
            wav_len=5,
            batch_start=0,
            batch_end=1,
            idx_in_batch_start=0,
            idx_in_batch_end=1
        )

        # Validate second waveform
        info2 = wav_infos[1]
        assert info2 == WavInfo(
            wav_len=8,
            batch_start=0,
            batch_end=2,
            idx_in_batch_start=1,
            idx_in_batch_end=1
        )

        assert len(batches) == 2
        assert batches[0].shape == (2, 5)
        assert batches[1].shape == (1, 5)

    def test_multiple_waveforms_in_single_batch(self):
        wave1 = torch.ones(5)
        wave2 = torch.ones(8)
        waves = [wave1, wave2]
        max_len = 5
        batch_size = 5
        wav_infos, batches = batchify_input(waves, max_len, batch_size)

        # Validate first waveform
        info1 = wav_infos[0]
        assert info1 == WavInfo(
            wav_len=5,
            batch_start=0,
            batch_end=1,
            idx_in_batch_start=0,
            idx_in_batch_end=1
        )

        # Validate second waveform
        info2 = wav_infos[1]
        assert info2 == WavInfo(
            wav_len=8,
            batch_start=0,
            batch_end=1,
            idx_in_batch_start=1,
            idx_in_batch_end=3
        )

        assert len(batches) == 1
        assert batches[0].shape == (3, 5)

    def test_multiple_waveforms_over_multiple_batches(self):
        wave1 = torch.ones(32)
        wave2 = torch.ones(18)
        wave3 = torch.ones(11)
        waves = [wave1, wave2, wave3]
        max_len = 8
        batch_size = 4
        wav_infos, batches = batchify_input(waves, max_len, batch_size)

        # Validate first waveform
        info1 = wav_infos[0]
        assert info1 == WavInfo(
            wav_len=32,
            batch_start=0,
            batch_end=1,
            idx_in_batch_start=0,
            idx_in_batch_end=4
        )

        # Validate second waveform
        info2 = wav_infos[1]
        assert info2 == WavInfo(
            wav_len=18,
            batch_start=1,
            batch_end=2,
            idx_in_batch_start=0,
            idx_in_batch_end=3
        )

        info3 = wav_infos[2]
        assert info3 == WavInfo(
            wav_len=11,
            batch_start=1,
            batch_end=3,
            idx_in_batch_start=3,
            idx_in_batch_end=1
        )

        assert len(batches) == 3
        assert batches[0].shape == (4, 8)
        assert batches[1].shape == (4, 8)
        assert batches[2].shape == (1, 8)

    def test_empty_input(self):
        waves = []
        max_len = 5
        batch_size = 64
        wav_infos, batches = batchify_input(waves, max_len, batch_size)

        assert len(wav_infos) == 0
        assert len(batches) == 0

    def test_zero_length_waveform(self):
        waves = [torch.zeros(0)]
        max_len = 5
        batch_size = 1
        with pytest.raises(AssertionError):
            wav_infos, batches = batchify_input(waves, max_len, batch_size)

    def test_multiple_waveforms_real_case_batch_end(self):
        wave1 = torch.ones(2160849)
        wave2 = torch.ones(84428)
        waves = [wave1, wave2]
        max_len = 19995 * 16  # 319920
        batch_size = 8
        wav_infos, batches = batchify_input(waves, max_len, batch_size)

        # Validate first waveform
        info1 = wav_infos[0]
        assert info1 == WavInfo(
            wav_len=len(wave1),
            batch_start=0,
            batch_end=1,
            idx_in_batch_start=0,
            idx_in_batch_end=7,
        )

        # Validate second waveform
        info2 = wav_infos[1]
        assert info2 == WavInfo(
            wav_len=len(wave2),
            batch_start=0,
            batch_end=1,
            idx_in_batch_start=7,
            idx_in_batch_end=8
        )

        assert len(batches) == 1
        assert batches[0].shape == (batch_size, max_len)

    def test_multiple_waveforms_2_batch_end(self):
        wave1 = torch.ones(53)
        wave2 = torch.ones(11)
        waves = [wave1, wave2]
        max_len = 10
        batch_size = 4
        wav_infos, batches = batchify_input(waves, max_len, batch_size)

        # Validate first waveform
        info1 = wav_infos[0]
        assert info1 == WavInfo(
            wav_len=len(wave1),
            batch_start=0,
            batch_end=2,
            idx_in_batch_start=0,
            idx_in_batch_end=2,
        )

        # Validate second waveform
        info2 = wav_infos[1]
        assert info2 == WavInfo(
            wav_len=len(wave2),
            batch_start=1,
            batch_end=2,
            idx_in_batch_start=2,
            idx_in_batch_end=4
        )

        assert len(batches) == 2
        assert batches[0].shape == (batch_size, max_len)


class TestCollect:
    processor = AutoFeatureExtractor.from_pretrained('facebook/w2v-bert-2.0')

    def _get_prcossor_len(self, L):
        # out_l = self.processor(
        #     np.zeros(L),
        #     return_tensors='pt',
        #     sampling_rate=16000,
        # )['attention_mask'][0].sum()
        # return out_l
        return L

    def _get_golden_logits(self, lens: list[int]):
        out_logits = []
        for L in lens:
            out_logits.append(torch.arange(
                L).repeat_interleave(2).reshape(-1, 2))
        return out_logits

    def _get_batched_logits(
        self,
        waves: list[torch.FloatTensor],
        max_len,
        batch_size,
    ):
        # logits are shape of batch, len, 2
        max_len = self._get_prcossor_len(max_len) * 2
        wav_logits = []
        lens = []
        for wav in waves:
            logits_len = self._get_prcossor_len(len(wav))
            lens.append(logits_len)
            wav_logits.append(torch.arange(logits_len).repeat_interleave(2))

        infos, batched_logits = batchify_input(
            wav_logits, max_len, max_batch_size=batch_size)
        batched_logits = list(batched_logits)

        for idx in range(len(batched_logits)):
            batch_size, _ = batched_logits[idx].shape
            batched_logits[idx] = batched_logits[idx].reshape(
                batch_size, -1, 2)

        return infos, lens, batched_logits

    def _check_collect(
        self,
        waves,
        wav_infos: list[WavInfo],
        max_len,
        batch_size
    ):
        infos, lens, batched_logits = self._get_batched_logits(
            waves, max_len, batch_size)
        golden_logits = self._get_golden_logits(lens)

        out_logits = collect_results(wav_infos, batched_logits)

        print(golden_logits)
        print(out_logits)

        assert len(golden_logits) == len(out_logits)
        for idx in range(len(golden_logits)):
            g_len = len(golden_logits[idx])
            assert len(golden_logits[idx]) <= len(out_logits[idx])
            assert (golden_logits[idx] == out_logits[idx][:g_len]).all()

    def test_single_waveform_no_padding(self):
        waves = [torch.ones(5)]
        max_len = 5
        batch_size = 1
        wav_infos, batches = batchify_input(waves, max_len, batch_size)

        assert len(wav_infos) == 1
        info = wav_infos[0]
        assert info == WavInfo(
            wav_len=5,
            batch_start=0,
            batch_end=1,
            idx_in_batch_start=0,
            idx_in_batch_end=1
        )
        assert len(batches) == 1
        assert batches[0].shape == (1, max_len)
        assert torch.all(batches[0] == 1)

        self._check_collect(waves, wav_infos, max_len, batch_size)

    def test_single_waveform_with_padding(self):
        waves = [torch.ones(3)]
        max_len = 5
        batch_size = 1
        wav_infos, batches = batchify_input(waves, max_len, batch_size)

        info = wav_infos[0]
        padded_wav = torch.cat((torch.ones(3), torch.zeros(2)))
        assert info == WavInfo(
            wav_len=3,
            batch_start=0,
            batch_end=1,
            idx_in_batch_start=0,
            idx_in_batch_end=1
        )
        assert batches[0].shape == (1, max_len)
        assert torch.all(batches[0][0] == padded_wav)

        self._check_collect(waves, wav_infos, max_len, batch_size)

    def test_exact_batch_fill(self):
        waves = [torch.ones(16)]
        max_len = 8
        batch_size = 2
        wav_infos, batches = batchify_input(waves, max_len, batch_size)

        info = wav_infos[0]
        assert info == WavInfo(
            wav_len=16,
            batch_start=0,
            batch_end=1,
            idx_in_batch_start=0,
            idx_in_batch_end=2  # Expected failure: code returns 0
        )
        assert len(batches) == 1
        assert batches[0].shape == (2, max_len)

        self._check_collect(waves, wav_infos, max_len, batch_size)

    def test_chunks_span_multiple_batches(self):
        waves = [torch.ones(15)]
        max_len = 5
        batch_size = 2
        wav_infos, batches = batchify_input(waves, max_len, batch_size)

        info = wav_infos[0]
        assert info == WavInfo(
            wav_len=15,
            batch_start=0,
            batch_end=2,
            idx_in_batch_start=0,
            idx_in_batch_end=1
        )
        assert len(batches) == 2
        assert batches[0].shape == (2, max_len)
        assert batches[1].shape == (1, max_len)

        self._check_collect(waves, wav_infos, max_len, batch_size)

    def test_multiple_waveforms(self):
        wave1 = torch.ones(5)
        wave2 = torch.ones(8)
        waves = [wave1, wave2]
        max_len = 5
        batch_size = 2
        wav_infos, batches = batchify_input(waves, max_len, batch_size)

        # Validate first waveform
        info1 = wav_infos[0]
        assert info1 == WavInfo(
            wav_len=5,
            batch_start=0,
            batch_end=1,
            idx_in_batch_start=0,
            idx_in_batch_end=1
        )

        # Validate second waveform
        info2 = wav_infos[1]
        assert info2 == WavInfo(
            wav_len=8,
            batch_start=0,
            batch_end=2,
            idx_in_batch_start=1,
            idx_in_batch_end=1
        )

        assert len(batches) == 2
        assert batches[0].shape == (2, max_len)
        assert batches[1].shape == (1, max_len)

        self._check_collect(waves, wav_infos, max_len, batch_size)

    def test_multiple_waveforms_in_single_batch(self):
        wave1 = torch.ones(5)
        wave2 = torch.ones(8)
        waves = [wave1, wave2]
        max_len = 5
        batch_size = 5
        wav_infos, batches = batchify_input(waves, max_len, batch_size)

        # Validate first waveform
        info1 = wav_infos[0]
        assert info1 == WavInfo(
            wav_len=5,
            batch_start=0,
            batch_end=1,
            idx_in_batch_start=0,
            idx_in_batch_end=1
        )

        # Validate second waveform
        info2 = wav_infos[1]
        assert info2 == WavInfo(
            wav_len=8,
            batch_start=0,
            batch_end=1,
            idx_in_batch_start=1,
            idx_in_batch_end=3
        )

        assert len(batches) == 1
        assert batches[0].shape == (3, max_len)

        self._check_collect(waves, wav_infos, max_len, batch_size)

    def test_multiple_waveforms_over_multiple_batches(self):
        wave1 = torch.ones(32)
        wave2 = torch.ones(18)
        wave3 = torch.ones(11)
        waves = [wave1, wave2, wave3]
        max_len = 8
        batch_size = 4
        wav_infos, batches = batchify_input(waves, max_len, batch_size)

        # Validate first waveform
        info1 = wav_infos[0]
        assert info1 == WavInfo(
            wav_len=32,
            batch_start=0,
            batch_end=1,
            idx_in_batch_start=0,
            idx_in_batch_end=4
        )

        # Validate second waveform
        info2 = wav_infos[1]
        assert info2 == WavInfo(
            wav_len=18,
            batch_start=1,
            batch_end=2,
            idx_in_batch_start=0,
            idx_in_batch_end=3
        )

        info3 = wav_infos[2]
        assert info3 == WavInfo(
            wav_len=11,
            batch_start=1,
            batch_end=3,
            idx_in_batch_start=3,
            idx_in_batch_end=1
        )

        assert len(batches) == 3
        assert batches[0].shape == (4, max_len)
        assert batches[1].shape == (4, max_len)
        assert batches[2].shape == (1, max_len)

        self._check_collect(waves, wav_infos, max_len, batch_size)


class TestGenerateTimeStamps:
    processor = AutoFeatureExtractor.from_pretrained('facebook/w2v-bert-2.0')

    def _get_len(self, L):
        out_l = self.processor(
            np.zeros(L),
            return_tensors='pt',
            sampling_rate=16000,
        )['attention_mask'].shape[1]
        return out_l

    def test_single_batch(self):
        max_len = 1000
        time_stamps = generate_time_stamps(
            self._get_len(max_len),
            max_duration_samples=320000,
            max_featrues_len=self._get_len(max_len),
            window=400,
            hop=160,
            stride=2,
        )
        golden_stamps = torch.tensor([0, 320])
        assert (golden_stamps == time_stamps).all()

    def test_single_batch_residual(self):
        max_len = 1000
        time_stamps = generate_time_stamps(
            self._get_len(max_len) * 1 + 1,
            max_duration_samples=max_len,
            max_featrues_len=self._get_len(max_len),
            window=400,
            hop=160,
            stride=2,
        )
        golden_stamps = torch.tensor([0, 320, 1000])
        assert (golden_stamps == time_stamps).all()

    def test_two_batchs(self):
        max_len = 1000
        time_stamps = generate_time_stamps(
            self._get_len(max_len) * 2,
            max_duration_samples=max_len,
            max_featrues_len=self._get_len(max_len),
            window=400,
            hop=160,
            stride=2,
        )
        golden_stamps = torch.tensor([0, 320, 1000, 1320])
        assert (golden_stamps == time_stamps).all()

    def test_two_batchs_with_residual(self):
        max_len = 1000
        time_stamps = generate_time_stamps(
            self._get_len(max_len) * 2 + 1,
            max_duration_samples=max_len,
            max_featrues_len=self._get_len(max_len),
            window=400,
            hop=160,
            stride=2,
        )
        print(time_stamps)
        golden_stamps = torch.tensor([0, 320, 1000, 1320, 2000])
        assert (golden_stamps == time_stamps).all()

    def test_three_batchs_with_residual(self):
        max_len = 2000
        print(self._get_len(max_len))
        time_stamps = generate_time_stamps(
            self._get_len(max_len) * 3 + 2,
            max_duration_samples=max_len,
            max_featrues_len=self._get_len(max_len),
            window=400,
            hop=160,
            stride=2,
        )
        print(time_stamps)
        golden_stamps = torch.tensor(
            [0, 320, 640, 960, 1280, 1600, 2000, 2320, 2640, 2960, 3280, 3600, 4000, 4320, 4640, 4960, 5280, 5600, 6000, 6320])
        assert (golden_stamps == time_stamps).all()


class TestExtractSpeechInervals:
    def test_merge_short_silence_and_no_pad(self):
        """Test merging short silence intervals and applying padding."""
        logits = torch.tensor([
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [1.0, 0.0],  # silence
            [0.0, 1.0],  # speech
        ])
        hop, stride = 160, 2
        time_stamps = torch.arange(len(logits)) * \
            hop * stride  # [0, 320, 640, 960]

        samples_out = extract_speech_intervals(
            logits=logits,
            time_stamps=time_stamps,
            speech_label=1,
            hop=hop,
            stride=stride,
        )
        output = clean_speech_intervals(
            samples_out.speech_intervals,
            samples_out.is_complete,
            min_silence_duration_ms=30,
            min_speech_duration_ms=30,
            pad_duration_ms=0,
        )

        print(output.clean_speech_intervals)
        print(output.speech_intervals)

        expected_clean = torch.tensor([[0, 1280]], dtype=torch.long)
        assert torch.allclose(output.clean_speech_intervals, expected_clean)
        assert output.is_complete is False

    def test_merge_short_silence_and_with_pad(self):
        """Test merging short silence intervals and applying padding."""
        logits = torch.tensor([
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [1.0, 0.0],  # silence
            [0.0, 1.0],  # speech
        ])
        hop, stride = 160, 2
        time_stamps = torch.arange(len(logits)) * \
            hop * stride  # [0, 320, 640, 960]

        samples_out = extract_speech_intervals(
            logits=logits,
            time_stamps=time_stamps,
            speech_label=1,
            hop=hop,
            stride=stride,
        )

        output = clean_speech_intervals(
            samples_out.speech_intervals,
            samples_out.is_complete,
            min_silence_duration_ms=30,
            min_speech_duration_ms=30,
            pad_duration_ms=30,
        )

        print(output.clean_speech_intervals)
        print(output.speech_intervals)

        expected_clean = torch.tensor([[0, 1760]], dtype=torch.long)
        assert torch.allclose(output.clean_speech_intervals, expected_clean)
        assert output.is_complete is False

    def test_no_speech_intervals(self):
        """Test when no speech intervals are detected."""
        logits = torch.tensor([[1.0, 0.0]] * 4)  # all silence
        time_stamps = torch.arange(4) * 160 * 2

        with pytest.raises(NoSpeechIntervals):
            samples_out = extract_speech_intervals(
                logits=logits,
                time_stamps=time_stamps,
                speech_label=1,
                silence_label=0,
            )

            clean_speech_intervals(
                samples_out.speech_intervals,
                samples_out.is_complete,
            )

    def test_too_high_min_speech_duration(self):
        """Test when min_speech_duration removes all intervals."""
        logits = torch.tensor([[0.0, 1.0]] * 2)  # two speech frames
        time_stamps = torch.tensor([0, 320])  # interval [0, 320] (320 samples)

        with pytest.raises(TooHighMinSpeechDuration):
            samples_out = extract_speech_intervals(
                logits=logits,
                time_stamps=time_stamps,
                speech_label=1,
            )

            clean_speech_intervals(
                samples_out.speech_intervals,
                samples_out.is_complete,
                min_speech_duration_ms=50,  # 800 samples
            )

    def test_return_seconds(self):
        """Test intervals returned in seconds."""
        logits = torch.tensor([[0.0, 1.0]] * 2)  # two speech frames
        # interval [0, 320] (0.02 seconds)
        time_stamps = torch.tensor([0, 320])

        samples_out = extract_speech_intervals(
            logits=logits,
            time_stamps=time_stamps,
            speech_label=1,
        )

        output = clean_speech_intervals(
            samples_out.speech_intervals,
            samples_out.is_complete,
            min_speech_duration_ms=20,  # 320 samples
            pad_duration_ms=30,  # 480 samples
            return_seconds=True,
        )

        # [0, 640] -> [0, 640 + 480] [0, 1120]

        expected_clean = torch.tensor([[0.0, 0.07]], dtype=torch.float32)
        assert torch.allclose(output.clean_speech_intervals, expected_clean)

    def test_padding_clamping(self):
        """Test padding clamping to avoid negative start."""
        logits = torch.tensor([[0.0, 1.0]] * 1)  # one speech frame
        time_stamps = torch.tensor([0])  # interval [0, 320] after appending

        samples_out = extract_speech_intervals(
            logits=logits,
            time_stamps=time_stamps,
            speech_label=1,
        )

        output = clean_speech_intervals(
            samples_out.speech_intervals,
            samples_out.is_complete,
            min_silence_duration_ms=1000,
            pad_duration_ms=30,  # 480 samples
            min_speech_duration_ms=0,
        )

        # [0, 320 + 480] -> [0, 800]

        expected_clean = torch.tensor([[0, 800]], dtype=torch.long)
        assert torch.allclose(output.clean_speech_intervals, expected_clean)

    def test_complete_intervals(self):
        """Test intervals ending with silence (is_complete=True)."""
        logits = torch.tensor([
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [1.0, 0.0],  # silence
            [1.0, 0.0],  # silence
        ])
        time_stamps = torch.arange(len(logits)) * 160 * 2

        output = extract_speech_intervals(
            logits=logits,
            time_stamps=time_stamps,
            speech_label=1,
        )

        assert output.is_complete is True

    def test_merge_multiple_silences(self):
        """Test merging multiple short silences."""
        logits = torch.tensor([
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [1.0, 0.0],  # silence
            [0.0, 1.0],  # speech
            [1.0, 0.0],  # silence
            [0.0, 1.0],  # speech
        ])
        time_stamps = torch.arange(len(logits)) * 160 * 2

        samples_out = extract_speech_intervals(
            logits=logits,
            time_stamps=time_stamps,
            speech_label=1,
        )

        output = clean_speech_intervals(
            samples_out.speech_intervals,
            samples_out.is_complete,
            min_silence_duration_ms=50,  # 800 samples
            min_speech_duration_ms=0,
            pad_duration_ms=0,
        )

        # [0, 1600 + 320] -> [0, 1920]
        print(output.clean_speech_intervals)

        # Expect single merged interval due to short silences
        assert output.clean_speech_intervals.shape[0] == 1
        expected_clean = torch.tensor([[0, 1920]], dtype=torch.long)
        assert torch.allclose(output.clean_speech_intervals, expected_clean)

    # TODO:
    def test_real_case_no_filters(self):
        """Test merging multiple short silences."""
        logits = torch.tensor([
            [0.0, 1.0],  # speech
            [1.0, 0.0],  # silence
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [1.0, 0.0],  # silence
            [1.0, 0.0],  # silence
            [1.0, 0.0],  # silence
            [1.0, 0.0],  # silence
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [1.0, 0.0],  # silence
            [1.0, 0.0],  # silence
            [0.0, 1.0],  # speech
        ])
        hop, stride = 10, 1
        time_stamps = torch.arange(len(logits)) * hop * stride

        samples_out = extract_speech_intervals(
            logits=logits,
            time_stamps=time_stamps,
            speech_label=1,
            hop=hop,
            stride=stride,
        )

        output = clean_speech_intervals(
            samples_out.speech_intervals,
            samples_out.is_complete,
            min_silence_duration_ms=0,
            min_speech_duration_ms=0,
            pad_duration_ms=0,
        )

        # [0, 1600 + 320] -> [0, 1920]
        print(output.clean_speech_intervals)

        expected_clean = torch.tensor([
            [0, 10],
            [20, 60],
            [100, 120],
            [140, 150],
        ],
            dtype=torch.long)
        assert torch.allclose(
            output.clean_speech_intervals, output.speech_intervals)
        assert torch.allclose(output.clean_speech_intervals, expected_clean)

    def test_real_case_silence_only(self):
        """Test merging multiple short silences."""
        logits = torch.tensor([
            [0.0, 1.0],  # speech
            [1.0, 0.0],  # silence
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [1.0, 0.0],  # silence
            [1.0, 0.0],  # silence
            [1.0, 0.0],  # silence
            [1.0, 0.0],  # silence
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [1.0, 0.0],  # silence
            [1.0, 0.0],  # silence
            [0.0, 1.0],  # speech
        ])
        hop, stride = 10, 1
        time_stamps = torch.arange(len(logits)) * hop * stride

        samples_out = extract_speech_intervals(
            logits=logits,
            time_stamps=time_stamps,
            speech_label=1,
            hop=hop,
            stride=stride,
        )

        output = clean_speech_intervals(
            samples_out.speech_intervals,
            samples_out.is_complete,
            min_silence_duration_ms=1.25,  # 20 samples
            min_speech_duration_ms=0,
            pad_duration_ms=0,
        )

        print(output.clean_speech_intervals)
        expected_clean = torch.tensor([
            [0, 60],
            [100, 120],
            [140, 150],
        ],
            dtype=torch.long)
        assert torch.allclose(output.clean_speech_intervals, expected_clean)

    def test_real_case_silence_collaspe_all(self):
        """Test merging multiple short silences."""
        logits = torch.tensor([
            [0.0, 1.0],  # speech
            [1.0, 0.0],  # silence
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [1.0, 0.0],  # silence
            [1.0, 0.0],  # silence
            [1.0, 0.0],  # silence
            [1.0, 0.0],  # silence
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [1.0, 0.0],  # silence
            [1.0, 0.0],  # silence
            [0.0, 1.0],  # speech
        ])
        hop, stride = 10, 1
        time_stamps = torch.arange(len(logits)) * hop * stride

        samples_out = extract_speech_intervals(
            logits=logits,
            time_stamps=time_stamps,
            speech_label=1,
            hop=hop,
            stride=stride,
        )

        output = clean_speech_intervals(
            samples_out.speech_intervals,
            samples_out.is_complete,
            min_silence_duration_ms=5,  # 80 samples
            min_speech_duration_ms=0,
            pad_duration_ms=0,
        )

        print(output.clean_speech_intervals)
        expected_clean = torch.tensor([
            [0, 150],
        ],
            dtype=torch.long)
        assert torch.allclose(output.clean_speech_intervals, expected_clean)

    def test_real_case_silence_speech(self):
        """Test merging multiple short silences."""
        logits = torch.tensor([
            [0.0, 1.0],  # speech
            [1.0, 0.0],  # silence
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [1.0, 0.0],  # silence
            [1.0, 0.0],  # silence
            [1.0, 0.0],  # silence
            [1.0, 0.0],  # silence
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [1.0, 0.0],  # silence
            [1.0, 0.0],  # silence
            [0.0, 1.0],  # speech
        ])
        hop, stride = 10, 1
        time_stamps = torch.arange(len(logits)) * hop * stride

        samples_out = extract_speech_intervals(
            logits=logits,
            time_stamps=time_stamps,
            speech_label=1,
            hop=hop,
            stride=stride,
        )

        output = clean_speech_intervals(
            samples_out.speech_intervals,
            samples_out.is_complete,
            min_silence_duration_ms=1.25,  # 20 samples
            min_speech_duration_ms=11/16,  # 11 samples
            pad_duration_ms=0,
        )

        print(output.clean_speech_intervals)
        expected_clean = torch.tensor([
            [0, 60],
            [100, 120],
        ],
            dtype=torch.long)
        assert torch.allclose(output.clean_speech_intervals, expected_clean)

    def test_real_case_silence_speech_padd(self):
        """Test merging multiple short silences."""
        logits = torch.tensor([
            [0.0, 1.0],  # speech
            [1.0, 0.0],  # silence
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [1.0, 0.0],  # silence
            [1.0, 0.0],  # silence
            [1.0, 0.0],  # silence
            [1.0, 0.0],  # silence
            [0.0, 1.0],  # speech
            [0.0, 1.0],  # speech
            [1.0, 0.0],  # silence
            [1.0, 0.0],  # silence
            [0.0, 1.0],  # speech
        ])
        hop, stride = 10, 1
        time_stamps = torch.arange(len(logits)) * hop * stride

        samples_out = extract_speech_intervals(
            logits=logits,
            time_stamps=time_stamps,
            speech_label=1,
            hop=hop,
            stride=stride,
        )

        output = clean_speech_intervals(
            samples_out.speech_intervals,
            samples_out.is_complete,
            min_silence_duration_ms=20/16,  # 20 samples
            min_speech_duration_ms=11/16,  # 11 samples
            pad_duration_ms=30/16,  # 30 samples
        )

        print(output.clean_speech_intervals)
        expected_clean = torch.tensor([
            [0, 90],
            [70, 150],
        ],
            dtype=torch.long)
        assert torch.allclose(output.clean_speech_intervals, expected_clean)


def test_segment_recitations():
    device = torch.device('cpu')
    dtype = torch.bfloat16
    processor = AutoFeatureExtractor.from_pretrained(
        "obadx/recitation-segmenter-v2")
    model = AutoModelForAudioFrameClassification.from_pretrained(
        "obadx/recitation-segmenter-v2",
    )

    model.to(device, dtype=dtype)

    file_path = './assets/hussary_053001.mp3'
    wav = read_audio(file_path)
    print(wav.shape)

    samples_out = segment_recitations(
        [wav],
        model,
        processor,
        device=device,
        dtype=dtype,
        batch_size=1,
        max_duration_ms=2000,
    )

    output = clean_speech_intervals(
        samples_out[0].speech_intervals,
        samples_out[0].is_complete,
        min_silence_duration_ms=30,
        min_speech_duration_ms=30,
        pad_duration_ms=30,
        return_seconds=True,
    )

    print(output.clean_speech_intervals)
    assert output.clean_speech_intervals.shape == (1, 2)


def test_cli():
    shutil.rmtree('./output', ignore_errors=True)
    subprocess.run([
        'recitations-segmenter',
        './assets/hussary_053001.mp3',
        '--max-duration-ms', '2000',
        '-o', 'output',
    ])

    file_path = Path('./output/hussary_053001_speech_intervals.json')
    assert file_path.is_file()
    with open(file_path, 'r') as f:
        data = json.load(f)

    assert data['is_complete']
    assert len(data['clean_speech_intervals']) == 1
    assert len(data['speech_intervals']) == 1
