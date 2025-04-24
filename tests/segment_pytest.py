import torch
import pytest

from recitations_segmenter.segment import batchify_input, WavInfo, collect_results
from transformers import AutoFeatureExtractor
import numpy as np


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

    def test_device_check(self):
        waves = [torch.ones(5).to('cuda')]  # Input on GPU
        max_len = 5
        batch_size = 1
        _, batches = batchify_input(waves, max_len, batch_size)

        for batch in batches:
            assert batch.device == torch.device('cpu')


class TestCollect:
    processor = AutoFeatureExtractor.from_pretrained('facebook/w2v-bert-2.0')

    def _get_prcossor_len(self, L):
        out_l = self.processor(
            np.zeros(L),
            return_tensors='pt',
            sampling_rate=16000,
        )['attention_mask'][0].sum()
        return out_l

    def _get_golden_logits(self, waves: list[torch.FloatTensor]):
        out_logits = []
        for wav in waves:
            logits_len = self._get_prcossor_len(len(wav))
            out_logits.append(torch.arange(
                logits_len).repeat_interleave(2).reshape(-1, 2))
        return out_logits

    def _get_batched_logits(
        self,
        waves: list[torch.FloatTensor],
        max_len,
        batch_size,
    ):
        out_logits = []
        # logits are shape of batch, len, 2
        max_len = self._get_prcossor_len(max_len) * 2
        wav_logits = []
        for wav in waves:
            logits_len = self._get_prcossor_len(len(wav))
            wav_logits.append(torch.arange(logits_len).repeat_interleave(2))

        _, batched_logits = batchify_input(
            wav_logits, max_len, max_batch_size=batch_size)
        batched_logits = list(batched_logits)

        for idx in range(len(batched_logits)):
            batch_size, _ = batched_logits[idx].shape
            batched_logits[idx] = batched_logits[idx].reshape(
                batch_size, -1, 2)

        return batched_logits

    def _check_collect(
        self,
        waves,
        wav_infos: list[WavInfo],
        max_len,
        batch_size
    ):
        golden_logits = self._get_golden_logits(waves)
        batched_logits = self._get_batched_logits(waves, max_len, batch_size)

        out_logits = collect_results(wav_infos, batched_logits)

        print(golden_logits)
        print(out_logits)

        assert len(golden_logits) == len(out_logits)
        for idx in range(len(batched_logits)):
            assert len(golden_logits[idx]) == len(out_logits[idx])
            assert (golden_logits[idx] == out_logits[idx]).all()

    # def test_single_waveform_no_padding(self):
    #     waves = [torch.ones(5000)]
    #     max_len = 5000
    #     batch_size = 1
    #     wav_infos, batches = batchify_input(waves, max_len, batch_size)
    #
    #     assert len(wav_infos) == 1
    #     info = wav_infos[0]
    #     assert info == WavInfo(
    #         wav_len=5000,
    #         batch_start=0,
    #         batch_end=1,
    #         idx_in_batch_start=0,
    #         idx_in_batch_end=1
    #     )
    #     assert len(batches) == 1
    #     assert batches[0].shape == (1, max_len)
    #     assert torch.all(batches[0] == 1)
    #
    #     self._check_collect(waves, wav_infos, max_len, batch_size)
    #
    # def test_single_waveform_with_padding(self):
    #     waves = [torch.ones(3000)]
    #     max_len = 5000
    #     batch_size = 1
    #     wav_infos, batches = batchify_input(waves, max_len, batch_size)
    #
    #     info = wav_infos[0]
    #     padded_wav = torch.cat((torch.ones(3000), torch.zeros(2000)))
    #     assert info == WavInfo(
    #         wav_len=3000,
    #         batch_start=0,
    #         batch_end=1,
    #         idx_in_batch_start=0,
    #         idx_in_batch_end=1
    #     )
    #     assert batches[0].shape == (1, max_len)
    #     assert torch.all(batches[0][0] == padded_wav)
    #
    #     self._check_collect(waves, wav_infos, max_len, batch_size)

    def test_exact_batch_fill(self):
        waves = [torch.ones(16000)]
        max_len = 8000
        batch_size = 2
        wav_infos, batches = batchify_input(waves, max_len, batch_size)

        info = wav_infos[0]
        assert info == WavInfo(
            wav_len=16000,
            batch_start=0,
            batch_end=1,
            idx_in_batch_start=0,
            idx_in_batch_end=2  # Expected failure: code returns 0
        )
        assert len(batches) == 1
        assert batches[0].shape == (2, max_len)

        self._check_collect(waves, wav_infos, max_len, batch_size)

    # def test_chunks_span_multiple_batches(self):
    #     waves = [torch.ones(15000)]
    #     max_len = 5000
    #     batch_size = 2
    #     wav_infos, batches = batchify_input(waves, max_len, batch_size)
    #
    #     info = wav_infos[0]
    #     assert info == WavInfo(
    #         wav_len=15000,
    #         batch_start=0,
    #         batch_end=2,
    #         idx_in_batch_start=0,
    #         idx_in_batch_end=1
    #     )
    #     assert len(batches) == 2
    #     assert batches[0].shape == (2, max_len)
    #     assert batches[1].shape == (1, max_len)
    #
    #     self._check_collect(waves, wav_infos, max_len, batch_size)
    #
    # def test_multiple_waveforms(self):
    #     wave1 = torch.ones(5000)
    #     wave2 = torch.ones(8000)
    #     waves = [wave1, wave2]
    #     max_len = 5000
    #     batch_size = 2
    #     wav_infos, batches = batchify_input(waves, max_len, batch_size)
    #
    #     # Validate first waveform
    #     info1 = wav_infos[0]
    #     assert info1 == WavInfo(
    #         wav_len=5000,
    #         batch_start=0,
    #         batch_end=1,
    #         idx_in_batch_start=0,
    #         idx_in_batch_end=1
    #     )
    #
    #     # Validate second waveform
    #     info2 = wav_infos[1]
    #     assert info2 == WavInfo(
    #         wav_len=8000,
    #         batch_start=0,
    #         batch_end=2,
    #         idx_in_batch_start=1,
    #         idx_in_batch_end=1
    #     )
    #
    #     assert len(batches) == 2
    #     assert batches[0].shape == (2, max_len)
    #     assert batches[1].shape == (1, max_len)
    #
    #     self._check_collect(waves, wav_infos, max_len, batch_size)
    #
    # def test_multiple_waveforms_in_single_batch(self):
    #     wave1 = torch.ones(5000)
    #     wave2 = torch.ones(8000)
    #     waves = [wave1, wave2]
    #     max_len = 5000
    #     batch_size = 5
    #     wav_infos, batches = batchify_input(waves, max_len, batch_size)
    #
    #     # Validate first waveform
    #     info1 = wav_infos[0]
    #     assert info1 == WavInfo(
    #         wav_len=5000,
    #         batch_start=0,
    #         batch_end=1,
    #         idx_in_batch_start=0,
    #         idx_in_batch_end=1
    #     )
    #
    #     # Validate second waveform
    #     info2 = wav_infos[1]
    #     assert info2 == WavInfo(
    #         wav_len=8000,
    #         batch_start=0,
    #         batch_end=1,
    #         idx_in_batch_start=1,
    #         idx_in_batch_end=3
    #     )
    #
    #     assert len(batches) == 1
    #     assert batches[0].shape == (3, max_len)
    #
    #     self._check_collect(waves, wav_infos, max_len, batch_size)
    #
    # def test_multiple_waveforms_over_multiple_batches(self):
    #     wave1 = torch.ones(32000)
    #     wave2 = torch.ones(18000)
    #     wave3 = torch.ones(11000)
    #     waves = [wave1, wave2, wave3]
    #     max_len = 8000
    #     batch_size = 4
    #     wav_infos, batches = batchify_input(waves, max_len, batch_size)
    #
    #     # Validate first waveform
    #     info1 = wav_infos[0]
    #     assert info1 == WavInfo(
    #         wav_len=32000,
    #         batch_start=0,
    #         batch_end=1,
    #         idx_in_batch_start=0,
    #         idx_in_batch_end=4
    #     )
    #
    #     # Validate second waveform
    #     info2 = wav_infos[1]
    #     assert info2 == WavInfo(
    #         wav_len=18000,
    #         batch_start=1,
    #         batch_end=2,
    #         idx_in_batch_start=0,
    #         idx_in_batch_end=3
    #     )
    #
    #     info3 = wav_infos[2]
    #     assert info3 == WavInfo(
    #         wav_len=11000,
    #         batch_start=1,
    #         batch_end=3,
    #         idx_in_batch_start=3,
    #         idx_in_batch_end=1
    #     )
    #
    #     assert len(batches) == 3
    #     assert batches[0].shape == (4, max_len)
    #     assert batches[1].shape == (4, max_len)
    #     assert batches[2].shape == (1, max_len)
    #
    #     self._check_collect(waves, wav_infos, max_len, batch_size)
