# Recitations Segmenter

<div align="center">
<strong>بتوفيق الله: نموذج ذكاء اصطناعي قادر على تقطيع التلاوات القرآنية على حسب الوقف بدقة عالية</strong>

[![Tests][tests-badge]][tests-url]
[![PyPI][pypi-badge]][pypi-url]
[![Python Versions][python-badge]][python-url]
[![Hugging Face][hf-badge]][hf-url]
[![Google Colab][colab-badge]][colab-url]
[![MIT License][mit-badge]][mit-url]

</div>

[tests-badge]: https://img.shields.io/github/actions/workflow/status/obadx/recitations-segmenter/tests.yml?branch=main&label=tests
[tests-url]: https://github.com/obadx/recitations-segmenter/actions/workflows/tests.yml
[pypi-badge]: https://img.shields.io/pypi/v/recitations-segmenter.svg
[pypi-url]: https://pypi.org/project/recitations-segmenter/
[mit-badge]: https://img.shields.io/github/license/obadx/recitations-segmenter.svg
[mit-url]: https://github.com/obadx/recitations-segmenter/blob/main/LICENSE
[python-badge]: https://img.shields.io/pypi/pyversions/recitations-segmenter.svg
[python-url]: https://pypi.org/project/recitations-segmenter/
[colab-badge]: https://img.shields.io/badge/Google%20Colab-Open%20in%20Colab-F9AB00?logo=google-colab&logoColor=white
[colab-url]: https://colab.research.google.com/drive/1-RuRQOj4l2MA_SG2p4m-afR7MAsT5I22?usp=sharing
[hf-badge]: https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue
[hf-url]: https://huggingface.co/obadx/recitation-segmenter-v2



بفضل الله نقدم نموذج اصطناعي محَّسن من [W2v2Bert](https://huggingface.co/docs/transformers/model_doc/wav2vec2-bert) على أساس مهمة Sequence Frame Level Classification بدقة 20 ملي ثانية (20 milliseconds)  ومعه أيضا مكتبة python تعمل بأداء عالي لأي عدد وأي طول (صغير أم كبير) من التلاوات القرآنية


![VAD Architecture](./assets/vad-arch.svg)

## المميزات

* تقطيع التلاوات القرآنية على حسب الوقف

* مدَّرب خصيصا للتلاوات القرآنية

* بدقة عالية تصل إلى 20 ملي ثانية

* تحتاج فقط إلي 2 GB من ذاكرة الرسوميات (GPU Memory)

* يستطيع تقطيع  التلاوات القرآنية لأي مدة من التلاوات من بضع دقايق إلى عدة ساعات من غير نقص في الأداء

## النموذج على Hugging Face 🤗
* [النموذج](https://huggingface.co/obadx/recitation-segmenter-v2)
* [بيانات التدريب] (https://huggingface.co/datasets/obadx/recitation-segmentation)
* [بيانات التدريب مع إضافة augmentation](https://huggingface.co/datasets/obadx/recitation-segmentation-augmented)

<!-- TOC -->
<!-- vim-markdown-toc Marked -->
## قائمة المحتويات

* [تثبيت المكتبة](#تثبيت-المكتبة)
    * [متطلبات التثبيت](#متطلبات-التثبيت)
    * [تثبيت المكتبة](#تثبيت-المكتبة)
* [API  باستخدام Python](#api-باستخدام-python)
* [Command Line Interface](#command-line-interface)
    * [وصف مفصل لل Command Line](#وصف-مفصل-لل-command-line)
* [توثيق المكتبة (API Refernece)](#توثيق-المكتبة-(api-refernece))
* [تفاصيل التدريب](#تفاصيل-التدريب)
    * [دوافع تدريب نموذج جديد وعدم استخدام الطرق الحالية](#دوافع-تدريب-نموذج-جديد-وعدم-استخدام-الطرق-الحالية)
    * [طريقة حل المشكلة](#طريقة-حل-المشكلة)
    * [تهيئة بيئة التطوير](#تهيئة-بيئة-التطوير)
    * [بيانات التدريب](#بيانات-التدريب)
    * [طريقة تجميع البيانات](#طريقة-تجميع-البيانات)
    * [تهيئة البيانات](#تهيئة-البيانات)
    * [التدريب](#التدريب)
    * [النتائج:](#النتائج:)
* [ملاحظات مهمة](#ملاحظات-مهمة)
* [TODO](#todo)

<!-- vim-markdown-toc -->

## تثبيت المكتبة

### متطلبات التثبيت

تثبيت مكتبتي:

* [ffmbeg](https://ffmpeg.org/download.html)
* [libsoundfile](https://github.com/libsndfile/libsndfile/releases)

#### Linux

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1 portaudio19-dev
```

#### Winodws & Mac

يمكنك إنشاء بيئة `anaconda` . ومن ثم تنزيل هاتين المكتبتين

```bash
conda create -n segment python=3.12
conda activate segment
conda install -c conda-forge ffmpeg libsndfile
```

### تثبيت المكتبة

####  باستخدام pip

```bash
pip install recitations-segmenter
```

####  باستخدام uv

```bash
uv add recitations-segmenter
```

## API  باستخدام Python

موضح أدناه مثال كامل لاتسخدام المكتبة بال python ويوجد أيضا مثال داخل Google Colab:

[![Google Colab][colab-badge]][colab-url]

```python
from pathlib import Path

from recitations_segmenter import segment_recitations, read_audio, clean_speech_intervals
from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification
import torch

if __name__ == '__main__':
    device = torch.device('cuda')
    dtype = torch.bfloat16

    processor = AutoFeatureExtractor.from_pretrained(
        "obadx/recitation-segmenter-v2")
    model = AutoModelForAudioFrameClassification.from_pretrained(
        "obadx/recitation-segmenter-v2",
    )

    model.to(device, dtype=dtype)

    # Change this to the file pathes of Holy Quran recitations
    # File pathes with the Holy Quran Recitations
    file_pathes = [
        './assets/dussary_002282.mp3',
        './assets/hussary_053001.mp3',
    ]
    waves = [read_audio(p) for p in file_pathes]

    # Extracting speech inervals in samples according to 16000 Sample rate
    sampled_outputs = segment_recitations(
        waves,
        model,
        processor,
        device=device,
        dtype=dtype,
        batch_size=8,
    )

    for out, path in zip(sampled_outputs, file_pathes):
        # Clean The speech intervals by:
        # * merging small silence durations
        # * remove small speech durations
        # * add padding to each speech duration
        # Raises:
        # * NoSpeechIntervals: if the wav is complete silence
        # * TooHighMinSpeechDruation: if `min_speech_duration` is too high which
        # resuls for deleting all speech intervals
        clean_out = clean_speech_intervals(
            out.speech_intervals,
            out.is_complete,
            min_silence_duration_ms=30,
            min_speech_duration_ms=30,
            pad_duration_ms=30,
            return_seconds=True,
        )

        print(f'Speech Intervals of: {Path(path).name}: ')
        print(clean_out.clean_speech_intervals)
        print(f'Is Recitation Complete: {clean_out.is_complete}')
        print('-' * 40)
```



## Command Line Interface

يمكنك مباشرة استخدام المكتبة من وبدون تثبيت المكتبة عن طريق:

```bash
uvx recitations-segmenter alfateha.mp3 
```

أو بعد التثبيت باستخدام: 

```bash
recitations-segmenter alfateha.mp3 
```

سيتم استخراج توقيتات التلاوات على حسب الوقف على هيئتين: 

### في ال terminal
```text
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.04s/it]
Speech Intervals:
tensor([[ 0.7300,  5.2900],
        [ 6.5100, 10.9900],
        [12.4700, 17.2100],
        [18.1500, 21.6850],
        [22.6850, 26.2650],
        [27.4450, 33.2050],
        [34.2650, 38.6250],
        [39.8250, 53.3200]])

```
### وعلى هيئة ملف JSON في المسار : `output/speech_intervals_alfateha.json`

سيتم استخراج توقيتات لكل مقطع قرآني بداخل سورة الفاتة في المسار : `output` ويحتوي على ملف `speech_intervals_alfateha.json`. ويحتوي هذا الملف على الآتي:

```json
{
    "clean_speech_intervals": [
        [
            0.7300000190734863,
            5.289999961853027
        ],
        [
            6.510000228881836,
            10.989999771118164
        ],
        [
            12.470000267028809,
            17.209999084472656
        ],
        [
            18.149999618530273,
            21.684999465942383
        ],
        [
            22.684999465942383,
            26.264999389648438
        ],
        [
            27.44499969482422,
            33.20500183105469
        ],
        [
            34.26499938964844,
            38.625
        ],
        [
            39.82500076293945,
            53.31999969482422
        ]
    ],
    "speech_intervals": [
        [
            0.7599999904632568,
            5.260000228881836
        ],
        [
            6.539999961853027,
            10.960000038146973
        ],
        [
            12.5,
            17.18000030517578
        ],
        [
            18.18000030517578,
            21.655000686645508
        ],
        [
            22.71500015258789,
            26.235000610351562
        ],
        [
            27.475000381469727,
            33.17499923706055
        ],
        [
            34.29499816894531,
            38.595001220703125
        ],
        [
            39.85499954223633,
            53.290000915527344
        ]
    ],
    "is_complete": true
}
```

يتضمن كل ملف JSON على هذه المفاتيح:

* `clean_speech_intervals`:     التوقيتات بالثانية لبداية ونهاية كل مقطع بعد التنقيح
* `speech_intervals`: التوقيتات بالثانية لبداية ونهاية كل مقطع 
*  `is_complete`: هل التلاوة القرآنية تامة أم أن آخر المقطع لا يتضمن وقفا محضا



### وصف مفصل لل Command Line




```text
usage: recitations-segmenter [-h] [-o OUTPUT] [--min-silence-duration-ms MIN_SILENCE_DURATION_MS] [--min-speech-duration-ms MIN_SPEECH_DURATION_MS] [--pad-duration-ms PAD_DURATION_MS]
                             [--return-samples] [--batch-size BATCH_SIZE] [--max-duration-ms MAX_DURATION_MS] [--device {cpu,cuda}] [--dtype {bfloat16,float16,float32}]
                             inputs [inputs ...]

Segment Holy Quran rectiations into speech intervals based on وقف using Wav2Vec2Bert model.

options:
  -h, --help            show this help message and exit

Input/Output Options:
  inputs                Input paths (files or directories) containing audio files
  -o OUTPUT, --output OUTPUT
                        Output directory for JSON results (default: ./output)

Segmentation Parameters:
  --min-silence-duration-ms MIN_SILENCE_DURATION_MS
                        Minimum silence duration (ms) between speech segments
                        - Silence shorter than this will be merged with speech
                        - Default: 30ms
  --min-speech-duration-ms MIN_SPEECH_DURATION_MS
                        Minimum valid speech duration (ms)
                        - Speech segments shorter than this will be removed
                        - Default: 30ms
  --pad-duration-ms PAD_DURATION_MS
                        Padding added around speech segments (ms)
                        Default: 30ms
  --return-samples      Return intervals in samples according to 16000 sampling rate.
                        - Default to return interval in seconds

Model Configuration:
  --batch-size BATCH_SIZE
                        Number of audio chunks processed simultaneously
                        - Higher values may increase speed but require more GPU memory.
                        - Default: 8 which occupies nearly 3GB of GPU memory.
  --max-duration-ms MAX_DURATION_MS
                        Maximum processing chunk duration (2-20000ms)
                        - Affects memory usage and processing granularity
                        - Do not Change it unless there exists a strong reason.
                        - Default: 19995ms
  --device {cpu,cuda}   Processing device selection
                        Default: cuda
  --dtype {bfloat16,float16,float32}
                        Numerical precision for model computation:
                        - bfloat16: Best performance (modern GPUs)
                        - float16: Legacy support
                        - float32: Maximum precision (CPU fallback)
                        Default: bfloat16

Examples:
  # Process single file with default settings
  recitations-segmenter input.mp3 -o results

  # Process multiple files file with default settings
  recitations-segmenter input1.mp3 input2.wav -o output

  # Process directory of audio files
  recitations-segmenter /path/to/recitations/ --output ./segmentation_results

  # Process: audio files and directory of audio files
  recitations-segmenter input.mp3 /path/to/recitations/ --output ./segmentation_results

  # Adjust segmentation parameters
  recitations-segmenter input.wav --min-silence-duration-ms 200 --min-speech-duration-ms 900 --pad-duration-ms 40

File Formats Supported:
  MP3, WAV, FLAC, OGG, AAC, M4A, OPUS

Output Format:
  Each input file generates a JSON file containing:
  - clean_speech_intervals: Final filtered speech segments
  - speech_intervals: Raw detected speech segments
  - is_complete: whether the recitaion is a complete وقف or the recitation is contining (has not stoped yet)

Error Handling:
  - Skips unsupported file types


```



## توثيق المكتبة (API Refernece)

### `segment_recitations`

```python
@torch.no_grad()
def segment_recitations(
    waves: list[torch.FloatTensor],
    model: Wav2Vec2BertForAudioFrameClassification,
    processor: Wav2Vec2BertProcessor,
    batch_size=64,
    device=torch.device('cpu'),
    dtype=torch.bfloat16,
    return_probabilities=False,
    sample_rate=16000,
    processor_window=400,
    processor_hop=160,
    processor_stride=2,
    max_duration_ms=19995,
    speech_label=1,
    silence_label=0,
    cache_dir: Optional[str | Path] = None,
    overwrite_cache: Optional[bool] = False,
) -> list[W2vBSegmentationOutput]:

```

Segment The Holy Quran rectiations into speech intervals based on وقف using Wav2Vec2Bert model.

**Arguments**:
- `waves` (`list[torch.FloatTensor]`): List of audio waveforms to process (each as FloatTensor)
- `model` (`Wav2Vec2BertForAudioFrameClassification`): Loaded Wav2Vec2BertForAudioFrameClassification model
- `processor` (`Wav2Vec2BertProcessor`): Wav2Vec2BertProcessor for feature extraction
- `batch_size` (`int`): Number of samples per batch
- `sample_rate` (`int`): Input audio sampling rate (must be 16000)
- `processor_window` (`int`): Processor window size (fixed at 400 samples)
- `processor_hop` (`int`): Processor hop length (fixed at 160 samples)
- `processor_stride` (`int`): Processor stride (fixed at 2)
- `max_duration_ms` (`int`): Maximum chunk duration in ms for processing (2-20000)
- `speech_label` (`int`): Class index for speech segments
- `silence_label` (`int`): Class index for silence segments
- `device` (`torch.device`): Torch device for inference
- `dtype` (`torch.dtype`): Data type for model computation only. Default it `torch.bfloat16` for post processing we use `torch.float32`
- `return_probabilities` (`bool`): Whether to return class probabilities
- `cache_dir` (`Optional[str | Path]`): Optional feature disabled by default: if it is not `None`. Saving speech intervals to the `cach_dir` so next time for inference with the sample input `waves` we did not have to recompute the speech_intervals
- `overwrite_cache` (`Optional[bool]`): if there exists a `cache_dir` overwrite it.

**Returns**:
- `list[W2vBSegmentationOutput]`:
-  Every `W2vBSegmentationOutput` is:
  - `clean_speech_intervals` (`torch.FloatTensor`):  `None`.
  - `speech_intervals` (`torch.FloatTensor`): Tensor of shape (N, 2) containing raw speech intervals before filtering. Format: `[[speech_start, speech_end], [speech_start, speech_end], ...]` in samples.
  - `probs` (`torch.FloatTensor | None`): Class probabilities (None if not requested)
  - `is_complete` (`bool`): Whether audio processing completed normally

**Note**:
- Processes audio in chunks of max_duration_ms for GPU memory efficiency
- Input waveforms are automatically padded and batched
- Final interval end is clamped to (audio_length + hop*stride) if not provided

---

### `clean_speech_intervals`

```python
def clean_speech_intervals(
    speech_intervals: torch.LongTensor,
    is_complete: bool,
    min_silence_duration_ms=30,
    min_speech_duration_ms=30,
    pad_duration_ms=30,
    sample_rate=16000,
    return_probabilities=False,
    return_seconds=False,
) -> W2vBSegmentationOutput:
```

Permores cleaning on raw speech intervals extracted by the model. Clean The speech intervals by:
* merging small silence durations.
* remove small speech durations.
* add padding to each speech duration.

**Arguments**:
- `speech_intervals` (`torch.LongTensor`): Tensor of shape (N, 2) containing raw speech intervals before filtering. Format: `[[speech_start, speech_end], [speech_start, speech_end], ...]` in samples.
- `is_complete` (`bool`): Whether audio processing completed normally
- `min_silence_duration_ms` (`int`): Minimum silence duration (ms) between speech segments. silence durations < `min_silence_duration_ms` will be merged into speech durations
- `min_speech_duration_ms` (`int`): Minimum duration (ms) for a valid speech segment. speech intervals durations < `min_speech_duration_ms` will be removed
- `pad_duration_ms` (`int`): Padding duration (ms) to add around speech segments
- `sample_rate` (`int`): Audio sample rate in Hz
- `return_probabilities` (`bool`): Whether to return class probabilities
- `return_seconds` (`bool`): Whether to return intervals in seconds instead of samples

**Returns**:
- `W2vBSegmentationOutput`:
  - `clean_speech_intervals` (`torch.FloatTensor`): Tensor of shape (N, 2) containing speech intervals after filtering. Format: `[[speech_start, speech_end], ...]` in samples if `return_seconds` is `false`. otherwise return the speech inervals in seconds.
  - `speech_intervals` (`torch.FloatTensor`): Tensor of shape (N, 2) containing raw speech intervals before filtering. Format: `[[speech_start, speech_end], ...]` in samples if `return_seconds` is `false`. otherwise return the speech inervals in seconds
  - `probs` (`torch.FloatTensor | None`): Class probabilities (None if not requested)
  - `is_complete` (`bool`): Whether audio processing completed normally

**Raises**:
- `NoSpeechIntervals`: If no speech segments are detected
- `TooHighMinSpeechDuration`: If filtering removes all speech segments

**Note**:
- Intervals are clamped to prevent negative starts or exceeding audio length
- Final interval end is clamped to (audio_length + hop*stride) if not provided

### Data Structures

### `W2vBSegmentationOutput`
Named tuple containing:
- `clean_speech_intervals`: `torch.Tensor` or `None`
- `speech_intervals`: `torch.Tensor`
- `probs`: `torch.Tensor` or `None`
- `is_complete`: `bool`

### Exceptions
- `NoSpeechIntervals`: Raised when input contains no speech
- `TooHighMinSpeechDuration`: Raised when filtering removes all segments

##  تفاصيل التدريب

### دوافع تدريب نموذج جديد وعدم استخدام الطرق الحالية
كان الهدف هو تقطيع التلاوات القرآنية على حسب الوقف بجودة عالية ودقة تصل ل 50 ملي ثانية باستخدام تقنية Voice Activity Detectoin (VAD) وتم تجربة:

* نموذج [sliero-vad-v5](https://github.com/snakers4/silero-vad) وللأسف كان سيئا جدا على الرغم من أن دقته تصل 32 ملي ثانية
* نموذج [sliero-vad-v4](https://github.com/snakers4/silero-vad/tree/v4.0stable) أفضل أداءا من النسخة الأخيرة على بعض التلاوات القرآنية ودقتخ تبلغ 95 ملي ثانية
* نمذوج pyannotate كان سيئا على الإطلاق 
تم تجربة نماذج أخرى كانت سيئة جدا

### طريقة حل المشكلة

الهدف في هو تقطيع التلاوات القرآنية على حسب الوقف لبناء قواعد بيانات قرآنية. فالهدف ليس ال streaming بل هو بناء قواعد بيانات من التلاوات القرآنية. ومن ثم فحجم النموذج لا يلزمه الكبر أو الصغر بال الدقة والجودة أهم الأشياء فوع الاختيار على [w2v2Bert](https://huggingface.co/docs/transformers/model_doc/wav2vec2-bert) لأنه:

* مدرب على 4.5 مليون ساعة متعدة اللهجات واللغات
* النموذج مدَّرب على أكثر من 100 ألف ساعة من الأصوات العربية
* صغر نافذته حيث كل نافذة من مستخرج المميزات (feature extractor) تبلغ 20 ملي ثانية

ومن هاهنا تم تدريب [w2v2Bert](https://huggingface.co/docs/transformers/model_doc/wav2vec2-bert) ك sequence labeling لكل نافذة على تلاوات قرآنية شبه معلّمة تلقائيا بساتخدام أفضل VAD تم الحصول عليه.


### تهيئة بيئة التطوير

#### تثبيت المتطلبات

تثبيت مكتبتي:

* [ffmbeg](https://ffmpeg.org/download.html)
* [libsoundfile](https://github.com/libsndfile/libsndfile/releases)

##### Linux
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1 portaudio19-dev
```

##### Winodws & Mac

يمكنك إنشاء بيئة `anaconda` . ومن ثم تنزيل هاتين المكتبتين

####  تثبيت بيئة التطوير

First of all glone the repo

```bash
git clone https://github.com/obadx/recitations-segmenter.git
```

```bash
cd recitations-segmenter
```

Create conda environment with python 3.12

```bash
conda create -n segment12 python=3.12
conda activate segment12
```


> Note: for data builindg we worked on python 3.13 and for augmentations we worked to python 3.12 due to audiomentatiosn depends on scipy

Install `ffmbeg` and `scipy` using conda

```bash
conda install -c conda-forge ffmpeg scipy=1.15.2

```

Install our package

```bash
pip install -e ./[agument]
```


### بيانات التدريب

### طريقة تجميع البيانات

* كان أفضل ال VAD أداءا هو [sliero-vad-v4](https://github.com/snakers4/silero-vad/tree/v4.0stable) فتم اختيار المصاحف القرآنية من [everyayh](everyayah.com)
* وبعد ذلك تم عمل دالة تقوم بتعويض عيوب النموذج عن طريق إضافة:
  - `min_silence_duration_ms`: تقوم بدمج المقاطع التي تحتوي على صمت مع المقاطع اللتي تحتوي على صوت
  - `min_speech_duration_ms`: تقوم بحذف المقاطع اللتي تحتوي على صوت
  ومع أيضا بعض المتغيرات الأخرى انظر [هنا](./src/recitations_segmenter/train/vad_utils.py)
  * وبعد ذلك تم تحديد تلك المتغيرات يدويا لتقسم التلاوت بدقة واستبعاد التلاوات التي فشل sliro-vad-v4 فيها
  * ومن ثم استقر التجميع على تلك [المصاحف](./recitations.yml)
  
  ```yml
  recitations:
    - reciter_name: محمود خليل الحصري
      id: 0
      url: https://everyayah.com/data/Husary_128kbps/000_versebyverse.zip
      window_size_samples: 1536
      threshold: 0.3
      min_silence_duration_ms: 500
      min_speech_duration_ms: 1000
      pad_duration_ms: 40
  
    - reciter_name: محمد صديق المنشاوي
      id: 1
      url: https://everyayah.com/data/Minshawy_Murattal_128kbps/000_versebyverse.zip
      window_size_samples: 1536
      threshold: 0.3
      min_silence_duration_ms: 400
      min_speech_duration_ms: 1000
      pad_duration_ms: 20
  
    - reciter_name: عبد الباسط عبد الصمد
      id: 2
      url: https://everyayah.com/data/Abdul_Basit_Murattal_192kbps/000_versebyverse.zip
      window_size_samples: 1536
      threshold: 0.3
      min_silence_duration_ms: 400
      min_speech_duration_ms: 700
      pad_duration_ms: 20
  
  
    - reciter_name: محمود علي البنا
      id: 3
      url: https://everyayah.com/data/mahmoud_ali_al_banna_32kbps/000_versebyverse.zip
      window_size_samples: 1536
      threshold: 0.3
      min_silence_duration_ms: 400
      min_speech_duration_ms: 700
      pad_duration_ms: 20
      
    - reciter_name: على الحذيفي
      id: 5
      url: https://everyayah.com/data/Hudhaify_128kbps/000_versebyverse.zip
      window_size_samples: 1536
      threshold: 0.3
      min_silence_duration_ms: 350
      min_speech_duration_ms: 700
      pad_duration_ms: 5
  
    - reciter_name: أيمن رشدي سويد
      id: 6
      url: https://everyayah.com/data/Ayman_Sowaid_64kbps/000_versebyverse.zip
      window_size_samples: 1536
      threshold: 0.3
      min_silence_duration_ms: 500
      min_speech_duration_ms: 1000
      pad_duration_ms: 10
  
    - reciter_name: محمد أيوب
      id: 7
      url: https://everyayah.com/data/Muhammad_Ayyoub_128kbps/000_versebyverse.zip
      window_size_samples: 1536
      threshold: 0.3
      min_silence_duration_ms: 400
      min_speech_duration_ms: 1000
      pad_duration_ms: 10
  
  
    - reciter_name: إبراهيم الأخضر
      id: 8
      url: https://everyayah.com/data/Ibrahim_Akhdar_32kbps/000_versebyverse.zip
      window_size_samples: 1536
      threshold: 0.3
      min_silence_duration_ms: 390
      min_speech_duration_ms: 700
      pad_duration_ms: 30
  ```

### تهيئة البيانات

1. تحميل المصاحف القرآنية وتحويلها لمصفوفات array بصيغة Hugging Face Audio Dataset بمعمدل (sample rate)   16000 HZ 
2. تقسيم الآيات تبعا للوقف باستخدام sliro-vad-v4
3. تطبيق تسريع وإبطاء لسرعة التلاوت على 40 % من التلاوات لمواكبة سرعات التلاوات المختلفة
4. تطيبق data augmentations باتسخدام مكتبة [audumentations](https://github.com/iver56/audiomentations) متبعين نفس طريقة sliro-vad وإعدادات الل augmentattions موجودة [هنا](./augment_config.yml)

5. ومن المعلوم أن w2v2Bert تدعم طول يصل إلى 100 ثانية. فقد وقع الاختيار على 20 ثانية.
6. وبعد ذلك تم تقسيم الآيات الأطول من 20 ثانية باستخدام خوارزمية النافذة المتحركة sliding window لاستخدام كل بيانات التدريب وهذه صورة توضيحية لاختيار الطول الأقصى: 
![durations-fig](./assets/durations_histogram.png)

إعدادات ال augmentations: 


```yml
# Audio processing parameters
min_size_samples: 32000       # Minimum audio length (2 seconds at 16kHz)
max_size_samples: 320000      # Maximum audio length (20 seconds at 16kHz)
truncate_window_overlap_length: 16000  # Overlap when splitting long audio

# Spectrogram feature extraction
window_length_samples: 400    # Window length for STFT
hop_length_samples: 160       # Hop length for STFT
sampling_rate: 16000          # Audio sample rate
stride: 2                     # Convolution stride for feature extraction

# Label configuration
speech_label: 1               # Label for speech segments
silence_label: 0              # Label for silence segments
ignored_idx: -100             # Index to ignore in loss calculations

# Model and processing
model_id: facebook/w2v-bert-2.0  # Pre-trained model identifier
batch-size: 32                # Batch size for processing
samples-per-shard: 1024       # Samples per Parquet shard

# Augmentation parameters
seed: 1                       # Random seed for reproducibility
min-stretch-ratio: 0.8        # Minimum time stretch ratio
max-stretch-ratio: 1.5        # Maximum time stretch ratio
augment-prob: 0.4             # Probability of applying augmentation

```

ال augmentations المستخدمة موجودة [هنا](./src/recitations_segmenter/train/augment.py):

```python
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

```
### التدريب

* تم التدريب بتوفيق الله سبحانه وتعالي على منصة [Lightning Studio](https://lightning.ai) باستخدام معالج رسوميات واحد من نوع Nvidia L40 (48GB) ولمدة ساعاتان تقريبا
* تم التدريب على نموذج: `Wav2Vec2BertForAudioFrameClassification` .تم استخدام [هذا الكود](./train.py)

قم بتنزيل متطلبات التدريب:

```bash
pip install -r train_requirements.txt
```

وبعد ذلك بتهيئة accelerate

```bash
accelereate config
```

ومن ثم ابتدأ التدريب:

```bash
accelerate launch train.py
```

### النتائج:

نتائج الاختبار على مصحف لم يتم التدريب عليه:

| Metric     | Value  |
|------------|--------|
| Accuracy   | 0.9958 |
| F1         | 0.9964 |
| Loss       | 0.0132 |
| Precision  | 0.9976 |
| Recall     | 0.9951 |

## ملاحظات مهمة

* تم تهيئة البيانات على الحاسوب الفائق لمكتبة الأسكندرية [Bibliotheca Alexandrina (BA), HPC](https://hpc.bibalex.org/about) باستخدام أداة [slurm](https://slurm.schedmd.com/overview.html)
* تم استخدام مكتبة [submitit](https://pypi.org/project/submitit/) لستهيل القيام بأكثر من علمية حسابية على الحاسب الفايق في نفس الوقت
* جميع أكواد الحاسب الفائف المستخدمة في تهيئة البيانات موجودة [هنا](./hpc_scripts/)
* المقسم يعتبر السكت وقفا ولا يعتبره سكتا وهذا يعتبر عيب.


## TODO

* [x] Test the model on notebook.
* [x] Add CI/CD checking python versions.
* [x] Add commdnad line tool to API.
* [x] Add pytest for the cli.
* [x] Whether to raise execption or not if no speech found
* [x] Add caching mechanism.
* [x] Project Description
* [x] API docs
* [ ] train docs
* [ ] datasets docs (create and description)
* [x] Add lock file for reproudcing training
* [ ] Steps to reprouduce Dev environment [see](https://chat.qwen.ai/s/75280423-a193-4f1b-a35b-93a5f8e03ff8?fev=0.0.87)
* [x] Add libsoundfile and ffmbeg as backend for reading mp3 files
* [ ] publish to pypip

* [ ] update colab docs
