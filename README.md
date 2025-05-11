# recitations-segmenter

[![Tests](https://github.com/obadx/recitations-segmenter/actions/workflows/tests.yml/badge.svg)](https://github.com/obadx/recitations-segmenter/actions/workflows/tests.yml)
[![Python 3.10](https://github.com/obadx/recitations-segmenter/actions/workflows/tests.yml/badge.svg?query=job%3A%22Python+3.10%22)](https://github.com/obadx/recitations-segmenter/actions/workflows/tests.yml?query=job%3A%22Python+3.10%22)
[![Python 3.11](https://github.com/obadx/recitations-segmenter/actions/workflows/tests.yml/badge.svg?query=job%3A%22Python+3.11%22)](https://github.com/obadx/recitations-segmenter/actions/workflows/tests.yml?query=job%3A%22Python+3.11%22)
[![Python 3.12](https://github.com/obadx/recitations-segmenter/actions/workflows/tests.yml/badge.svg?query=job%3A%22Python+3.12%22)](https://github.com/obadx/recitations-segmenter/actions/workflows/tests.yml?query=job%3A%22Python+3.12%22)
[![Python 3.13](https://github.com/obadx/recitations-segmenter/actions/workflows/tests.yml/badge.svg?query=job%3A%22Python+3.13%22)](https://github.com/obadx/recitations-segmenter/actions/workflows/tests.yml?query=job%3A%22Python+3.13%22)

<!-- [![Tests](https://github.com/obadx/recitations-segmenter/actions/workflows/tests.yml/badge.svg)](https://github.com/obadx/recitations-segmenter/actions/workflows/tests.yml) -->

A Machine Learning Model that split a recitations on pauses (وقف) not by Ayah.

## TODO

* [x] Test the model on notebook.
* [x] Add CI/CD checking python versions.
* [x] Add commdnad line tool to API.
* [x] Add pytest for the cli.
* [x] Whether to raise execption or not if no speech found
* [ ] Add caching mechanism.
* [ ] Project Description
* [ ] API docs
* [ ] train docs
* [ ] datasets docs (create and description)
* [ ] Add lock file for reproudcing training
* [ ] Steps to reprouduce Dev environment [see](https://chat.qwen.ai/s/75280423-a193-4f1b-a35b-93a5f8e03ff8?fev=0.0.87)
* [ ] Add libsoundfile and ffmbeg as backend for reading mp3 files
* [ ] publish to pypip

## Installtion

> Note: for data builindg we worked on python 3.13 and for augmentations we worked to python 3.12 due to audiomentatiosn depends on scipy

* Install `ffmbeg` using conda

```bash
conda install -c conda-forge ffmpeg
```

* Install `scipy` using anaconda

```bash
conda install -c conda-forge scipy=1.15.2
```

## Chosing hypberparameters

We have chosed the `max_recitaton_seconds` to be `20` secons and the rests of the samples will be split uising sliding window algorigh with overlap of `1` second

![durations-fig](./assets/durations_histogram.png)
