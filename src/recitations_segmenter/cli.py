import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification

from .segment import (
    NoSpeechIntervals,
    TooHighMinSpeechDuration,
    read_audio,
    segment_recitations,
)

SUPPORTED_EXTENSIONS = {".mp3", ".wav",
                        ".flac", ".ogg", ".aac", ".m4a", ".opus"}


def prepare_args():
    parser = argparse.ArgumentParser(
        description="Segment Holy Quran rectiations into speech intervals based on وقف using Wav2Vec2Bert model.",
        formatter_class=argparse.RawTextHelpFormatter,  # Preserves formatting
        epilog="""\
Examples:
  # Process single file with default settings
  recitations-segmenter input.mp3 -o results


  # Process multiple files file with default settings
  recitations-segmenter input1.mp3 input2.wav -o output


  # Process directory of audio files
  recitations-segmenter /path/to/recitations/ --output ./segmentation_results


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
"""
    )

    # Input/Output Group
    io_group = parser.add_argument_group("Input/Output Options")
    io_group.add_argument(
        "inputs",
        nargs="+",
        help="Input paths (files or directories) containing audio files"
    )
    io_group.add_argument(
        "-o", "--output",
        default="./output",
        help="Output directory for JSON results (default: %(default)s)"
    )

    # Processing Parameters Group
    processing_group = parser.add_argument_group("Segmentation Parameters")
    processing_group.add_argument(
        "--min-silence-duration-ms",
        type=int,
        default=30,
        dest="min_silence_duration_ms",
        help="""\
Minimum silence duration (ms) between speech segments
- Silence shorter than this will be merged with speech
- Default: %(default)dms"""
    )
    processing_group.add_argument(
        "--min-speech-duration-ms",
        type=int,
        default=30,
        dest="min_speech_duration_ms",
        help="""\
Minimum valid speech duration (ms)
- Speech segments shorter than this will be removed
- Default: %(default)dms"""
    )
    processing_group.add_argument(
        "--pad-duration-ms",
        type=int,
        default=30,
        dest="pad_duration_ms",
        help="Padding added around speech segments (ms)\nDefault: %(default)dms"
    )
    processing_group.add_argument(
        "--return-samples",
        action="store_true",
        help="""\
Return intervals in samples according to 16000 sampling rate.
- Default to return interval in seconds""",
    )

    # Model Parameters Group
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="""\
Number of audio chunks processed simultaneously
- Higher values may increase speed but require more GPU memory
- Default: %(default)d"""
    )
    model_group.add_argument(
        "--max-duration-ms",
        type=int,
        default=19995,
        help="""\
Maximum processing chunk duration (2-20000ms)
- Affects memory usage and processing granularity
- Do not Change it unless there exists a strong reason.
- Default: %(default)dms"""
    )

    # Runtime Options Group
    model_group.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Processing device selection\nDefault: %(default)s"
    )
    model_group.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="""\
Numerical precision for model computation:
- bfloat16: Best performance (modern GPUs)
- float16: Legacy support
- float32: Maximum precision (CPU fallback)
Default: %(default)s"""
    )

    args = parser.parse_args()

    return args


def main():
    args = prepare_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Map dtype string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Initialize processor and model
    processor = AutoFeatureExtractor.from_pretrained(
        "obadx/recitation-segmenter-v2")
    model = AutoModelForAudioFrameClassification.from_pretrained(
        "obadx/recitation-segmenter-v2",
        torch_dtype=dtype,
    )

    # attching model to device
    model.to(args.device, dtype=dtype)

    # Collect input files
    input_files = []
    for input_path in args.inputs:
        path = Path(input_path)
        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                input_files.append(path)
            else:
                print(f"Skipping unsupported file: {path}")
        elif path.is_dir():
            for file_path in path.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    input_files.append(file_path)
                else:
                    print(f"Skipping unsupported file: {file_path}")

        else:
            print(f"Warning: {path} does not exist. Skipping.")

    # Process each file
    waves = []
    valid_pathes = []
    for file_path in input_files:
        try:
            # Read audio file
            wave = read_audio(str(file_path))
            waves.append(wave)
            valid_pathes.append(file_path)
        except Exception as e:
            print(
                f'Error reading this media file: {file_path.absolute()}: {e}')

    # Segment the audio
    outputs = segment_recitations(
        waves,
        model,
        processor,
        batch_size=args.batch_size,
        min_silence_duration_ms=args.min_silence_duration_ms,
        min_speech_duration_ms=args.min_speech_duration_ms,
        pad_duration_ms=args.pad_duration_ms,
        max_duration_ms=args.max_duration_ms,
        return_seconds=not args.return_samples,
        device=torch.device(args.device),
        dtype=dtype,
    )

    # Get the result (assuming one output per file)
    for result, file_path in zip(outputs, valid_pathes):

        # Prepare JSON data
        json_data = {
            "clean_speech_intervals": result.clean_speech_intervals.tolist(),
            "speech_intervals": result.speech_intervals.tolist(),
            "is_complete": result.is_complete,
        }

        # Generate output path
        output_filename = f"{file_path.stem}_speech_intervals.json"
        output_path = Path(args.output) / output_filename

        # Write JSON file
        with open(output_path, "w+") as f:
            json.dump(json_data, f, indent=4)

    if len(outputs) == 1:
        print('Speech Intervals:')
        print(outputs[0].clean_speech_intervals)


if __name__ == "__main__":
    main()
