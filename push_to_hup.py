import os
import argparse

from huggingface_hub import login as hf_login
from dotenv import load_dotenv
from transformers import AutoModel


def load_secrets():
    # Load environment variables from .env
    load_dotenv()

    # Retrieve tokens
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    # Log into HuggingFace Hub
    if hf_token:
        hf_login(token=hf_token)
    else:
        print("HuggingFace token not found!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True, type=str)
    parser.add_argument(
        '--repo-name', required=False, type=str,
        default='obadx/recitation-segmenter-v2_0')

    args = parser.parse_args()

    load_secrets()

    model = AutoModel.from_pretrained(args.model_dir)
    model.push_to_hub(args.repo_name)
