git clone git@github.com:obadx/recitations-segmenter.git
pip install -r train_requirements.txt
python -c "from datasets import load_dataset; ds = load_dataset('obadx/recitation-segmentation-augmented', num_proc=32)"
