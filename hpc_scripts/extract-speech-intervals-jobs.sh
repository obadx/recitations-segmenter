#!/bin/bash

bash /cluster/users/shams035u1/.bashrc
source /cluster/users/shams035u1/data/miniconda3/bin/activate
conda activate segment
cd ..

BASE_PATH=/cluster/users/shams035u1/data/segmentation-datasets

python extract_speech_intervals_jobs.py \
    --dataset-path "$BASE_PATH/segment-ds-before-process" \
    --device cpu \
    --batch-size 1 \
    --samples-per-shard 1024 \
    --out-path "$BASE_PATH/recitation-segmentation"
