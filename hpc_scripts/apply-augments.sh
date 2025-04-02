#!/bin/bash

bash /cluster/users/shams035u1/.bashrc
source /cluster/users/shams035u1/data/miniconda3/bin/activate
conda activate segment12
cd ..

BASE_PATH=/cluster/users/shams035u1/data/segmentation-datasets

python apply_augments.py \
    --dataset-path "$BASE_PATH/recitation-segmentation" \
    --out-path "$BASE_PATH/recitation-segmentation-augmented" \
    --batch-size 32 \
    --samples-per-shard 1024 \
    --seed 1 \
    --min-stretch-ratio 0.8 \
    --max-stretch-ratio 1.5 \
    --augment-prob 0.4
