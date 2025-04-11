#!/bin/bash

bash /cluster/users/shams035u1/.bashrc
source /cluster/users/shams035u1/data/miniconda3/bin/activate
conda activate segment12
cd ..

# Issue in the loading libraries of scipy (forcing to load conda g++ libraries
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

BASE_PATH=/cluster/users/shams035u1/data/segmentation-datasets

python apply_augments.py \
    --dataset-path "$BASE_PATH/recitation-segmentation" \
    --out-path "$BASE_PATH/recitation-segmentation-augmented" \
