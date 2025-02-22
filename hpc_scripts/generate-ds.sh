#!/bin/bash

#SBATCH --job-name=GenerateSegmentDS
#SBATCH --account=shams035
#SBATCH --output=generate-ds.out
#SBATCH --time=1-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16




echo 'Starting Generate dataset task #########'
source /cluster/users/shams035u1/data/miniconda3/bin/activate
conda activate segment
cd ..

BASE_PATH=/cluster/users/shams035u1/data/segmentation-datasets

python generate_datasets.py \
    --download-dir "$BASE_PATH/downloads" \
    --out-path "$BASE_PATH/segment-ds-before-process" \
    --samples-per-shard 1024

