#!/bin/bash
#SBATCH --job-name=GenerateDS
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --output=generate-ds.out
#SBATCH --ntasks=16
#SBATCH --account=g.shams035u1
#SBATCH --time=1-01:00:00


echo 'Starting Generate dataset task #########'
source /cluster/users/shams035u1/data/miniconda3/bin/activate
conda activate segment
cd ..

BASE_PATH=/cluster/users/shams035u1/data/segmentation-datasets

python generate_datasets.py \
    --download-dir "$BASE_PATH/downloads" \
    --out-path "$BASE_PATH/segment-ds-before-process" \
    --samples-per-shard 1024

