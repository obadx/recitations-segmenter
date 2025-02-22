#!/bin/sh

#SBATCH --job-name=ExtractSpeechIntervals
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --account=g.shams035u1
#SBATCH --time=1-01:00:00


echo 'Extracting Speech intervals #########'
source /cluster/users/shams035u1/data/miniconda3/bin/activate
conda activate segment
cd ..

BASE_PATH=/cluster/users/shams035u1/data/segmentation-datasets

python extract_speech_intervals.py \
    --dataset-path "$BASE_PATH/segment-ds-before-process" \
    --device cuda \
    --batch-size 1024 \
    --samples-per-shard 1024 \
    --out-path "$BASE_PATH/recitation-segmentation"

