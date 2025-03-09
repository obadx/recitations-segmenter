#!/bin/sh

#!/bin/bash

#SBATCH --job-name=QVADcpu
#SBATCH --account=shams035
#SBATCH --partition=cpu
#SBATCH --output=generate-ds.out
#SBATCH --time=1-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16


echo 'Extracting Speech intervals #########'
bash /cluster/users/shams035u1/.bashrc
source /cluster/users/shams035u1/data/miniconda3/bin/activate
conda activate segment
cd ..

BASE_PATH=/cluster/users/shams035u1/data/segmentation-datasets

python extract_speech_intervals_normal.py \
    --dataset-path "$BASE_PATH/segment-ds-before-process" \
    --samples-per-shard 1024 \
    --out-path "$BASE_PATH/recitation-segmentation"

