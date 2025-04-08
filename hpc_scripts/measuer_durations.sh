#!/bin/sh

#!/bin/bash

#SBATCH --job-name=DStates
#SBATCH --account=shams035
#SBATCH --partition=cpu
#SBATCH --output=generate-ds.out
#SBATCH --time=1-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16


echo 'Saving data'
bash /cluster/users/shams035u1/.bashrc
source /cluster/users/shams035u1/data/miniconda3/bin/activate
conda activate segment12
cd ..

BASE_PATH=/cluster/users/shams035u1/data/segmentation-datasets

python measure_durations_stats.py
