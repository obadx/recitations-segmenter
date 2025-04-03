#!/bin/sh

#SBATCH --job-name=ds-hf
#SBATCH --account=shams035
#SBATCH --partition=cpu
#SBATCH --output=ds-hf.out
#SBATCH --time=4-01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16


bash /cluster/users/shams035u1/.bashrc
source /cluster/users/shams035u1/data/miniconda3/bin/activate
conda activate segment

cd /cluster/users/shams035u1/data/segmentation-datasets/recitation-segmentation-augmented
git add --all
git commit -m 'add: first version of the ds'
git push

