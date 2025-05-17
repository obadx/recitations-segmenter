#!/bin/sh

#SBATCH --job-name=nvidia-smi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=g.shams035
#SBATCH --time=00:10:00


echo 'Set up #########'
bash /cluster/users/shams035u1/.bashrc
source /cluster/users/shams035u1/data/miniconda3/bin/activate
conda activate segment12

python tests/all_modules_wihout_cache.py
