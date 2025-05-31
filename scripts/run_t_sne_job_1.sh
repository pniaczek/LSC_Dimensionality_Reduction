#!/bin/bash
#SBATCH --job-name=tsne_open
#SBATCH --output=logs/tsne_opentsne_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --partition=plgrid
#SBATCH --account=plglscclass24-cpu

mkdir -p logs

source ~/.bashrc
conda activate dimred_env_2

cd $SLURM_SUBMIT_DIR

echo "[$(date)] Start openTSNE"
python t_sne_reduction.py
echo "[$(date)] Done"
