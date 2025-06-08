#!/bin/bash
#SBATCH --job-name=tsne_open
#SBATCH --output=logs/tsne/tsne_opentsne_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --partition=plgrid-gpu-a100 
#SBATCH --account=plglscclass24-gpu-a100

mkdir -p logs/tsne

source ~/.bashrc
source activate dimred_env

cd $SLURM_SUBMIT_DIR
cd scripts

echo "[$(date)] Start openTSNE"
python tsne_cpu.py
echo "[$(date)] Done"
