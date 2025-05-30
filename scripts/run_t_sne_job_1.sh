#!/bin/bash
#SBATCH --job-name=tsne_viz
#SBATCH --output=logs/tsne_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --partition=plgrid
#SBATCH --account=plglscclass24-cpu

mkdir -p logs/tsne

source ~/.bashrc
conda activate dimred_env

cd $SLURM_SUBMIT_DIR


echo "[$(date)] Uruchamiam t-SNE..."
python t_sne_reduction.py
echo "[$(date)] Zakończono."