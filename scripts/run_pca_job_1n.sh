#!/bin/bash
#SBATCH --job-name=pca_viz
#SBATCH --output=logs/pca_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --partition=plgrid
#SBATCH --account=plglscclass24-cpu

mkdir -p logs/pca

source ~/.bashrc
conda activate dimred_env

cd $SLURM_SUBMIT_DIR


echo "[$(date)] Uruchamiam PCA..."
python pca_reduction.py
echo "[$(date)] Zakończono."