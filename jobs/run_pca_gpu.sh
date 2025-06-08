#!/bin/bash
#SBATCH --job-name=pca_gpu
#SBATCH --output=logs/pca_gpu/pca_gpu_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plglscclass24-gpu-a100
#SBATCH --gpus=1

mkdir -p logs/pca_gpu

source ~/.bashrc
source activate dimred_gpu

cd $SLURM_SUBMIT_DIR
cd scripts

python pca_gpu.py
