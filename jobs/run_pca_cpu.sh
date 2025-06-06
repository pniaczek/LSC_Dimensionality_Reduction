#!/bin/bash
#SBATCH --job-name=pca_viz
#SBATCH --output=logs/pca/pca_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --partition=plgrid-gpu-a100 
#SBATCH --account=plglscclass24-gpu-a100

mkdir -p logs/pca

source ~/.bashrc
source activate dimred_env

cd $SLURM_SUBMIT_DIR
cd scripts

python pca_cpu.py