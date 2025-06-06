#!/bin/bash
#SBATCH --job-name=normalization
#SBATCH --output=logs/normalization_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --partition=plgrid
#SBATCH --account=plglscclass24-cpu

mkdir -p logs/tsne

source ~/.bashrc
conda activate dimred_env

cd $SLURM_SUBMIT_DIR
cd common

python normalization.py
