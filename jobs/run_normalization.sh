#!/bin/bash
#SBATCH --job-name=normalization
#SBATCH --output=logs/normalization/normalization_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --partition=plgrid-gpu-a100 
#SBATCH --account=plglscclass24-gpu-a100

mkdir -p logs/normalization

source ~/.bashrc
source activate dimred_env

cd $SLURM_SUBMIT_DIR
cd common

python normalization.py
