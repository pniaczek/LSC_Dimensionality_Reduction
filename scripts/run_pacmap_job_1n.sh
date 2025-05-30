#!/bin/bash
#SBATCH --job-name=pacmap_viz
#SBATCH --output=results/pacmap_job_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --partition=plgrid
#SBATCH --account=plglscclass24-cpu

mkdir -p logs/tsne

source ~/.bashrc
conda activate dimred_env

cd $SLURM_SUBMIT_DIR

python pacmap_reduction.py
