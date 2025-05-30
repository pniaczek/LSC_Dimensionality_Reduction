#!/bin/bash
#SBATCH --job-name=umap_mnist_test
#SBATCH --output=results/test/umap_mnist_test_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --partition=plgrid
#SBATCH --account=plglscclass24-cpu

source ~/.bashrc
conda activate dimred_env

cd $SLURM_SUBMIT_DIR
python umap_try.py
