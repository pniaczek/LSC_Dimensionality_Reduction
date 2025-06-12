#!/bin/bash
#SBATCH --job-name=pca_multinode_Dask
#SBATCH --output=logs/pca_multinode/pca_multinode%j.out
#SBATCH --error=logs/pca_multinode/pca_multinode%j.err
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --partition=plgrid
#SBATCH --account=plglscclass24-cpu

mkdir -p logs/pca_multinode

source ~/.bashrc
source activate dimred_env

cd $SLURM_SUBMIT_DIR
cd scripts

python pca_cpu_multinode.py