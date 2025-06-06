#!/bin/bash
#SBATCH --job-name=ds_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=results/test/datashader_test_%j.out

source ~/.bashrc
conda activate dimred_env

cd $SLURM_SUBMIT_DIR
python datashader_try.py
