#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --job-name=test-gpu
#SBATCH --output=logs/gpu_check_%j.out
#SBATCH --ntasks=1
#SBATCH --partition=plgrid-gpu-a100 
#SBATCH --account=plglscclass24-gpu-a100

module load cuda
nvidia-smi
