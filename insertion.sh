#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/fvsalpha.log
#SBATCH --time=24:00:00  # Change the time limit to 72 hours
#SBATCH --job-name=fvsalpha

python fvsalpha.py