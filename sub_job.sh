#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00


# Load conda env
export PATH=$PATH:$HOME/jq2/anaconda3/envs/dps/bin

# Run python script
./run.sh


nvidia-smi
hostname