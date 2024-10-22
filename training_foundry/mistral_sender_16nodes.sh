#!/bin/bash

#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=4
#SBATCH --time=0-24:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --wait-all-nodes=1
#SBATCH --qos=normal
#SBATCH --account=IscrC_TRAVEL
#SBATCH --job-name=llm_mistral
#SBATCH --output=slurm_logs/llm_mistral_sender-%j.out

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export MASTER_PORT="12345"
export PYTHONUNBUFFERED="1"

eval "$(/path/to/conda shell.bash hook)" # init conda
conda activate /path/to/conda_env # activate conda env

srun /<path_to_foundry>/llm-foundry/scripts/train/mistral_runner.sh