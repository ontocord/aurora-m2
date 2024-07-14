#!/bin/bash
#SBATCH --account EUHPC_E03_068
#SBATCH -p dcgp
#SBATCH --time 01:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=8 # 4 tasks out of 32
#SBATCH --gpus-per-node=1
#SBATCH --mem=123000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=flickr
#SBATCH --output=flickr-%j-%t.out

# source ~/miniconda3/bin/activate

PYTHONPATH='.' srun python ./src/scripts/slurm_download_sharded.py --in-folder $1 --out-folder $2
