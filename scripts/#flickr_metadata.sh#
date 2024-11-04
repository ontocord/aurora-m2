#!/bin/bash
#SBATCH --account EUHPC_E03_068
#SBATCH -p dcgp_usr_prod
#SBATCH --time 01:00:00     # format: HH:MM:SS
#SBATCH -N 2                # 1 node
#SBATCH --ntasks-per-node=8 # 4 tasks out of 32
#SBATCH --cpus-per-task=16
#SBATCH --mem=63000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=flickr
#SBATCH --output=flickr-%j-%t.out

# source ~/miniconda3/bin/activate

PYTHONPATH='.' srun python ./src/scripts/slurm_download_sharded.py --in-folder ../flickr/dataset/ --out-folder  ../flickr/dataset_processed/
