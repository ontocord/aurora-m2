#!/bin/bash
#SBATCH --account EUHPC_E03_068
#SBATCH -p boost_usr_prod
#SBATCH --time 16:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4          # 4 gpus per node out of 4
#SBATCH --mem=123000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=create_multimodal_data
#SBATCH --output=/leonardo_work/EUHPC_E03_068/slurm_out/huu-%j-%t.out


export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
source ~/miniconda3/bin/activate

cache_dir="/leonardo_work/EUHPC_E03_068/.cache"

#/leonardo_scratch/fast/EUHPC_E03_068/.cache"

directory="/leonardo_work/EUHPC_E03_068/staging_data/redteam"
mkdir -p $directory

echo 'creating images and captions'
srun python -m src.create_multimodal_data \
     --task generate_images_then_captions \
     --input_dir $directory \
     --output_dir $directory \
     --input_path $directory/llavaguard_captions_rating.jsonl \
     --output_path $directory/llavaguard_images_then_recaptions.jsonl

