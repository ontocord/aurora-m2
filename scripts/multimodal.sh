#!/bin/bash
#SBATCH --account EUHPC_E03_068
#SBATCH -p boost_usr_prod
#SBATCH --time 08:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4          # 4 gpus per node out of 4
#SBATCH --mem=123000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=purple_team
#SBATCH --output=slurm_out/multimodal-%j-%t.out

export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# # Accept parameters from the command line
# input_file_path=$1
# output_file_path=$2
# score_cutoff=$3

# # Echo the arguments for debugging
# echo "Input file: $input_file_path"
# echo "Output file: $output_file_path"
# echo "Score cutoff: $score_cutoff"

# shift 
# shift
# shift 

source ~/miniconda3/bin/activate

cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache"
input_dir="/leonardo_work/EUHPC_E03_068/safellm/data/test-commoncatalog-cc-by/0/least_dim_range=256-512/aspect_ratio_bucket=1-1"

srun python -m src.purpleteam.create_caption_from_image \
    --cache_dir $cache_dir \
    --input_dir $input_dir \
    --output_path data/multimodal/commoncatalog-out-test.jsonl

# srun python -m src.purpleteam.create_imgs_and_captions \
#     --cache_dir $cache_dir \
#     --input_path $input_file_path \
#     --output_path $output_file_path \
#     --score_cutoff $score_cutoff

srun python -m src.purpleteam.create_captions_from_instr \
    --cache_dir $cache_dir \
    --input_path data/purpleteam-teknium-OpenHermes-2.5-Mistral-7B-teknium-OpenHermes-2.5-Mistral-7B.jsonl \
    --output_path data/multimodal/step-1-test.jsonl
# srun python -m src.purpleteam.create_imgs_and_captions \
#     --cache_dir $cache_dir \
#     --input_path data/multimodal/step-1-test.jsonl \
#     --output_path data/multimodal/step-2-test.jsonl
# srun python -m src.purpleteam.create_instr_response_captions \
#     --cache_dir $cache_dir \
#     --input_path data/multimodal/step-2-test.jsonl \
#     --output_path data/multimodal/processed-test.jsonl
