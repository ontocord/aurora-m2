#!/bin/bash
#SBATCH --account EUHPC_E03_068
#SBATCH -p boost_usr_prod
#SBATCH --time 1:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4          # 4 gpus per node out of 4
#SBATCH --mem=123000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=create_multimodal_data
#SBATCH --output=/leonardo_work/EUHPC_E03_068/slurm_out/harsh-%j-%t.out


export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
source ~/miniconda3/bin/activate

# cache_dir="/leonardo_work/EUHPC_E03_068/.cache"
cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache"

#/leonardo_scratch/fast/EUHPC_E03_068/.cache"

directory="/leonardo_work/EUHPC_E03_068/staging_data/harsh-autoredteam-test"
mkdir -p $directory

# echo 'generating autoredteam data'
# srun python -m src.create_multimodal_data-harsh \
#      --batch_size 2 \
#      --task generate_autoredteam \
#      --LLM_large_model teknium/OpenHermes-2.5-Mistral-7B \
#      --use_LLM_size large \
#      --cache_dir $cache_dir \
#      --output_dir $directory \
#      --output_path $directory/autoredteam.jsonl  \

# echo 'generating generate_captions_from_autoredteam data'
# srun python -m src.create_multimodal_data-harsh \
#      --batch_size 2 \
#      --task generate_captions_from_autoredteam \
#      --LLM_large_model teknium/OpenHermes-2.5-Mistral-7B \
#      --use_LLM_size large \
#      --cache_dir $cache_dir \
#      --output_dir $directory \
#      --input_path $directory/autoredteam.jsonl \
#      --output_path $directory/autoredteam_caption.jsonl \

# echo 'generating generate_images_then_recaption_from_autoredteam data'
# srun python -m src.create_multimodal_data-harsh \
#      --batch_size 2 \
#      --task generate_images_then_recaption_from_autoredteam \
#      --LLM_large_model teknium/OpenHermes-2.5-Mistral-7B \
#      --use_LLM_size large \
#      --cache_dir $cache_dir \
#      --output_dir $directory \
#      --input_path $directory/autoredteam_caption.jsonl \
#      --output_path $directory/autoredteam_image_with_recaption.jsonl \

echo 'generating generate_revised_instruction_then_response_from_autoredteam data'
srun python -m src.create_multimodal_data-harsh \
     --batch_size 2 \
     --task generate_revised_instruction_then_response_from_autoredteam \
     --LLM_large_model teknium/OpenHermes-2.5-Mistral-7B \
     --use_LLM_size large \
     --cache_dir $cache_dir \
     --output_dir $directory \
     --input_path $directory/autoredteam_image_with_recaption.jsonl \
     --output_path $directory/autoredteam_revised_instr_response.jsonl \