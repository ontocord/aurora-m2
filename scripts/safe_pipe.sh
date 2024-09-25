#!/bin/bash
#SBATCH --account EUHPC_E03_068
#SBATCH -p boost_usr_prod
#SBATCH --time 16:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4          # 4 gpus per node out of 4
#SBATCH --mem=123000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=purple_team
#SBATCH --output=slurm_out/purpleteam-%j-%t.out


export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

source ~/miniconda3/bin/activate

purpleteam_model_path="teknium/OpenHermes-2.5-Mistral-7B"
target_model_path="teknium/OpenHermes-2.5-Mistral-7B"
cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache"

output_file="/leonardo_work/EUHPC_E03_068/safellm/data/purpleteam-${purpleteam_model_path//\//-}-${target_model_path//\//-}.jsonl"

echo 'running autoredteam'
srun python -m src.purpleteam.autoredteam \
  --llamaguard_path llamas-community/LlamaGuard-7b \
  --purpleteam_model_path $purpleteam_model_path \
  --target_model_path $target_model_path \
  --output_path $output_file \
  --cache_dir $cache_dir \
#   --verb_types_to_include EU_Act_and_transparency_violations_against_people,adversarial_EU_Act_and_transparency_violations_against_people,exercise_of_rights_by_and_with_adults,adversarial_exercise_of_rights_by_and_with_adults,EU_Act_and_transparency_violations_using_EU_tools,adversarial_EU_Act_and_transparency_violations_using_EU_tools \
#   --obj_types_to_include EU_Act_high_risk_EU_tools

srun python -m src.purpleteam.create_captions_from_instr \
    --cache_dir $cache_dir \
    --input_path data/purpleteam-teknium-OpenHermes-2.5-Mistral-7B-teknium-OpenHermes-2.5-Mistral-7B.jsonl \
    --output_path data/multimodal/step-1.jsonl

srun python -m src.purpleteam.create_imgs_and_captions \
    --cache_dir $cache_dir \
    --input_path data/multimodal/step-1.jsonl \
    --output_path data/multimodal/step-2.jsonl

srun python -m src.purpleteam.create_instr_response_captions \
    --cache_dir $cache_dir \
    --input_path data/multimodal/step-2.jsonl \
    --output_path data/multimodal/processed.jsonl
