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

directory="data/autoredteam-2"
mkdir -p $directory

output_file="$directory/purpleteam-${purpleteam_model_path//\//-}-${target_model_path//\//-}.jsonl"

echo 'running autoredteam'
srun python -m src.purpleteam.autoredteam \
   --llamaguard_path llamas-community/LlamaGuard-7b \
   --purpleteam_model_path $purpleteam_model_path \
   --target_model_path $target_model_path \
   --output_path $output_file \
   --cache_dir $cache_dir \
   --verb_types_to_include EU_Act_and_transparency_violations_against_people

# echo 'running create_caption_from_instr'
# srun python -m src.purpleteam.create_caption_from_instr \
#     --cache_dir $cache_dir \
#     --input_path $output_file \
#     --output_path $directory/step-1.jsonl


# echo 'running create_img_and_caption'
# srun python -m src.purpleteam.create_img_and_caption \
#     --cache_dir $cache_dir \
#     --score_cutoff 0.14 \
#     --input_path $directory/step-1.jsonl \
#     --output_path $directory/step-2.jsonl

# echo 'running create_revised_instr_response'
# srun python -m src.purpleteam.create_revised_instr_response \
#     --cache_dir $cache_dir \
#     --input_path $directory/step-2.jsonl \
#     --output_path $directory/step-3.jsonl

#echo 'running create_multiturn_conv'
#srun python -m src.purpleteam.create_multiturn_conv \
#    --cache_dir $cache_dir \
#    --input_path $directory/step-3.jsonl \
#    --output_path $directory/processed.jsonl



# directory="data/commoncatalog"
# mkdir -p $directory

# echo 'running create_caption_from_img'
# srun python -m src.purpleteam.create_caption_from_img \
#     --cache_dir $cache_dir \
#     --score_cutoff 0.14 \
#     --input_dir data/test-commoncatalog-cc-by \
#     --output_path $directory/commoncatalog_recaption.jsonl \

# echo 'running create_revised_instr_response'
# srun python -m src.purpleteam.create_revised_instr_response \
#     --cache_dir $cache_dir \
#     --input_path $directory/commoncatalog_recaption.jsonl \
#     --output_path $directory/step-3.jsonl

# echo 'running create_multiturn_conv'
# srun python -m src.purpleteam.create_multiturn_conv \
#     --cache_dir $cache_dir \
#     --input_path $directory/step-3.jsonl \
#     --output_path $directory/processed.jsonl
