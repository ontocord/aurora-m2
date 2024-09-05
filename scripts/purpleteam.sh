#!/bin/bash
#SBATCH --account EUHPC_E03_068
#SBATCH -p boost_usr_prod
#SBATCH --time 6:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gpus-per-node=2
#SBATCH --gres=gpu:2          # 4 gpus per node out of 4
#SBATCH --mem=123000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=purple_team
#SBATCH --output=slurm_out/purpleteam-%j-%t.out


source ~/miniconda3/bin/activate

purpleteam_model_path="teknium/OpenHermes-2.5-Mistral-7B"
target_model_path="teknium/OpenHermes-2.5-Mistral-7B"

output_file="/leonardo_work/EUHPC_E03_068/safellm/data/purpleteam-${purpleteam_model_path//\//-}-${target_model_path//\//-}.jsonl"

srun python -m src.purpleteam.autoredteam \
  --llamaguard_path meta-llama/Llama-Guard-3-8B \
  --purpleteam_model_path $purpleteam_model_path \
  --target_model_path $target_model_path \
  --output_path $output_file
