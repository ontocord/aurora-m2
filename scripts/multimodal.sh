#!/bin/bash
#SBATCH --account EUHPC_E03_068
#SBATCH -p boost_usr_prod
#SBATCH --time 8:00:00     # format: HH:MM:SS
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

# nvidia-smi 

source ~/miniconda3/bin/activate

srun python -m src.purpleteam.create_captions_from_instr --input_path data/purpleteam-teknium-OpenHermes-2.5-Mistral-7B-teknium-OpenHermes-2.5-Mistral-7B.jsonl
srun python -m src.purpleteam.create_imgs_and_captions
srun python -m src.purpleteam.create_instr_response_captions
