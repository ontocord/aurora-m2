#!/bin/bash
#SBATCH --account EUHPC_E03_068
#SBATCH -p boost_usr_prod
#SBATCH --time 01:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=1 # 4 tasks out of 32
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1        # 4 gpus per node out of 4
#SBATCH --mem=123000          # memory per node out of 494000MB (481GB)
#SBATCH --job-name=atomic_stories
#SBATCH --output=atomic-stories-%j-%t.out

# source ~/miniconda3/bin/activate

PYTHONPATH='.' srun python ./src/scripts/gen_stories_slurm.py \
  --batch-size 32 \
  --stories-per-src-shard 1000 \
  --stories-per-target-shard 1000 \
  --model-name "/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/Llama-3-Instruct-8B-SPPO-Iter3" \
  --tokenizer-name "/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/Llama-3-Instruct-8B-SPPO-Iter3" \
  --src-file /leonardo_work/EUHPC_E03_068/synthetic_datasets/atomic_stories/atomic_stories.jsonl \
  --src-file-url https://huggingface.co/datasets/ontocord/atomic_2024/resolve/main/data/atomic_stories.jsonl \
  --prompts-template-name default \
  --huggingface-or-vllm vllm \
  --dst-file-path /leonardo_work/EUHPC_E03_068/synthetic_datasets/atomic_stories_processed_vllm
