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

directory="/leonardo_work/EUHPC_E03_068/staging_data/huu-math-cot-python-test"
mkdir -p $directory

#     --input_path /leonardo_work/EUHPC_E03_068/staging_data/commoncatalog_test/part-00000-tid-5902856173521346022-881c09c0-8d26-4696-ab27-ae38e76eef3e-4766908-1-c000.parquet \

echo 'creating python javascript roundtrip'
srun python -m src.create_multimodal_data \
     --batch_size 100 \
     --LLM_code_model Qwen/Qwen2.5-Coder-1.5B-Instruct \
     --task generate_python_plus \
     --input_dir /leonardo_work/EUHPC_E03_068/staging_data/fineweb_test \
     --input_path /leonardo_work/EUHPC_E03_068/staging_data/fineweb_test/CC-MAIN-2013-20_1000.jsonl \
     --output_dir $directory \
     --output_path $directory/python_plus.jsonl     

#echo 'creating images and captions'
#srun python -m src.create_multimodal_data \
#     --task generate_images_then_captions \
#     --input_dir $directory \
#     --output_dir $directory \
#     --input_path $directory/stories.jsonl \
#     --output_path $directory/stories_with_image_and_recaption.jsonl

