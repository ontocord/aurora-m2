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
export HF_HUB_OFFLINE=1

directory=$1 
shift 

source ~/miniconda3/bin/activate

purpleteam_model_path="teknium/OpenHermes-2.5-Mistral-7B"
target_model_path="teknium/OpenHermes-2.5-Mistral-7B"
cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache"

mkdir -p $directory
log_file="$directory/log.txt"

# Clear the log file if it exists
echo "" > $log_file

## step 1
echo 'running create_caption_from_img'
start_time=$(date +%s)
srun python -m src.purpleteam.create_caption_from_img \
    --cache_dir $cache_dir \
    --score_cutoff 0.14 \
    --input_dir data/test-commoncatalog-cc-by \
    --output_path $directory/commoncatalog_recaption.jsonl

# Check if the command was successful
if [ $? -eq 0 ]; then
    end_time=$(date +%s)
    elapsed=$(( end_time - start_time ))
    echo "'create_caption_from_img' completed successfully at $(date) - Duration: ${elapsed}s" >> $log_file
else
    echo "'create_caption_from_img' failed at $(date)" >> $log_file
    exit 1
fi

## step 2
echo 'running create_img_and_caption'
start_time=$(date +%s)
srun python -m src.purpleteam.create_img_and_caption-2 \
     --cache_dir $cache_dir \
     --score_cutoff 0.14 \
     --input_path  $directory/commoncatalog_recaption.jsonl \
     --output_dir $directory \
     --output_path $directory/commoncatalog_image_and_recaption.jsonl

# Check if the command was successful
if [ $? -eq 0 ]; then
    end_time=$(date +%s)
    elapsed=$(( end_time - start_time ))
    echo "'create_img_and_caption' completed successfully at $(date) - Duration: ${elapsed}s" >> $log_file
else
    echo "'create_img_and_caption' failed at $(date)" >> $log_file
    exit 1
fi

## step 3
echo 'running create_instr_response_from_caption'
start_time=$(date +%s)
srun python -m src.purpleteam.create_instr_response_from_caption \
    --cache_dir $cache_dir \
    --input_path $directory/commoncatalog_image_and_recaption.jsonl \
    --output_path $directory/processed.jsonl

# Check if the command was successful
if [ $? -eq 0 ]; then
    end_time=$(date +%s)
    elapsed=$(( end_time - start_time ))
    echo "'create_instr_response_from_caption' completed successfully at $(date) - Duration: ${elapsed}s" >> $log_file
    
    # Count number of lines in processed.jsonl
    jsonl_length=$(wc -l < "$directory/processed.jsonl")
    echo "processed.jsonl has $jsonl_length lines" >> $log_file
    
    # Mark process as completed
    echo "completed" >> $log_file
else
    echo "'create_instr_response_from_caption' failed at $(date)" >> $log_file
    exit 1
fi