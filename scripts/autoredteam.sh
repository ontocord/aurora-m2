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
#SBATCH --output=slurm_out/autoredteam-%j-%t.out


export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

directory=$1 
shift 

source ~/miniconda3/bin/activate

# purpleteam_model_path="teknium/OpenHermes-2.5-Mistral-7B"
# target_model_path="teknium/OpenHermes-2.5-Mistral-7B"
purpleteam_model_path="Qwen/Qwen2.5-3B-Instruct"
target_model_path="Qwen/Qwen2.5-3B-Instruct"
cache_dir="/leonardo_scratch/fast/EUHPC_E03_068/.cache"

mkdir -p $directory
log_file="$directory/log.txt"

# Clear the log file if it exists
echo "" > $log_file

## step 1
echo 'running autoredteam'
start_time=$(date +%s)
srun python -m src.purpleteam.autoredteam \
  --llamaguard_path llamas-community/LlamaGuard-7b \
  --purpleteam_model_path $purpleteam_model_path \
  --target_model_path $target_model_path \
  --output_path $directory/autoredteam.jsonl \
  --cache_dir $cache_dir \
  --batch_size 128 \
  --verb_types_to_include EU_Act_and_transparency_violations_using_EU_tools \
  --obj_types_to_include EU_Act_high_risk_EU_tools

# Check if the command was successful
if [ $? -eq 0 ]; then
    end_time=$(date +%s)
    elapsed=$(( end_time - start_time ))
    echo "'autoredteam' completed successfully at $(date) - Duration: ${elapsed}s" >> $log_file
else
    echo "'autoredteam' failed at $(date)" >> $log_file
    exit 1
fi


## step 2
echo 'running create_caption_from_instr'
start_time=$(date +%s)
srun python -m src.purpleteam.create_caption_from_instr \
     --cache_dir $cache_dir \
     --input_path $directory/autoredteam.jsonl \
     --output_path $directory/autoredteam_caption.jsonl \
     --batch_size 64

# Check if the command was successful
if [ $? -eq 0 ]; then
    end_time=$(date +%s)
    elapsed=$(( end_time - start_time ))
    echo "'create_caption_from_instr' completed successfully at $(date) - Duration: ${elapsed}s" >> $log_file
else
    echo "'create_caption_from_instr' failed at $(date)" >> $log_file
    exit 1
fi


## step 3
echo 'running create_img'
start_time=$(date +%s)
srun python -m src.purpleteam.create_img \
     --cache_dir $cache_dir \
     --input_path  $directory/autoredteam_caption.jsonl \
     --output_dir $directory \
     --output_path $directory/autoredteam_image.jsonl \
     --batch_size 8

# Check if the command was successful
if [ $? -eq 0 ]; then
    end_time=$(date +%s)
    elapsed=$(( end_time - start_time ))
    echo "'create_img' completed successfully at $(date) - Duration: ${elapsed}s" >> $log_file
else
    echo "'create_img' failed at $(date)" >> $log_file
    exit 1
fi


# step 4
echo 'running create_caption_and_recaption'
start_time=$(date +%s)
srun python -m src.purpleteam.create_caption_and_recaption \
     --cache_dir $cache_dir \
     --score_cutoff 0.14 \
     --input_path $directory/autoredteam_image.jsonl \
     --output_path $directory/autoredteam_img_and_recaption.jsonl \
     --output_dir $directory \
     --batch_size 64

# Check if the command was successful
if [ $? -eq 0 ]; then
    end_time=$(date +%s)
    elapsed=$(( end_time - start_time ))
    echo "'create_caption_and_recaption' completed successfully at $(date) - Duration: ${elapsed}s" >> $log_file
else
    echo "'create_caption_and_recaption' failed at $(date)" >> $log_file
    exit 1
fi


## step 4
echo 'running create_revised_instr_response'
start_time=$(date +%s)
srun python -m src.purpleteam.create_revised_instr_response \
     --cache_dir $cache_dir \
     --input_path $directory/autoredteam_img_and_recaption.jsonl \
     --output_path $directory/processed.jsonl \
     --batch_size 64

# Check if the command was successful
if [ $? -eq 0 ]; then
    end_time=$(date +%s)
    elapsed=$(( end_time - start_time ))
    echo "'create_revised_instr_response' completed successfully at $(date) - Duration: ${elapsed}s" >> $log_file
    
    # Count number of lines in processed.jsonl
    jsonl_length=$(wc -l < "$directory/processed.jsonl")
    echo "processed.jsonl has $jsonl_length lines" >> $log_file
    
    # Mark process as completed
    echo "completed" >> $log_file
else
    echo "'create_revised_instr_response' failed at $(date)" >> $log_file
    exit 1
fi

# # echo 'running create_multiturn_conv'
# # srun python -m src.purpleteam.create_multiturn_conv \
# #      --cache_dir $cache_dir \
# #      --input_path $directory/step-3.jsonl \
# #      --output_path $directory/processed.jsonl