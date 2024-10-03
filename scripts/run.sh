#!/bin/bash

# bash run.sh autoredteam data/harsh-autoredteam-4-oct

split=$1
directory=$2

# Check if split is "autoredteam"
if [ "$split" == "autoredteam" ]; then
    # Upload to Hugging Face and run the autoredteam SLURM job
    python -m src.upload_hf --repo_id ontocord/aurora-m2-dataset --folder_path "$directory" --split "$split"
    sbatch scripts/autoredteam.sh "$directory"
else
    # Default to running the commoncatalog job
    python -m src.upload_hf --repo_id ontocord/aurora-m2-dataset --folder_path "$directory" --split "$split"
    sbatch scripts/commoncatalog.sh "$directory"
fi