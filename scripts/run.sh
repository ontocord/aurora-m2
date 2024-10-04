#!/bin/bash

# bash run.sh autoredteam data/harsh-autoredteam-4-oct

split=$1
directory=$2

# Check if split is "autoredteam"
if [ "$split" == "autoredteam" ]; then
    # run the autoredteam SLURM job
    sbatch scripts/autoredteam.sh "$directory"
else
    # Default to running the commoncatalog job
    sbatch scripts/commoncatalog.sh "$directory"
fi

# upload to Hugging Face
python -m src.upload_to_hf --repo_id ontocord/aurora-m2-dataset --folder_path "$directory" --split "$split"