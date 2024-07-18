#!/bin/bash

# Loop from 0 to 2499 (2500 iterations)
for (( i=0; i<2500; i++ ))
do
    # Update the variable SLURM_PROCID with the current iteration number
    export SLURM_PROCID=$i
    
    # Execute the Python script with the specified arguments
    PYTHONPATH="." python ./src/scripts/slurm_download_sharded.py --in-folder ../dataset/ --out-folder ../dataset_processed/
    
    # Sleep for a random number of seconds between 1 and 6
    sleep $(( RANDOM % 6 + 1 ))
done