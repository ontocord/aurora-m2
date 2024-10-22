#!/bin/bash

export LOCAL_WORLD_SIZE=4
export RANK=$(($SLURM_PROCID * $LOCAL_WORLD_SIZE))
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$(($SLURM_NTASKS * $LOCAL_WORLD_SIZE))
export NODE_RANK=$SLURM_NODEID

echo "${NODE_RANK} => MASTER_ADDR=${MASTER_ADDR}"
echo "${NODE_RANK} => MASTER_PORT=${MASTER_PORT}"
echo "${NODE_RANK} => RANK=${RANK}"
echo "${NODE_RANK} => LOCAL_RANK=${LOCAL_RANK}"
echo "${NODE_RANK} => LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"
echo "${NODE_RANK} => WORLD_SIZE=${WORLD_SIZE}"
echo "${NODE_RANK} => NODE_RANK=${NODE_RANK}"

train_folder="/<path_to_foundry>/llm-foundry/scripts/train/"

composer ${train_folder}/train.py ${train_folder}/full_mistral.yaml