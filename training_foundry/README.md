# Training with foundry

Steps:
 - Clone and Install [foundry](https://github.com/mosaicml/llm-foundry) and [composer](https://github.com/mosaicml/composer).
 - Follow the README in `llm-foundry/scripts/train/README.md` for dataset preparation and training
 - In this folder you can find example script to run the training of Mistral 7b on 16 nodes
 - Copy the other files in this folder into `/<path_to_foundry>/llm-foundry/scripts/train/`
 - Fix the paths in all 3 files so that they point to each other inside `<path_to_foundry>` and the path for where to save ckpts in `full_mistral.yaml`
 - The command is `sbatch mistral_sender_16nodes.sh`