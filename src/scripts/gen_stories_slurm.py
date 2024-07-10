import os
import warnings
from time import sleep
from typing import Optional

from pathlib import Path
from tqdm import tqdm
import click

# Importing functions from other modules as per your original script
from src.create_shard_prorotype import create_shard
from src.pipeline import get_llm
from src.slurm_utils import get_rank, get_world_size
from src.prompts import PROMPT_REGISTRY
from src.utils import download_dataset, get_splits, get_target_path, check_done, mark_done


@click.command()
@click.option('--batch-size', type=int, help='Batch size for processing.')
@click.option('--stories-per-src-shard', type=int, help='Number of stories per source shard.')
@click.option('--stories-per-target-shard', type=int, help='Number of stories per target shard.')
@click.option('--model-name', type=str, help='Name of the language model.')
@click.option('--tokenizer-name', type=str, help='Name of the tokenizer.')
@click.option('--src-file', type=str, help='Path to the source file.')
@click.option('--src-file-url', type=str, default=None, help='URL to download the source file if not present locally.')
@click.option('--prompts-template-name', type=str, help='Name of the prompts template.')
@click.option('--dst-file-path', type=str, help='Destination path to save processed shards.')
@click.option('--huggingface-or-vllm', type=str, default='vllm')
def main(
        batch_size: int,
        stories_per_src_shard: int,
        stories_per_target_shard: int,
        model_name: str,
        tokenizer_name: str,
        src_file: str,
        src_file_url: Optional[str],
        prompts_template_name: str,
        dst_file_path: str,
        huggingface_or_vllm: str
):
    rank, world_size = get_rank(), get_world_size()
    print(f"[rank {rank}]\t process started with world size {world_size}")
    if rank == 0:
        print(f'[rank {rank}]\t creating target dir')
        Path(dst_file_path).mkdir(exist_ok=True, parents=True)
    else:
        print(f'[rank {rank}]\t waiting for root process to create target dir if necessary')
        # This is just to avoid weird behavior in the rare case that root is behind other nodes
        sleep(1)
    if stories_per_src_shard > stories_per_target_shard:
        warnings.warn(f"Not all samples per shard are processed due to stories_per_src_shard ({stories_per_target_shard}) > stories_per_target_shard ({stories_per_target_shard})")

    # get the llm pipeline
    print(f'[rank {rank}]\t instantiating pipeline')
    llm = get_llm(model_name=model_name, tokenizer_name=tokenizer_name, batch_size=batch_size, huggingface_or_vllm=huggingface_or_vllm)
    print(f'[rank {rank}]\t pipeline assembled')
    # download the dataset
    download_dataset(path=src_file, url=src_file_url, rank=rank)
    print(f'[rank {rank}]\t dataset setup complete')


    # split the dataset into manageable chunks (if we are root) if not allready done so
    shards = get_splits(path=src_file, rank=rank, world_size=world_size, samples_per_shard=stories_per_src_shard)
    print(f'[rank {rank}]\t got assigned {len(shards)} shards: {shards}')

    # get the prompts
    prompts = PROMPT_REGISTRY[prompts_template_name]
    for shard_path in tqdm(shards, desc=f"Processing shard shards rank: {rank}"):
        # generate a target path for our shard
        target_path = get_target_path(shard_path, Path(dst_file_path))

        # if the marker file exists, we know the shard has been processed and we skip it
        if check_done(target_path):
            print(f'[rank {rank}]\t shard {shard_path} is done already, skipping...')
            continue


        # this is where the (llm) magic happens
        create_shard(
            llm=llm, stories_per_shard=stories_per_target_shard,
            src_file=str(shard_path), shard_path=str(target_path),
            prompts=prompts, batch_size=batch_size
        )

        # create a market file to show that this shard was processed correctly and completely
        mark_done(target_path)
    print(f'[rank {rank}]\t all shards processed.')


if __name__ == '__main__':
    main()
