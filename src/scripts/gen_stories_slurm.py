from typing import Optional

from pipeline import get_llm
from slurm_utils import get_rank, get_world_size
from prompts import PROMPT_REGISTRY

def main(
        batch_size: int,
        stories_per_shard: int,
        shard_per_rank: int,
        model_name: str,
        tokenizer_name: str,
        src_file: str,
        src_file_url: Optional[str],
        prompts_template_name: str,
        dst_file_path: str
):
    rank, world_size = get_rank(), get_world_size()
    llm = get_llm(model_name=model_name, tokenizer_name=tokenizer_name)
    for shard in range(shard_per_rank):
        shard_path = get_shard_path(dst_file_path, shard, rank, world_size)
        remove_shard_if_incomplete_or_corrupted(shard_path)
        if check_shard_done():
            continue
        prompts = PROMPT_REGISTRY[prompts_template_name]
        create_shard(llm=llm, stories_per_shard=stories_per_shard, src_file=src_file, shard_path=shard_path, prompts=prompts)


if __name__ == '__main__':
    main()