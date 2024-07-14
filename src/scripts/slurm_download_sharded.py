import click
from tqdm import tqdm

from src.scripts.downloader import process_paquet_file
from src.slurm_utils import get_rank, get_world_size
from src.utils import sort_files_by_number, filter_splits
from pathlib import Path


def fetch_shards(path: Path, rank: int, world_size):
    sorted_shards = sort_files_by_number(path.with_suffix(""))
    return filter_splits(sorted_shards, rank, world_size)


@click.command()
@click.option('--in-folder', type=str, help='Input folder containing shards.')
@click.option('--out-folder', type=int, help='output folder.')
def main(in_folder: str, out_folder):
    path = Path(in_folder)
    rank = get_rank()
    world_size = get_world_size()
    print("rank", rank, "world size", world_size)
    shards = fetch_shards(path=path, rank=rank, world_size=world_size)
    print(shards)
    for i, shard in enumerate(tqdm(shards, 'processing shards')):
        if (Path(out_folder) / Path(shard).name).exists():
            print(f"shard {shard} processed allready, skipping")
            continue
        process_paquet_file(shard, out_folder=out_folder)


if __name__ == '__main__':
    main()
