import click
import pyarrow.parquet as pq
import os
from pathlib import Path
from tqdm import tqdm
import wget


def shard_parquet_file(input_file, output_directory, num_shards):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Read the parquet file
    table = pq.read_table(input_file)

    # Calculate number of rows and total size
    num_rows = len(table)

    # Determine number of rows per shard (except the last one)
    rows_per_shard = num_rows // num_shards

    # Create shards
    for i in tqdm(range(num_shards), 'processing shards'):
        # Determine rows for this shard
        if i < num_shards - 1:
            rows_to_take = rows_per_shard
        else:
            rows_to_take = num_rows - (rows_per_shard * (num_shards - 1))

        # Write shard to file
        start_row = i * rows_per_shard
        end_row = start_row + rows_to_take

        shard_filename = os.path.join(output_directory, f"{i}_{Path(input_file).name}")
        pq.write_table(table.slice(start_row, end_row), shard_filename)

        print(f"Shard {i}: rows {start_row} to {end_row} written to {shard_filename}")



# input_file = "dataset.parquet"


@click.command()
@click.option('--input-file', type=str, help='Input file.')
@click.option('--num-shards', type=int, help='num shards.')
@click.option('--url', type=str, default='https://huggingface.co/datasets/SilentAntagonist/test/resolve/main/dataset.parquet?download=true', help='Input download.')
def main(input_file: str, num_shards: int, url: str):
    # Example usage:
    if not Path(input_file).exists():
        print("did not detect downloaded file, downloading from provided url", url)
        wget.download(url, input_file)
    else:
        print('no download needed')
    output_directory = Path(input_file).with_suffix("")
    shard_parquet_file(input_file, output_directory, num_shards)

if __name__ == '__main__':
    main()