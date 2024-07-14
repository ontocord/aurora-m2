import click
import pyarrow.parquet as pq
import pyarrow as pyr
import os
from tqdm import tqdm


def merge_parquet_files(input_directory, output_file):
    # Get list of all Parquet files in the input directory
    files = [file for file in os.listdir(input_directory) if file.endswith(".parquet")]

    # Initialize an empty list to hold all tables
    tables = []

    # Read each shard and append its table to the tables list
    for file in tqdm(files, "loading"):
        file_path = os.path.join(input_directory, file)
        table = pq.read_table(file_path)
        tables.append(table)

    # Concatenate all tables into a single table
    merged_table = pyr.concat_tables(tables)

    # Write the merged table to a single Parquet file
    pq.write_table(merged_table, output_file)

    print(f"Merged Parquet file written to {output_file}")


def check_parquet_files_equal(file1, file2):
    # Read Parquet files into pandas DataFrames
    df1 = pq.read_table(file1).to_pandas()
    df2 = pq.read_table(file2).to_pandas()
    print(len(df1), len(df2))

    # Sort DataFrames by all columns to ensure rows are ordered consistently
    df1_sorted = df1.sort_values(by=list(df1.columns)).reset_index(drop=True)
    df2_sorted = df2.sort_values(by=list(df2.columns)).reset_index(drop=True)

    # Check if sorted DataFrames are equal
    if df1_sorted.equals(df2_sorted):
        print(f"The Parquet files {file1} and {file2} have the same content.")
    else:
        print(f"The Parquet files {file1} and {file2} do not have the same content.")

# Example usage:
input_directory = "dataset"
output_file = "merged_output.parquet"
original_file = "dataset.parquet"


@click.command()
@click.option('--input-directory', type=str, help='Input file.')
@click.option('--output-file', type=int, help='num shards.')
def main(input_directory: str, output_file: str):
    merge_parquet_files(input_directory, output_file)