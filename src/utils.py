from time import sleep
from typing import Dict, List
import os
import wget
from pathlib import Path
from tqdm import tqdm

try:
    from vllm import RequestOutput
except:
    pass


def get_target_path(shard_path: Path, dst_file_path: Path) -> Path:
    filename = shard_path.name
    return dst_file_path / filename


def mark_done(filepath: Path):
    with open(filepath.with_suffix('.done'), "w") as fp:
        pass
    return


def check_done(filepath: Path):
    return filepath.with_suffix('.done').exists()


def get_sublist(elements, rank, world_size):
    # Calculate the number of elements in each sublist
    base_size = len(elements) // world_size
    remainder = len(elements) % world_size
    overhang_index = (base_size*world_size)+rank


    # Calculate the start and end indices for the desired rank
    start = (rank * base_size)
    end = (rank+1)*base_size

    # Return the sublist
    result =  elements[start:end]
    if remainder != 0 and rank < remainder:
        result.append(elements[overhang_index])
    return result


def sort_files_by_number(directory: Path):
    # Get all files in the directory
    files = os.listdir(directory)

    # Extract numbers and file names
    numbered_files = []
    for filename in files:
        # Split the filename by the first '_' to separate number from the rest
        parts = filename.split('_', 1)

        if len(parts) > 1:
            try:
                num = int(parts[0])
                numbered_files.append((num, filename))
            except ValueError:
                continue

    # Sort files based on the extracted numbers
    numbered_files.sort(key=lambda x: x[0])

    # Extract sorted file names
    sorted_files = [directory / filename for _, filename in numbered_files]

    return sorted_files


def split_file(input_filename, output_template, shard_size):
    with open(input_filename, 'r') as infile:
        shard_number = 1
        lines_in_shard = 0
        output_filename = output_template.format(shard_number)
        outfile = open(output_filename, 'w')
        for line in tqdm(infile, "processing lines in source file"):
            outfile.write(line)
            lines_in_shard += 1
            if lines_in_shard == shard_size:
                # Move to the next shard
                shard_number += 1
                output_filename = output_template.format(shard_number)
                outfile.close()
                outfile = open(output_filename, 'w')
                lines_in_shard = 0
        outfile.close()
    print(f"File '{input_filename}' has been split into {shard_number} shards.")


def filter_splits(sorted_shards: List[Path], rank: int, world_size: int) -> List[Path]:
    return get_sublist(sorted_shards, rank, world_size)


def get_splits(path, rank: int, world_size: int, samples_per_shard: int):
    path = Path(path)
    donefile = path.with_suffix(".split.done")
    if not donefile.exists():
        if rank == 0:
            path.with_suffix('').mkdir(parents=True, exist_ok=True)
            output_file_template = str(path.with_suffix("") / ("{}_" + path.name))
            split_file(path, output_file_template, samples_per_shard)
            with donefile.open("w"):
                pass
        else:
            sleep(60)
            return get_splits(path, rank, samples_per_shard)

    sorted_shards = sort_files_by_number(path.with_suffix(""))
    return filter_splits(sorted_shards, rank, world_size)


def download_dataset(path, url, rank):
    if not os.path.exists(path):
        if rank == 0:
            print("downloading dataset")
            Path(path).parent.mkdir(exist_ok=True, parents=True)
            wget.download(url)
        else:
            print("no file detecting, downloading with root")
            sleep(600)



def postprocess_hf_results(result: List[Dict[str, List[Dict[str, str]]]], result_key: str = "generated_text", txt_key: str = "content") -> List[str]:
    return [r[0][result_key][1][txt_key] for r in result]

def postprocess_vllm_results(results):
    return [i.outputs[0].text for i in results]

def postprocess_results(results):
    if isinstance(results[0], RequestOutput):
        return postprocess_vllm_results(results)
    else:
        return postprocess_hf_results(results)

if __name__ == '__main__':
    split = get_splits('./scripts/atomic_stories.jsonl', 0, 2, 10000)
    print(split)
    print(len(split))



# Example usage:
#elements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#world_size = 3

#for rank in range(world_size):
#    result = get_sublist(elements, rank, world_size)
#    print(f"Sublist for rank {rank}: {result}")


