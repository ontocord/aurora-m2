import os

def get_rank() -> int:
    rank = int(os.environ['SLURM_PROCID'])
    return rank


def _get_tasks_per_node() -> int:
    return int(os.environ['SLURM_NTASKS_PER_NODE'])


def _get_num_nodes() -> int:
    return int(os.environ['SLURM_JOB_NUM_NODES'])


def get_world_size() -> int:
    return _get_num_nodes() * _get_tasks_per_node()