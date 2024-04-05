from pivotpoint import *
import random
from glob import glob

def load_pivot_inputs(path: str):
    return read_csv_to_pivots(path)

def reduce_to_mz(pivots: list[PivotingIntervalPair]):
    return unique([mz for pivot in pivots for mz in pivot.data()])

def recreate_pivots(data: list[float], gapset: list[float], precision = 0.1):
    return find_pivoting_interval_pairs(data,gapset,precision)

def save_pivot_outputs(path: str, pivots: list[PivotingIntervalPair]):
    write_pivots_to_csv(path, pivots)

def compare_pivot_sets(path_A: str, path_B: str):
    pivots_A = load_pivot_inputs(path_A)
    pivots_B = load_pivot_inputs(path_B)
    return len(pivots_B) / len(pivots_A), subset([a.data() for a in pivots_A], [b.data() for b in pivots_B])

input_paths = glob("./test/input/test_pivots_*")
output_paths = [f"{x[0:-4]}_recreated.txt".replace("input","output") for x in input_paths]
gapset = AMINO_ACID_MASS_AVERAGE
for (input_path,output_path) in zip(input_paths,output_paths):
    pivots = load_pivot_inputs(input_path)
    raw_data = reduce_to_mz(pivots)
    new_pivots = recreate_pivots(raw_data, gapset)
    save_pivot_outputs(output_path,new_pivots)
    print(input_path)
    print(output_path)
    print(compare_pivot_sets(input_path,output_path))