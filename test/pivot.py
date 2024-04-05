from pivotpoint.io import *
from pivotpoint import *
import random

def load_pivot_inputs(path: str):
    return read_csv_to_pivots(path)

def reduce_to_mz(pivots: list[PivotingIntervalPair]):
    return unique([mz for pivot in pivots for mz in pivot.data()])

def recreate_pivots(data: list[float], gapset: list[float], precision = 0.01):
    return find_pivoting_interval_pairs(data,gapset,precision)

def save_pivot_outputs(path: str, pivots: list[PivotingIntervalPair]):
    write_pivots_to_csv(path, pivots)

def compare_pivot_sets(path_A: str, path_B: str):
    pivots_A = load_pivot_inputs(path_A)
    pivots_B = load_pivot_inputs(path_B)
    return len(pivots_B) / len(pivots_A),subset([a.data() for a in pivots_A],
            [b.data() for b in pivots_B])

input_path = "./test/input/pivots_test.txt"
output_path = "./test/output/pivots_test_recreated.txt"
gapset = [1/2, 1/3, 1/7]
pivots = load_pivot_inputs(input_path)
raw_data = reduce_to_mz(pivots)
new_pivots = recreate_pivots(raw_data, gapset)
save_pivot_outputs(output_path,new_pivots)
print(compare_pivot_sets(input_path,output_path))