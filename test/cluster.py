from pivotpoint import *
import random
from glob import glob
from pprint import pprint
from statistics import mean,variance,quantiles

def make_dist(path: str, bins = 100):
    pivots = read_csv_to_pivots(path)
    dist = PivotPointsDistribution(pivots,bins)
    return dist

input_paths = glob("./test/input/test_pivots_*")

for input_path in input_paths:
    dist = make_dist(input_path)
    dump_object_data(dist)
    for cluster in dist.clusters:
        pivot_points = [dist.pivot_points[pivot_id] for pivot_id in cluster]
        if len(pivot_points) > 1:
            print(len(pivot_points), '\t', mean(pivot_points), '\t', variance(pivot_points),end='\t')
            print(quantiles(pivot_points))
        else:
            print(1, '\t', pivot_points[0], '\t', 0)
    input()