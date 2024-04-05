from pivotpoint import *
import random
from pprint import pprint

def make_dist(path: str, bins = 100):
    pivots = read_csv_to_pivots(path)
    dist = PivotPointsDistribution(pivots,bins)
    return dist

path = "./test/input/pivots_test.txt"
dist = make_dist(path)

dump_object_data(dist)
assert all([dist.clusters[i].count(x) == 0
    for i in range(dist.n_clusters())
    for j in range(i + 1, dist.n_clusters())
    for x in dist.clusters[j]])