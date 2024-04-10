from pivotpoint import *
import random
from glob import glob
from pprint import pprint
from statistics import mean,variance,quantiles
import numpy as np

def make_dist(path: str, bins = 100):
    pivots = read_csv_to_pivots(path)
    dist = PivotPointsDistribution(pivots,bins)
    return dist

input_paths = glob("./test/input/test_pivots_*")

for input_path in input_paths:
    file_name_suffix = input_path.split('/')[-1]
    dist = make_dist(input_path)
    dump_object_data(dist)
    print("\nclusters stats:")
    print(" \tsize\t average\t\t variance\t\t IQR\n")
    pivot_clusters = [[dist.pivot_points[pivot_id] for pivot_id in cluster] for cluster in dist.clusters]
    pivot_clusters.sort(key = lambda x: -len(x))
    n_clusters = dist.n_clusters()
    cluster_pair_sums = np.zeros((n_clusters,n_clusters))
    for i in range(n_clusters):
        pivot_points = pivot_clusters[i]
        #pivot_points = [dist.pivot_points[pivot_id] for pivot_id in cluster]
        if len(pivot_points) > 1:
            qnt = quantiles(pivot_points)
            iqr = qnt[2] - qnt[0]
            print(i, '\t', len(pivot_points), '\t', mean(pivot_points), '\t', variance(pivot_points), '\t', iqr)
        else:
            print(i, '\t', 1, '\t', pivot_points[0], '\t', 0, '\t', 0)
        for j in range(n_clusters):
            cluster_pair_sums[i,j] = 0.5 * (mean(pivot_clusters[i]) + mean(pivot_clusters[j]))
    np.set_printoptions(precision=3,suppress=True)
    primary_cluster = cluster_pair_sums[0,0]
    print("\ncluster pair averages")
    for i in range(n_clusters):
        for j in range(i,n_clusters):
            if abs(cluster_pair_sums[i,j] - primary_cluster) < 5:
                print((i,j), ":\t", cluster_pair_sums[i,j])
    input()