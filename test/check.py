import sys
from copy import deepcopy
from pivotpoint import *

def subset(A,B):
    return all([B.count(a) > 0 for a in A])

def collect_data(pivotmap):
    return [pivot.data() for pivot in deepcopy(pivotmap)]

def unique(a):
    return list(set(a))

seq = "AAAAAA"
max_gap = 250
precision = 0.1
resolution = 100

spec = generate_spectrum_from_sequence(seq,parametize_yb_spectrum())
gapset = [71.037]
pivotpairs = find_pivoting_interval_pairs(spec,gapset,precision)
pivotpairdata = collect_data(pivotpairs)
clusters = find_pivot_clusters(pivotpairs,resolution)
cluster_data = [collect_data(clust) for clust in clusters]
print("spectrum",spec)
print("\t",[spec[i + 2] - spec[i] for i in range(len(spec) - 2)])
print("gapset",gapset)
print("data",pivotpairdata)
print("clusters",cluster_data)