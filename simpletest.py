# investigating the cluster pairs (A,B) whose centroid sum is twice that of the primary 
# cluster P. after P has been identified, these pairs are defined by the relation
# center(A) + center(B) = 2 * center(P).
from pivotpoint import *
from statistics import mean,variance,quantiles
import numpy as np
import networkx as nx
import random
import sys

def main(sequence):
    np.set_printoptions(precision=3,suppress=True)
    mz_values = generate_spectrum_from_sequence(sequence,parametize_yb_spectrum())
    
    gapset = AMINO_ACID_MASS_AVERAGE
    pivots = find_pivoting_interval_pairs(mz_values,gapset,0.1)
    n = len(pivots)
    print("pivots:")
    for i in range(n):
        p = pivots[i]
        print(i,'\t',p.data())

    bins = 100
    pivot_dist = PivotPointsDistribution(pivots,bins)
    print("clusters:\n\t",pivot_dist.clusters)
    
    n = pivot_dist.n_clusters()
    print("adjacency:")
    for i in range(n):
        cluster = pivot_dist.get_pivot_cluster(i)
        sampler = PivotClusterSampler(cluster)
        print(i,'\n',sampler.succession_matrix())

if __name__ == "__main__":
    main(sys.argv[1])
