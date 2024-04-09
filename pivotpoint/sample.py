from .pivot import *
from distribution import PivotPointsDistribution

class PivotClusterSampler:
    def __init__(self, cluster: list[PivotingIntervalPair]):
        pivot_avg_diameter = lambda pivot: 0.5 * (pivot.inner_diameter() + pivot.outer_diameter())
        self.pivots = sorted(cluster, by = pivot_avg_diameter)
        self.centers = [pivot.center() for pivot in cluster]
        self.data = sorted(unique(collect_pivot_data(cluster)))
        
def sort_pivot_cluster(cluster: list[PivotingIntervalPair]):
    average_diameter = lambda pivot: (1/2) * (pivot.outer_diameter() + pivot.inner_diameter())
    return sorted(cluster, key = average_diameter)

def successor_search_simple(pivots: list[PivotingIntervalPair]):
    n = len(arr)
    candidates = list[tuple[float,float]]()
    for i in range(n):
        a = pivots[i]
        for j in range(i + 1,n):
            b = pivots[j]
            if b.succeeds(a):
                # match.
                candidates.append((i,j))
                continue
            elif a.contains(b):
                # out of bounds, look elsewhere.
                break
            else:
                # otherwise, keep looking.
                continue
    return candidates

class ConnectedPivotCluster:
    def __init__(self, cluster: list[PivotingIntervalPair]):
        self.pivot_sequence = sort_pivot_cluster(cluster)
        self.edges = successor_search_simple(pivot_sequence)
        # self.pivot_point_error = cluster.error

def sequentialize(connected_cluster: ConnectedPivotCluster):
    forward_sequence = list[float]()
    reverse_sequence = list[float]()
    for pivot in connected_cluster.pivot_sequence:
        forward_sequence += list(pivot.left())
        reverse_sequence += list(pivot.right())
    return forward_sequence, reverse_sequence
    