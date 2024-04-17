from .pivot import PivotingIntervalPair, pivot_sort_key, collect_pivot_data
from .util import unique,AMINO_ACID_CHAR_REPRESENTATION,AMINO_ACID_MASS_AVERAGE,AMINO_ACID_MASS_MONOISOTOPIC
from .distribution import PivotPointsDistribution,PivotPointsCluster
import numpy as np

def calculate_id_vector(gap: float, mode = "average"):
    if mode == "monoisotropic":
        alphabet = AMINO_ACID_MASS_MONOISOTOPIC
    else: # mode == "average":
        alphabet = AMINO_ACID_MASS_AVERAGE
    return [abs(a - gap) for a in alphabet]

class PivotClusterSampler:
    def __init__(self, cluster: PivotPointsCluster):
        self.pivots = cluster.pivots#sorted(cluster, key = pivot_avg_diameter)
        self.centers = [pivot.center() for pivot in self.pivots]
        self.data = sorted(unique(collect_pivot_data(self.pivots)))

    def succession_matrix(self):
        n = len(self.pivots)
        succeeds = np.zeros((n,n))
        for i in range(n):
            for j in range(i,n):
                if self.pivots[j].succeeds(self.pivots[i]):
                    succeeds[j,i] = 1
        return succeeds

    def succession_list(self):
        n = len(self.pivots)
        successions = list[tuple[int,int]]()
        for i in range(n):
            for j in range(i,n):
                if self.pivots[j].succeeds(self.pivots[i]):
                    successions.append((i,j))
        return successions

    def linearize(self):
        return sorted(unique(collect_pivot_data(self.pivots)))

    def bilinearize(self):
        pivots = sorted(self.pivots, key = pivot_sort_key)
        forward_read = list[float]()
        reverse_read = list[float]()
        for pivot in pivots:
            forward_read += list(pivot.left())
            reverse_read += list(pivot.right())
        forward_read = unique(sorted(forward_read))
        reverse_read = unique(sorted(forward_read))
        return forward_read, reverse_read

    def sequence(self):
        forward_read, reverse_read = self.bilinearize()
        if forward_read == reverse_read:
            n = len(forward_read)
            gap_sequence = [forward_read[i + 1] - forward_read[i] for i in range(n - 1)]
            identifier_vectors = list(map(calculate_identifier_vector,gap_sequence))
            return identifier_vectors
        else:
            return "forward-reverse discrepancies are not yet implemented"