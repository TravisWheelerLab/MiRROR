from .pivot import *
from .util import *
from .distribution import PivotPointsDistribution

def calculate_identifier_vector(gap: float, mode = "average"):
    if mode == "monoisotropic":
        alphabet = AMINO_ACID_MASS_MONOISOTOPIC
    else: # mode == "average":
        alphabet = AMINO_ACID_MASS_AVERAGE
    return [round(1/abs(a - gap),4) for a in alphabet]

class PivotClusterSampler:
    def __init__(self, cluster: list[PivotingIntervalPair]):
        pivot_avg_diameter = lambda pivot: 0.5 * (pivot.inner_diameter() + pivot.outer_diameter())
        self.pivots = sorted(cluster, key = pivot_avg_diameter)
        self.centers = [pivot.center() for pivot in cluster]
        self.data = sorted(unique(collect_pivot_data(cluster)))

    def bilinearize(self):
        forward_read = list[float]()
        reverse_read = list[float]()
        for pivot in self.pivots:
            forward_read += list(pivot.left())
            reverse_read += list(pivot.right())
        forward_read = sorted(forward_read)
        reverse_read = sorted(forward_read)
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