import numpy as np

from .util import reflect
from .scan import ScanConstraint, constrained_pair_scan

class Pivot:
    def __init__(self, pair_a, pair_b, indices_a, indices_b):
        self.index_data = sorted([*indices_a, *indices_b])
        self.data = sorted([*pair_a, *pair_b])

        self.indices_a = indices_a
        self.pair_a = pair_a
        self.gap_a = pair_a[1] - pair_a[0]
        
        self.indices_b = indices_b
        self.pair_b = pair_b
        self.gap_b = pair_b[1] - pair_b[0]
    
    def outer_left(self):
        return self.index_data[0]
    
    def inner_left(self):
        return self.index_data[1]
    
    def inner_right(self):
        return self.index_data[2]
    
    def outer_right(self):
        return self.index_data[3]

    def center(self):
        return sum(self.data) / 4
    
    def gap(self):
        return (self.gap_a + self.gap_b) / 2

    def __repr__(self):
        return f"gap:\t{self.gap()}\npeaks:\t{[round(x,3) for x in self.data]}\nindices:\t{self.index_data}"

#=============================================================================#
# finding pivots in the gaps of a spectrum

class AbstractPivotConstraint(ScanConstraint):
    
    def __init__(self, tolerance):
        self.tolerance = tolerance
    
    def evaluate(self, state):
        pair_a, pair_b = state
        gap_a = pair_a[1] - pair_a[0]
        gap_b = pair_b[1] - pair_b[0]
        gap_dif_ba = gap_b - gap_a
        return (gap_dif_ba, pair_a, pair_b)
    
    def stop(self, val):
        gap_dif_ba, _, __ = val
        return gap_dif_ba > self.tolerance

    def is_ordered(self, pair_a, pair_b):
        raise NotImplementedError("use PivotOverlapConstraint, or PivotDisjointConstraint, or roll your own subclass.")

    def match(self, val):
        gap_dif_ba, pair_a, pair_b = val
        if abs(gap_dif_ba) <= self.tolerance:
            return self.is_ordered(pair_a, pair_b) or self.is_ordered(pair_b, pair_a)
        else:
            return False

class PivotOverlapConstraint(AbstractPivotConstraint):

    def is_ordered(self, pair_a, pair_b):
        return pair_a[0] < pair_b[0] < pair_a[1] < pair_b[1]

class PivotDisjointConstraint(AbstractPivotConstraint):

    def is_ordered(self, pair_a, pair_b):
        return pair_a[0] < pair_a[1] < pair_b[0] < pair_b[1]

def find_pivots(
    spectrum,
    gap_indices,
    pivot_constraint
):
    gap_mz = [(spectrum[i], spectrum[j]) for (i, j) in gap_indices]
    pivot_indices = constrained_pair_scan(
        gap_mz,
        pivot_constraint
    )
    pivots = []
    for (i, j) in pivot_indices:
        p = gap_mz[i]
        q = gap_mz[j]
        indices_p = gap_indices[i]
        indices_q = gap_indices[j]
        if pivot_constraint.is_ordered(p,q):
            pivots.append(Pivot(p, q, indices_p, indices_q))
        elif pivot_constraint.is_ordered(q,p):
            pivots.append(Pivot(q, p, indices_q, indices_p))
        else:
            raise ValueError(f"malformed pivot! {p, q}")
    return pivots

#=============================================================================#
# use a pivot to analyze mirror symmetries

def _mirror_symmetric_gaps(
    gaps_mz: list[np.array],
    center: float,
    threshold: float,
):
    "return the subset of gaps that is closed under mirror symmetry w.r.t. the pivot center."
    reflector = lambda x: reflect(x, center)
    mirrored_gaps_mz = [np.array([reflector(peak_2), reflector(peak_1)]) for (peak_1, peak_2) in gaps_mz]
    n = len(gaps_mz)
    for i in range(n):
        x = gaps_mz[i]
        for j in range(i + 1, n):
            z = mirrored_gaps_mz[j]
            if abs(sum(x - z)) < threshold:
                yield i
                break

def pivot_symmetric_gaps(
    spectrum,
    gap_indices: list[tuple[int,int]],
    pivot: Pivot,
    threshold = 0.01
):
    gaps_mz = [np.array([spectrum[i], spectrum[j]]) for (i, j) in gap_indices]
    pivot_center = pivot.center()
    return [gap_indices[i] for i in _mirror_symmetric_gaps(gaps_mz, pivot_center, threshold)]