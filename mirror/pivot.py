import numpy as np
from statistics import mode

from .util import mass_error, count_mirror_symmetries, residue_lookup, reflect, INTERGAP_TOLERANCE
from .scan import ScanConstraint, constrained_pair_scan

class Pivot:
    def __init__(self, pair_a, pair_b, indices_a, indices_b):
        self.index_data = sorted([*indices_a, *indices_b])
        self.data = sorted([*pair_a, *pair_b])

        self._indices_a = indices_a
        self._indices_b = indices_b
        self._pair_a = pair_a
        self._pair_b = pair_b
    
    def peaks(self):
        return self.data

    def indices(self):
        return self.indices

    def peak_pairs(self):
        return self._pair_a, self._pair_b
    
    def index_pairs(self):
        return self._indices_a, self._indices_b
    
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
        gap_a = data[1] - data[0]
        gap_b = data[3] - data[2]
        return (gap_a + gap_b) / 2

    def __repr__(self):
        peaks_a, peaks_b = self.peak_pairs()
        ind_a, ind_b = self.index_pairs()
        return f"Pivot{peaks_a, peaks_b, ind_a, ind_b}"

class VirtualPivot(Pivot):

    def __init__(self, data, index_data):
        self.index_data = index_data
        self.data = data

    def data_pairs(self):
        raise NotImplementedError("Virtual pivots are not composed of pairs.")
    
    def index_pairs(self):
        raise NotImplementedError("Virtual pivots are not composed of pairs.")

    def __repr__(self):
        return f"VirtualPivot{self.peaks(), self.indices()}"
        

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

def locate_pivot_point(
    spectrum,
    gap_indices,
    tolerance = INTERGAP_TOLERANCE,
):
    pivots_overlap = find_pivots(spectrum, gap_indices, PivotOverlapConstraint(tolerance))
    if len(pivots_overlap) > 0:
        return 'o', pivots_overlap
    else:
        pivots_disjoint = find_pivots(spectrum, gap_indices = PivotDisjointConstraint(tolerance))
        target = mode(pivot.center() for pivot in pivots_disjoint)
        return 'd', [pivot for pivot in pivots_disjoint if pivot.center() == target]

def score_pivot(
    spectrum,
    pivot
):
    return (
        count_mirror_symmetries(spectrum, pivot.center()),
        mass_error(pivot.gap()),
        residue_lookup(pivot.gap()))