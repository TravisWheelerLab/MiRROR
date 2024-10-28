from .pivot import Pivot
from .scan import ScanConstraint, constrained_pair_scan

class GapTargetConstraint(ScanConstraint):
    # match to a single gap

    def __init__(self, gap, tolerance):
        self.target_gap = gap
        self.tolerance = tolerance

    def evaluate(self, state):
        return state[1] - state[0]
    
    def stop(self, gap):
        return gap > self.target_gap + self.tolerance

    def match(self, gap):
        return self.target_gap - self.tolerance <= gap <= self.target_gap + self.tolerance 

class GapRangeConstraint(ScanConstraint):
    # match to a range of gap values

    def __init__(self, gaps, tolerance):
        self.min_gap = min(gaps)
        self.max_gap = max(gaps)
        self.tolerance = tolerance

    def evaluate(self, state):
        return state[1] - state[0]
    
    def stop(self, gap):
        return gap > self.max_gap + self.tolerance
    
    def match(self, gap):
        # under the assumption that `stop` has already returned false if 
        # `match` is being called, only the first half of the inequality needs 
        # to be checked. the second half is unnecessary, but is still included 
        # for the sake of legibility.
        return self.min_gap - self.tolerance <= gap <= self.max_gap + self.tolerance

def find_gaps(
    spectrum,
    gap_constraint
):
    gap_indices = constrained_pair_scan(
        spectrum,
        gap_constraint
    )
    gap_indices = gap_indices
    gap_indices = sorted(gap_indices, key = lambda idx: spectrum[idx[1]] - spectrum[idx[0]])
    gap_mz = [(spectrum[i],spectrum[j]) for (i,j) in gap_indices]
    return gap_indices, gap_mz

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
    gap_indices,
    gap_mz,
    pivot_constraint
):
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
