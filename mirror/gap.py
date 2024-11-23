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

    if type(gap_constraint) == GapRangeConstraint:
        gap_indices = sorted(gap_indices, key = lambda idx: spectrum[idx[1]] - spectrum[idx[0]])
    return gap_indices