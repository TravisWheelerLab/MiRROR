from .scan import ScanConstraint, constrained_pair_scan
import itertools

#=============================================================================#

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

def find_all_gaps(
    spectrum,
    gap_constraints,
):
    return list(itertools.chain.from_iterable(find_gaps(spectrum, constraint) for constraint in gap_constraints))

def _fast_find_gaps(
    arr,
    target,
    tolerance
):
    n = len(arr)
    for i in range(n):
        x = arr[i]
        for j in range(i + 1, n):
            y = arr[j]
            dif = y - x
            if dif > target + tolerance:
                break
            elif dif >= target - tolerance:
                yield (i, j)

def fast_find_gaps(
    arr,
    target,
    tolerance
):
    return list(_fast_find_gaps(arr, target, tolerance))

# about 2x faster. but still weirdly slow?
def fast_find_all_gaps(
    arr,
    targets,
    tolerance
):
    return list(itertools.chain.from_iterable(_fast_find_gaps(arr, target, tolerance) for target in targets))