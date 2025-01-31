#from .scan import ScanConstraint, constrained_pair_scan
from .util import collapse_second_order_list
from bisect import bisect_left, bisect_right
import numpy as np

#=============================================================================#

"""def _fast_find_gaps(
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

def find_gaps(
    arr,
    target,
    tolerance
):
    return list(_fast_find_gaps(arr, target, tolerance))

# about 2x faster. but still weirdly slow?
def find_all_gaps(
    arr,
    targets,
    tolerance
):
    return [find_gaps(arr, target, tolerance) for target in target]
"""

def find_all_gaps(
    spectrum: np.ndarray,
    target_groups: list[list[float]],
    tolerance: float,
    verbose = False,
):
    # create the target space
    all_targets = collapse_second_order_list(
        [[(target, group_idx, local_idx) for (local_idx, target) in enumerate(group)] for (group_idx, group) in enumerate(target_groups)])
    all_targets.sort(key = lambda x: x[0])
    n_targets = len(all_targets)
    target_values, target_group_idx, target_local_idx = zip(*all_targets)
    min_target = min(target_values) - tolerance
    max_target = max(target_values) + tolerance
    if verbose:
        print(f"target space:\n\t{target_values}\n\t{target_group_idx}\n\t{target_local_idx}\n\tmax: {max_target}\n\tmin: {min_target}")
    
    # locate gaps and index them to the target space
    gap_candidates = []
    def bound_bisection(v):
        return max(0, min(n_targets - 1, v))
    for (i, x) in enumerate(spectrum):
        for (j, y) in enumerate(spectrum[i + 1:]):
            dif = y - x
            if min_target <= dif <= max_target:
                l = bound_bisection(bisect_left(target_values, dif - tolerance))
                r = bound_bisection(bisect_right(target_values, dif + tolerance))
                candidate = (dif, (i, i + j + 1), (l, r))
                gap_candidates.append(candidate)
                if verbose > 1:
                    print(f"candidate: {candidate}")
            elif dif > max_target:
                break

    # binsort gaps and discard low quality matches
    n_groups = len(target_groups)
    gaps_by_group = [[] for _ in range(n_groups)]
    for (dif, gap, target_range) in gap_candidates:
        for target_match_idx in range(target_range[0], target_range[1] + 1):
            target_val = target_values[target_match_idx]
            if abs(dif - target_val) <= tolerance:
                target_grp_idx = target_group_idx[target_match_idx]
                target_lcl_idx = target_local_idx[target_match_idx]
                gaps_by_group[target_grp_idx].append((dif, gap, target_lcl_idx))
    
    return gaps_by_group