from bisect import bisect_left, bisect_right

import numpy as np

from .types import Gap, TargetGroup
from .util import collapse_second_order_list

#=============================================================================#

class GapResult:
    """Interface to the result of gap search. Implements length, values, indices, index_tuples, and local_ids."""

    def __init__(self,
        group_id: int,
        group_residue: str,
        target_group: TargetGroup,
        gaps: list[Gap],
    ):
        self.group_id = group_id
        self.group_residue = group_residue
        self.group_values = target_group
        self.n_gaps = len(gaps)
        self._gap_values = np.zeros(shape = self.n_gaps, dtype = float)
        self._gap_data = np.zeros(shape = (self.n_gaps, 3), dtype = int)
        for (i, (val, (l_idx, r_idx), local_id)) in enumerate(gaps):
            self._gap_values[i] = val
            self._gap_data[i, :] = [l_idx, r_idx, local_id]
    
    def __len__(self) -> int:
        return self.n_gaps
    
    def values(self) -> np.ndarray:
        """Returns the gap values as a numpy array."""
        return self._gap_values 

    def indices(self) -> np.ndarray:
        """Returns the gap indices as a 2×n numpy array."""
        return self._gap_data[:, :2]
    
    def index_tuples(self) -> list[tuple[int,int]]:
        """Returns the gap indices as a list of integer two-tuples."""
        return [(i, j) for i, j in self.indices()]
    
    def local_ids(self) -> np.ndarray:
        """Returns the list of local indices, relating gaps to potential modifications of that the ideal gap value."""
        return self._gap_data[:, 2]

#=============================================================================#

class TargetSpace:
    """Driver for gap search. Wraps linearithmic methods to find gaps in a peak list.""" 

    def __init__(self, 
        target_groups: list[TargetGroup],
        residues: list[chr],
        tolerance: float
    ):
        # store initialization data
        self.target_groups = target_groups
        self.residues = residues
        self.tolerance = tolerance
        # restructure data to support searching across all targets while recovering the group and local identities of a match.
        all_targets = collapse_second_order_list(
            [[(target, group_idx, local_idx) for (local_idx, target) in enumerate(group)] for (group_idx, group) in enumerate(target_groups)])
        all_targets.sort(key = lambda x: x[0])
        self.target_values, self.target_group_idx, self.target_local_idx = zip(*all_targets)
        self.min_target = min(self.target_values) - tolerance
        self.max_target = max(self.target_values) + tolerance
        # object size
        self.n_groups = len(target_groups)
        self.n_targets = len(all_targets)
    
    def _bound_bisection(self,
        idx: int
    ) -> int:
        return max(0, min(self.n_targets - 1, idx))
    
    def _bisect_gaps (self,
        peaks: np.ndarray
    ) -> list[Gap]:
        for (i, x) in enumerate(peaks):
            for (j, y) in enumerate(peaks[i + 1:]):
                dif = y - x
                if self.min_target <= dif <= self.max_target:
                    l = self._bound_bisection(bisect_left(self.target_values, dif - self.tolerance))
                    r = self._bound_bisection(bisect_right(self.target_values, dif + self.tolerance))
                    yield (dif, (i, i + j + 1), (l, r))
                elif dif > self.max_target:
                    break
    
    def _assign_target_groups(self,
        unassigned_gaps
    ) -> list[list[Gap]]:
        gaps_by_group = [[] for _ in range(self.n_groups)]
        uncategorized = []
        for (dif, gap, target_range) in unassigned_gaps:
            for target_match_idx in range(target_range[0], target_range[1] + 1):
                target_val = self.target_values[target_match_idx]
                if abs(dif - target_val) <= self.tolerance:
                    target_grp_idx = self.target_group_idx[target_match_idx]
                    target_lcl_idx = self.target_local_idx[target_match_idx]
                    gaps_by_group[target_grp_idx].append((dif, gap, target_lcl_idx))
                else:
                    uncategorized.append((dif, gap, -1))
        return gaps_by_group, uncategorized
    
    def find_gaps(self,
        peaks: np.ndarray
    ) -> list[GapResult]:
        "Given a np array of peak mz values, construct all gaps as a list of GapResult objects."
        unassigned_gaps = self._bisect_gaps(peaks)
        gaps_by_group, _ = self._assign_target_groups(unassigned_gaps)
        return [GapResult(group_id, self.residues[group_id], self.target_groups[group_id], result) 
            for group_id, result in enumerate(gaps_by_group)]
    
    def get_group_residue(self,
        group_id: int
    ) -> str:
        "Returns the identifier residue of a target group."
        return self.residues[group_id]

#=============================================================================#

def find_all_gaps(
    spectrum: np.ndarray,
    target_groups: list[TargetGroup],
    tolerance: float,
    verbose = False,
) -> list[list[Gap]]:
    "deprecated method, but identical to TargetSpace(target_groups, [residues], tolerance).find_gaps(spectrum)"
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