from bisect import bisect_left, bisect_right

import numpy as np

from .types import TargetGroup
from .util import collapse_second_order_list, WATER_LOSS_OFFSET, AMMONIA_LOSS_OFFSET

#=============================================================================#

class GapResult:
    """Interface to the result of gap search.
    Collects tuples of indices into a spectrum, representing pairs of peaks whose distance relates to a given set of target values.
    Implements __len__, values, indices, index_tuples, local_ids, internal_indices, and charge_state_pairs"""

    def __init__(self,
        group_id: int,
        group_residue: str,
        target_group: TargetGroup,
        gap_data: list[list[float]],
    ):
        self.group_id = group_id
        self.group_residue = group_residue
        self.group_values = target_group
        self.n_gaps = len(gap_data)
        self._gap_data = np.vstack(gap_data)
    
    def __len__(self) -> int:
        return self.n_gaps
    
    def values(self) -> np.ndarray:
        """Return the gap values as a numpy array."""
        return self._gap_data[:, 0]
    
    def index_tuples(self) -> list[tuple[int,int]]:
        """Return the gap indices as a list of integer two-tuples."""
        return [(i, j) for i, j in self.indices()]
    
    def local_ids(self) -> np.ndarray:
        """Return the list of local indices, relating gaps to potential modifications of that the ideal gap value."""
        return self._gap_data[:, 3].astype(int)

    def indices(self) -> np.ndarray:
        """Return the gap indices as a 2×n numpy array."""
        return self._gap_data[:, 1:3].astype(int)
    
    def internal_indices(self) -> np.ndarray:
        """Return the internal gap indices as a 2×n numpy array. 
        Internal indices reference the combined peak list that includes 2x and 3x peak values,
        simulating the inverse of +2 and +3 charge states."""
        return self._gap_data[:, 4:6].astype(int)
    
    def charge_state_pairs(self) -> np.ndarray:
        """Return the charge states of each peak implicated in a gap as a 2×n numpy array.
        Taken with `internal_indices`, these data can be used to recover the peak lists from which
        a given gap was constructed."""
        return self._gap_data[:, 6:8]

#=============================================================================#

class TargetSpace:
    """Driver for gap search. Given a list of TargetGroups, themselves each a list of target values, the TargetSpace constructs methods 
    for finding gaps in spectra that match these target values. Supports +2/+3 charge states and loss of water and ammonia.
    Implements find_gaps, get_group_residue.""" 

    def __init__(self, 
        target_groups: list[TargetGroup],
        residues: list[chr],
        tolerance: float,
        charge_states: list[float] = [1.0, 2.0, 3.0],
        first_order_losses = [WATER_LOSS_OFFSET, AMMONIA_LOSS_OFFSET],
    ):
        # store initialization data
        self.target_groups = target_groups
        self.residues = residues
        self.tolerance = tolerance
        self.charge_states = sorted(charge_states)
        self.first_order_loses = first_order_losses
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
    
    def _create_charge_states(self,
        original_peaks: np.ndarray,
    ):
        decharged_peaks = [charge * original_peaks for charge in self.charge_states]
        indices = [np.arange(arr.shape[0]) for arr in decharged_peaks]
        charges = [np.full(shape = arr.shape, fill_value = charge) for (charge, arr) in zip(self.charge_states, decharged_peaks)]
        combined_peaks = np.concatenate(decharged_peaks)
        combined_indices = np.concatenate(indices)
        combined_charges = np.concatenate(charges)
        order = np.argsort(combined_peaks)
        return combined_peaks[order], combined_indices[order], combined_charges[order]
    
    def _bound_bisection(self,
        idx: int,
    ) -> int:
        return max(0, min(self.n_targets - 1, idx))
    
    def _match_bounds(self,
        query: float,
    ) -> (int, int):
        l = self._bound_bisection(bisect_left(self.target_values, query - self.tolerance))
        r = self._bound_bisection(bisect_right(self.target_values, query + self.tolerance))
        return (l, r)
    
    def _bisect_gaps (self,
        peaks: np.ndarray,
    ) -> list[list[float]]:
        for (i, x) in enumerate(peaks):
            for (j, y) in enumerate(peaks[i + 1:]):
                dif = y - x
                if self.min_target <= dif <= self.max_target:
                    l, r = self._match_bounds(dif)
                    yield (dif, (i, i + j + 1), (l, r))
                elif dif > self.max_target:
                    break
    
    def _assign_target_groups(self,
        unassigned_gaps,
        index_lookup,
        charge_lookup,
    ) -> list[list[list[float]]]:
        gaps_by_group = [[] for _ in range(self.n_groups)]
        uncategorized = []
        for (dif, gap, target_range) in unassigned_gaps:
            topological_gap = (index_lookup[gap[0]], index_lookup[gap[1]])
            if topological_gap[0] == topological_gap[1]:
                continue
            charge_states = (charge_lookup[gap[0]], charge_lookup[gap[1]])
            for target_match_idx in range(target_range[0], target_range[1] + 1):
                target_val = self.target_values[target_match_idx]
                if abs(dif - target_val) <= self.tolerance:
                    target_grp_idx = self.target_group_idx[target_match_idx]
                    target_lcl_idx = self.target_local_idx[target_match_idx]
                    gaps_by_group[target_grp_idx].append(
                        [dif, *topological_gap, target_lcl_idx, *gap, *charge_states]
                    )
                else:
                    uncategorized.append((dif, gap, -1))
        return gaps_by_group, uncategorized
    
    def find_gaps(self,
        peaks: np.ndarray,
    ) -> list[GapResult]:
        "Given a numpy array of peaks (mz values), construct all gaps matching targets as a list of GapResult objects."
        combined_peaks, index_table, charge_table = self._create_charge_states(peaks)
        unassigned_gaps = self._bisect_gaps(combined_peaks)
        gaps_by_group, _ = self._assign_target_groups(unassigned_gaps, index_table, charge_table)
        return [GapResult(group_id, self.residues[group_id], self.target_groups[group_id], result) 
            for group_id, result in enumerate(gaps_by_group)]
    
    def get_group_residue(self,
        group_id: int,
    ) -> str:
        "Given the group id integer, return the identifier residue of a target group."
        return self.residues[group_id]

#=============================================================================#

def _find_gaps_without_targets(
    spectrum: np.ndarray,
    min_gap: float,
    max_gap: float,
    tolerance: float,
):
    n = len(spectrum)
    for i in range(n):
        for j in range(i + 1, n):
            gap_dif = spectrum[j] - spectrum[i]
            if gap_dif > max_gap + tolerance:
                break
            elif gap_dif >= min_gap - tolerance:
                yield (i, j)