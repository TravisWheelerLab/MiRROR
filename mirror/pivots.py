from statistics import mode

import numpy as np

from .gaps import GapResult, _find_gaps_without_targets
from .util import collapse_second_order_list, mass_error, count_mirror_symmetries, residue_lookup, reflect, find_initial_b_ion, find_terminal_y_ion

#=============================================================================#

class Pivot:
    """Interface to a pivot identified from a pair of gaps."""

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

    def residue(self):
        return residue_lookup(self.gap())
    
    def gap(self):
        peak_pairs = self.peak_pairs()
        gap_a = peak_pairs[0][1] - peak_pairs[0][0]
        gap_b = peak_pairs[1][1] - peak_pairs[1][0]
        return (gap_a + gap_b) / 2
    
    def negative_index_pairs(self):
        """index pairs that should not be present in the gap set."""
        inds_a, inds_b = self.index_pairs()
        negative_pairs = [
            (inds_a[0], inds_b[0]),
            (inds_a[1], inds_b[1]),
            (inds_a[1], inds_b[0]),
            (inds_a[0], inds_b[1])]
        negative_pairs = [(min(e), max(e)) for e in negative_pairs]
        return negative_pairs

    def __repr__(self):
        return f"""Pivot(
\tpeaks = {self.peak_pairs()}
\tindices = {self.index_pairs()}
\tresidue = {self.residue()}
)"""
    
    def __eq__(self, other):
        return self.peaks() == other.peaks()
        
#=============================================================================#
# utility functions for finding pivots in the gap structure of a spectrum

def _find_disjoint_pivots(
    spectrum,
    gap_indices,
    tolerance,
):
    n = len(spectrum)
    for (i, i2) in gap_indices:
        i_gap = spectrum[i2] - spectrum[i]
        for j in range(i2 + 1, n):
            for j2 in range(j + 1, n):
                j_gap = spectrum[j2] - spectrum[j]
                gap_dif = abs(i_gap - j_gap)
                if gap_dif < tolerance:
                    yield Pivot((spectrum[i], spectrum[i2]), (spectrum[j], spectrum[j2]), (i, i2), (j, j2))
                elif j_gap > i_gap:
                    break

def find_disjoint_pivots(
    spectrum,
    gap_indices,
    tolerance,
):
    """Find pairs of indexes (i1,i2), (j1,j2) into a list of peaks such that 
    1. i1 < i2 < j1 < j2, that is, the intervals [i1, i2] and [j1, j2] are disjoint.
    2. the difference  between spectrum[i2] - spectrum[i1] and spectrum[j2] - spectrum[j1] is less than `tolerance`.
    
    :spectrum: a sorted array of floats (peak mz values).
    :gap_indices: a list of integer 2-tuples which index into `spectrum`.
    :tolerance: float, the threshold difference for equating two gaps."""
    return list(_find_disjoint_pivots(spectrum, gap_indices, tolerance))

def _find_overlapping_pivots(
    spectrum,
    gap_indices,
    tolerance,
):
    n = len(spectrum)
    for (i, i2) in gap_indices:
        i_gap = spectrum[i2] - spectrum[i]
        for j in range(i, i2):
            for j2 in range(i2 + 1, n):
                j_gap = spectrum[j2] - spectrum[j]
                gap_dif = abs(i_gap - j_gap)
                if gap_dif < tolerance:
                    yield Pivot((spectrum[i], spectrum[i2]), (spectrum[j], spectrum[j2]), (i, i2), (j, j2))
                elif j_gap > i_gap:
                    break

def find_overlapping_pivots(
    spectrum,
    gap_indices,
    tolerance,
):
    """Find pairs of indexes (i1,i2), (j1,j2) into a list of peaks such that 
    1. i1 < j1 < i2 < j2, that is, the intervals [i1, i2] and [j1, j2] overlap.
    2. the difference  between spectrum[i2] - spectrum[i1] and spectrum[j2] - spectrum[j1] is less than `tolerance`.
    
    :spectrum: a sorted array of floats (peak mz values).
    :gap_indices: a list of integer 2-tuples which index into `spectrum`.
    :tolerance: float, the threshold difference for equating two gaps."""
    return list(_find_overlapping_pivots(spectrum, gap_indices, tolerance))

def _find_adjacent_pivots(
    spectrum,
    tolerance,
):
    n = len(spectrum)
    for i in range(n - 4):
        i2 = i + 1
        i_gap = spectrum[i2] - spectrum[i]
        j = i + 2
        j2 = j + 1
        j_gap = spectrum[j2] - spectrum[j]
        if abs(i_gap - j_gap) < tolerance:
            yield Pivot((spectrum[i], spectrum[i2]), (spectrum[j], spectrum[j2]), (i, i2), (j, j2))

def find_adjacent_pivots(
    spectrum,
    tolerance,
):
    """Find pairs of indexes (i1,i2), (j1,j2) into a list of peaks such that 
    1. i1 < i2 < j1 < j2, that is, the intervals [i1, i2] and [j1, j2] are disjoint.
    2. (i1, i2, j1, j2) = (i1, i1 + 1, i1 + 2, i1 + 3), that is, the indices are all adjacent.
    3. the difference  between spectrum[i2] - spectrum[i1] and spectrum[j2] - spectrum[j1] is less than `tolerance`.
    
    :spectrum: a sorted array of floats (peak mz values).
    :tolerance: float, the threshold difference for equating two gaps."""
    return list(_find_adjacent_pivots(spectrum, tolerance))

#=============================================================================#

def filter_viable_pivots(
    spectrum: np.ndarray,
    symmetry_threshold: float,
    pivots: list[Pivot],
    filter_unknown_residues: bool = True
):
    """Filter out pivots that fall below the symmetry threshold and do not produce legible residues.
    
    :spectrum: a sorted array of floats (peak mz values).
    :symmetry_threshold: the expected number of mirror-symmetric peaks in the spectrum.
    :pivots: a list of Pivot objects."""
    if len(pivots) == 0:
        return pivots
    else:
        pivot_symmetries = np.array([count_mirror_symmetries(spectrum, pivot.center()) for pivot in pivots])
        #pivot_initial_b_ions = np.array([pivot.initial_b(spectrum) != [] for pivot in pivots])
        #pivot_terminal_y_ions = np.array([pivot.terminal_y(spectrum) != [] for pivot in pivots])
        pivot_residues = np.array([residue_lookup(pivot.gap()) for pivot in pivots])
        try:
            viable = pivot_symmetries > symmetry_threshold
            #viable *= pivot_initial_b_ions
            #viable *= pivot_terminal_y_ions
        except Exception as e:
            print(e)
            print(pivots)
            print(viable)
            #print(pivot_initial_b_ions)
            #print(pivot_terminal_y_ions)
            raise e
        if any(pivot_residues != 'X') and filter_unknown_residues:
            viable *= pivot_residues != 'X'
        n = len(pivots)
        return [pivots[i] for i in range(n) if viable[i]]

def construct_all_pivots(
    spectrum: np.ndarray,
    gap_results: list[GapResult],
    tolerance: float,
):
    """Find pairs of indexes (i1,i2), (j1,j2) into a list of peaks such that 
    1. the difference  between spectrum[i2] - spectrum[i1] and spectrum[j2] - spectrum[j1] is less than `tolerance`.
    2. the pivots are either (a) overlapping or (b) disjoint:
        2a. i1 < j1 < i2 < j2, that is, the intervals [i1, i2] and [j1, j2] overlap.
        2b. (i1, i2, j1, j2) = (i1, i1 + 1, i1 + 2, i1 + 3), that is, the indices are all adjacent.
    
    :spectrum: a sorted array of floats (peak mz values).
    :gap_results: a list of GapResult objects, organizing gaps by gap value according to targeted residue weights.
    :tolerance: float, the threshold difference for equating two gaps."""
    o_pivots = collapse_second_order_list(
        [find_overlapping_pivots(spectrum, r.get_index_pairs(), tolerance) for r in gap_results]
    )
    a_pivots = find_adjacent_pivots(spectrum, tolerance)
    return o_pivots + a_pivots

def find_all_pivots(
    spectrum: np.ndarray,
    symmetry_threshold: float,
    gap_results: list[GapResult],
    tolerance: float,
):
    """Find pairs of indexes (i1,i2), (j1,j2) into a list of peaks such that 
    1. the difference  between spectrum[i2] - spectrum[i1] and spectrum[j2] - spectrum[j1] is less than `tolerance`.
    2. the pivots are either (a) overlapping or (b) disjoint:
        2a. i1 < j1 < i2 < j2, that is, the intervals [i1, i2] and [j1, j2] overlap.
        2b. (i1, i2, j1, j2) = (i1, i1 + 1, i1 + 2, i1 + 3), that is, the indices are all adjacent.
    Then, filter the pivots to remove pivots whose centers do not induce sufficient mirror symmetry in the peak list.
    
    :spectrum: a sorted array of floats (peak mz values).
    :symmetry_threshold: the expected number of mirror-symmetric peaks in the spectrum.
    :gap_results: a list of GapResult objects, organizing gaps by gap value according to targeted residue weights.
    :tolerance: float, the threshold difference for equating two gaps."""
    pivots = construct_all_pivots(spectrum, gap_results, tolerance)
    viable_pivots = filter_viable_pivots(spectrum, symmetry_threshold, pivots)
    return viable_pivots

#=============================================================================#

def _find_overlapping_pivots_without_targets(
    spectrum: np.ndarray,
    sorted_gaps: list[tuple[int, int]],
    tolerance: float, 
):
    n = len(sorted_gaps)
    for i in range(n):
        p1, p2 = sorted_gaps[i]
        p_gap = spectrum[p2] - spectrum[p1]
        for j in range(i + 1, n):
            q1, q2 = sorted_gaps[j]
            q_gap = spectrum[q2] - spectrum[q1]
            if not(p1 < q1 < p2 < q2) or (q_gap - p_gap > tolerance):
                # the gap list is sorted, so increasing j will only increase q gap. 
                # if the difference is already greater than the threshold, there aren't any (more) matches. 
                break
            elif q_gap - p_gap < tolerance:
                yield Pivot((spectrum[p1], spectrum[p2]), (spectrum[q1], spectrum[q2]), (p1, p2), (q1, q2))

def find_overlapping_pivots_without_targets(
    spectrum: np.ndarray,
    sorted_gaps: list[tuple[int, int]],
    tolerance: float, 
):
    """Find pairs of indexes (i1,i2), (j1,j2) into a list of peaks such that 
    1. i1 < j1 < i2 < j2, that is, the intervals [i1, i2] and [j1, j2] overlap.
    2. the difference  between spectrum[i2] - spectrum[i1] and spectrum[j2] - spectrum[j1] is less than `tolerance`.
    
    :spectrum: a sorted array of floats (peak mz values).
    :gap_indices: a list of integer 2-tuples which index into `spectrum`, sorted to be ascending w.r.t. gap length.
    :tolerance: float, the threshold difference for equating two gaps."""
    return list(_find_overlapping_pivots(spectrum, sorted_gaps, tolerance))

def find_all_pivots_gap_agnostic(
    spectrum: np.ndarray,
    symmetry_threshold: float,
    gap_bounds: tuple[float, float],
    tolerance: float,
):
    """Without using precomputed gaps or a TargetSpace object, 
    find pairs of indexes (i1,i2), (j1,j2) into a list of peaks such that 
    1. the difference  between spectrum[i2] - spectrum[i1] and spectrum[j2] - spectrum[j1] is less than `tolerance`.
    2. the pivots are either (a) overlapping or (b) disjoint:
        2a. i1 < j1 < i2 < j2, that is, the intervals [i1, i2] and [j1, j2] overlap.
        2b. (i1, i2, j1, j2) = (i1, i1 + 1, i1 + 2, i1 + 3), that is, the indices are all adjacent.
    Then, filter the pivots to remove pivots whose centers do not induce sufficient mirror symmetry in the peak list.
    
    :spectrum: a sorted array of floats (peak mz values).
    :symmetry_threshold: the expected number of mirror-symmetric peaks in the spectrum.
    :gap_bounds: a float 2-tuple (lower_bound, upper_bound) that determines when quadratic search should be cut off.
    :tolerance: float, the threshold difference for equating two gaps."""
    # find all gaps; sort them. 
    # in lieu of binsorting by targets, the pivot finding functions must be passed a list sorted by gap value.
    candidate_pairs = sorted(
        _find_gaps_without_targets(spectrum, *gap_bounds, tolerance), 
        key = lambda x: spectrum[x[1]] - spectrum[x[0]])
    # find overlapping and adjacent pivots
    overlap_candidates = find_overlapping_pivots_without_targets(spectrum, candidate_pairs, tolerance)
    adjacent_candidates = find_adjacent_pivots(spectrum, tolerance)
    candidate_pivots = overlap_candidates + adjacent_candidates
    # remove pivots below the symmetry threshold
    return filter_viable_pivots(
        spectrum, 
        symmetry_threshold, 
        candidate_pivots, 
        filter_unknown_residues = False)