from statistics import mode

import numpy as np

from .gaps import GapResult
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
        return f"Pivot{*self.peak_pairs(), *self.index_pairs()}"
    
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
    """Find pairs of indexes (i₁,i₂), (j₁,j₂) into a list of peaks such that 
    1. i₁ < i₂ < j₁ < j₂, that is, the intervals [i₁, i₂] and [j₁, j₂] are disjoint.
    2. the difference  between spectrum[i₂] - spectrum[i₁] and spectrum[j₂] - spectrum[j₁] is less than `tolerance`.
    
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
    """Find pairs of indexes (i₁,i₂), (j₁,j₂) into a list of peaks such that 
    1. i₁ < j₁ < i₂ < j₂, that is, the intervals [i₁, i₂] and [j₁, j₂] overlap.
    2. the difference  between spectrum[i₂] - spectrum[i₁] and spectrum[j₂] - spectrum[j₁] is less than `tolerance`.
    
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
    """Find pairs of indexes (i₁,i₂), (j₁,j₂) into a list of peaks such that 
    1. i₁ < i₂ < j₁ < j₂, that is, the intervals [i₁, i₂] and [j₁, j₂] are disjoint.
    2. (i₁, i₂, j₁, j₂) = (i₁, i₁ + 1, i₁ + 2, i₁ + 3), that is, the indices are all adjacent.
    3. the difference  between spectrum[i₂] - spectrum[i₁] and spectrum[j₂] - spectrum[j₁] is less than `tolerance`.
    
    :spectrum: a sorted array of floats (peak mz values).
    :tolerance: float, the threshold difference for equating two gaps."""
    return list(_find_adjacent_pivots(spectrum, tolerance))

#=============================================================================#

def filter_viable_pivots(
    spectrum: np.ndarray,
    symmetry_threshold: float,
    pivots: list[Pivot],
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
        if any(pivot_residues != 'X'):
            viable *= pivot_residues != 'X'
        n = len(pivots)
        return [pivots[i] for i in range(n) if viable[i]]

def construct_all_pivots(
    spectrum: np.ndarray,
    gap_results: list[GapResult],
    tolerance: float,
):
    """Find pairs of indexes (i₁,i₂), (j₁,j₂) into a list of peaks such that 
    1. the difference  between spectrum[i₂] - spectrum[i₁] and spectrum[j₂] - spectrum[j₁] is less than `tolerance`.
    2. the pivots are either (a) overlapping or (b) disjoint:
        2a. i₁ < j₁ < i₂ < j₂, that is, the intervals [i₁, i₂] and [j₁, j₂] overlap.
        2b. (i₁, i₂, j₁, j₂) = (i₁, i₁ + 1, i₁ + 2, i₁ + 3), that is, the indices are all adjacent.
    
    :spectrum: a sorted array of floats (peak mz values).
    :gap_indices: a list of integer 2-tuples which index into `spectrum`.
    :tolerance: float, the threshold difference for equating two gaps."""
    o_pivots = collapse_second_order_list(
        [find_overlapping_pivots(spectrum, r.index_tuples(), tolerance) for r in gap_results]
    )
    a_pivots = find_adjacent_pivots(spectrum, tolerance)
    return o_pivots + a_pivots

def find_all_pivots(
    spectrum: np.ndarray,
    symmetry_threshold: float,
    gap_results: list[GapResult],
    tolerance: float,
):
    """Find pairs of indexes (i₁,i₂), (j₁,j₂) into a list of peaks such that 
    1. the difference  between spectrum[i₂] - spectrum[i₁] and spectrum[j₂] - spectrum[j₁] is less than `tolerance`.
    2. the pivots are either (a) overlapping or (b) disjoint:
        2a. i₁ < j₁ < i₂ < j₂, that is, the intervals [i₁, i₂] and [j₁, j₂] overlap.
        2b. (i₁, i₂, j₁, j₂) = (i₁, i₁ + 1, i₁ + 2, i₁ + 3), that is, the indices are all adjacent.
    Then, filter the pivots to remove pivots whose centers do not induce sufficient mirror symmetry in the peak list.
    
    :spectrum: a sorted array of floats (peak mz values).
    :symmetry_threshold: the expected number of mirror-symmetric peaks in the spectrum.
    :gap_indices: a list of integer 2-tuples which index into `spectrum`.
    :tolerance: float, the threshold difference for equating two gaps."""
    pivots = construct_all_pivots(spectrum, gap_results, tolerance)
    viable_pivots = filter_viable_pivots(spectrum, symmetry_threshold, pivots)
    return viable_pivots