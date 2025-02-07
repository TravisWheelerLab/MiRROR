from statistics import mode

import numpy as np

from .util import mass_error, count_mirror_symmetries, residue_lookup, reflect, find_initial_b_ion, find_terminal_y_ion

#=============================================================================#

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
        peak_pairs = self.peak_pairs()
        gap_a = peak_pairs[0][1] - peak_pairs[0][0]
        gap_b = peak_pairs[1][1] - peak_pairs[1][0]
        return (gap_a + gap_b) / 2
    
    def negative_index_pairs(self):
        """index pairs that should not be present in the gap set."""
        inds_a, inds_b = self.index_pairs()
        negative_pairs = [(inds_a[0], inds_b[0]),
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
    return list(_find_adjacent_pivots(spectrum, tolerance))

#=============================================================================#

def filter_viable_pivots(
    spectrum: np.ndarray,
    symmetry_threshold: float,
    pivots: list[Pivot],
):
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
    gap_indices: list[tuple[int,int]],
    tolerance: float,
):
    o_pivots = find_overlapping_pivots(spectrum, gap_indices, tolerance)
    a_pivots = find_adjacent_pivots(spectrum, tolerance)
    return o_pivots + a_pivots

def find_pivots(
    spectrum: np.ndarray,
    symmetry_threshold: float,
    gap_indices: list[tuple[int,int]],
    tolerance: float,
):
    pivots = construct_all_pivots(spectrum, gap_indices, tolerance)
    viable_pivots = filter_viable_pivots(spectrum, symmetry_threshold, pivots)
    return viable_pivots