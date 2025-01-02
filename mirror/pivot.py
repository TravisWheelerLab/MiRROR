import numpy as np
from statistics import mode

from .util import mass_error, count_mirror_symmetries, residue_lookup, reflect, INTERGAP_TOLERANCE, find_initial_b_ion, find_terminal_y_ion
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
        peak_pairs = self.peak_pairs()
        gap_a = peak_pairs[0][1] - peak_pairs[0][0]
        gap_b = peak_pairs[1][1] - peak_pairs[1][0]
        return (gap_a + gap_b) / 2

    def initial_b(
        self,
        spectrum,
    ):
        return list(find_initial_b_ion(spectrum, self.outer_right(), len(spectrum), self.center()))

    def terminal_y(
        self,
        spectrum,
    ):
        return list(find_terminal_y_ion(spectrum, self.outer_left()))
    
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

class VirtualPivot(Pivot):

    def __init__(self, index_data, virtual_center):
        self.index_data = index_data
        self._center = virtual_center
    
    def peaks(self):
        raise NotImplementedError("Virtual pivots are not associated to peaks.")

    def indices(self):
        return self.index_data

    def peak_pairs(self):
        raise NotImplementedError("Virtual pivots are not associated to peaks.")
    
    def index_pairs(self):
        raise NotImplementedError("Virtual pivots are not composed of pairs.")
    
    def outer_left(self):
        return self.index_data[0]
    
    def inner_left(self):
        return self.index_data[0]
    
    def inner_right(self):
        return self.index_data[1]
    
    def outer_right(self):
        return self.index_data[1]

    def center(self):
        return self._center
    
    def gap(self):
        return -1
    
    def negative_index_pairs(self):
        """index pairs that should not be present in the gap set."""
        return []

    def __repr__(self):
        return f"VirtualPivot{self.indices(), self.center()}"
        

#=============================================================================#
# utility functions for finding pivots in the gap structure of a spectrum

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

def find_disjoint_pivots(
    spectrum,
    gap_indices,
    tolerance,
):
    return find_pivots(spectrum, gap_indices, PivotDisjointConstraint(tolerance))

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
# discerning (or otherwise constructing) good pivots.

def _construct_virtual_pivot(
    spectrum: np.ndarray,
    low: int,
    high: int,
    center: float,
):
    for i in range(low, high - 1):
        left_peak = spectrum[i]
        right_peak = spectrum[i + 1]
        if left_peak < center < right_peak:
            return VirtualPivot(
                [i, i + 1],
                center
            )

def _filter_viable_pivots(
    spectrum: np.ndarray,
    symmetry_threshold: float,
    pivots: list[Pivot],
):
    if len(pivots) == 0:
        return pivots
    else:
        pivot_symmetries = np.array([count_mirror_symmetries(spectrum, pivot.center()) for pivot in pivots])
        pivot_initial_b_ions = np.array([pivot.initial_b(spectrum) != [] for pivot in pivots])
        pivot_terminal_y_ions = np.array([pivot.terminal_y(spectrum) != [] for pivot in pivots])
        pivot_residues = np.array([residue_lookup(pivot.gap()) for pivot in pivots])
        try:
            viable = pivot_symmetries > symmetry_threshold
            viable *= pivot_initial_b_ions
            viable *= pivot_terminal_y_ions
        except Exception as e:
            print(e)
            print(pivots)
            print(viable)
            print(pivot_initial_b_ions)
            print(pivot_terminal_y_ions)
            raise e
        if any(pivot_residues != 'X'):
            viable *= pivot_residues != 'X'
        n = len(pivots)
        return [pivots[i] for i in range(n) if viable[i]]

def _reconstruct_pivots(
    spectrum: np.ndarray,
    symmetry_threshold: float,
    pivots: list[Pivot],
):
    n = len(pivots)
    pivot_center = [pivot.center() for pivot in pivots]
    for i in range(n):
        for j in range(i + 1, n):
            reconstructed_center = pivot_center[i] + pivot_center[j]
            if count_mirror_symmetries(spectrum, reconstructed_center) > symmetry_threshold:
                all_indices = pivots[i].indices + pivots[j].indices
                yield _construct_virtual_pivot(spectrum, min(all_indices), max(all_indices), reconstructed_center)

def _construct_viable_pivots(
    spectrum: np.ndarray,
    symmetry_threshold: float,
    gap_indices: list[tuple[int,int]],
    tolerance: float,
):
    # look for overlapping pivots; if none are found, fall back on disjoint pivots.
    pivots = find_overlapping_pivots(spectrum, gap_indices, tolerance)
    if len(pivots) == 0:
        disjoint_pivots = find_disjoint_pivots(spectrum, gap_indices, tolerance)
        mode_center = mode(pivot.center() for pivot in disjoint_pivots)
        pivot = _construct_virtual_pivot(spectrum, 0, len(spectrum), mode_center)
        return [pivot]
    
    # discard low-scoring pivots. if there are no viable pivots, reconstruct.
    viable_pivots = _filter_viable_pivots(spectrum, symmetry_threshold, pivots)
    if len(viable_pivots) == 0:
        return list(_reconstruct_pivots(spectrum, symmetry_threshold, pivots))
    else:
        return viable_pivots

def construct_all_pivots(
    spectrum: np.ndarray,
    gap_indices: list[tuple[int,int]],
    tolerance: float,
):

    o_pivots = find_overlapping_pivots(spectrum, gap_indices, tolerance)
    a_pivots = find_adjacent_pivots(spectrum, tolerance)
    return o_pivots + a_pivots

def construct_viable_pivots(
    spectrum: np.ndarray,
    symmetry_threshold: float,
    gap_indices: list[tuple[int,int]],
    tolerance: float = INTERGAP_TOLERANCE,
):
    pivots = construct_all_pivots(spectrum, gap_indices, tolerance)
    viable_pivots = _filter_viable_pivots(spectrum, symmetry_threshold, pivots)
    #print(len(viable_pivots), len(pivots))
    return viable_pivots