#

from sortedcontainers import SortedList
import numpy as np
from copy import deepcopy

from .util import product, collapse_second_order_list, reflect, residue_lookup, find_initial_b_ion, find_terminal_y_ion
from .gaps import GapSearchParameters, GapResult, find_gaps
from .pivots import Pivot

#=============================================================================#
# helper functions for augmentations

def _create_augmented_spectrum(
    spectrum: np.ndarray,
    pivot: Pivot,
    b_idx: tuple[int, str],
    y_idx: tuple[int, str],
    tolerance: float,
    padding: int = 3
):
    # reflect the boundaries
    center = pivot.center()
    b_mz = spectrum[b_idx]
    y_mz = spectrum[y_idx]
    
    # augment the spectrum
    n = len(spectrum)
    subspectrum = spectrum[max(0, y_idx - padding):min(n - 1, b_idx + padding + 1)]
    augmented_spectrum = SortedList(subspectrum)
    boundaries = []
    for val in [b_mz, y_mz]:
        reflected_val = reflect(val, center)
        for aug_peak in [val, reflected_val]:
            min_dif = np.inf
            closest_peak = -1
            for peak in subspectrum:  
                dif = abs(peak - aug_peak)
                if dif < min_dif:
                    min_dif = dif
                    closest_peak = peak
            if min_dif > tolerance:
                augmented_spectrum.add(aug_peak)
                boundaries.append(aug_peak)
            else:
                boundaries.append(closest_peak)

    # shift the pivot
    pivot_left = pivot.outer_left()
    pivot_left_mz = spectrum[pivot_left]
    new_left = len([val for val in augmented_spectrum if val < pivot_left_mz])
    offset = new_left - pivot_left
    
    boundary_indices = [augmented_spectrum.index(mz) for mz in boundaries]
    return np.array(augmented_spectrum), offset, boundaries, boundary_indices

def _create_augmented_pivot(
    augmented_spectrum: np.ndarray,
    offset: int,
    pivot: Pivot,
):
    if type(pivot) == Pivot:
        index_pairs = pivot.index_pairs()
        new_index_pairs = [(i + offset, j + offset) for (i, j) in index_pairs]
        new_peak_pairs = [(augmented_spectrum[i], augmented_spectrum[j]) for (i,j) in new_index_pairs]
        index_shifted_pivot = Pivot(*new_peak_pairs, *new_index_pairs)
        try:
            assert index_shifted_pivot.peaks() == pivot.peaks()
        except AssertionError as e:
            print(index_shifted_pivot.peaks(), pivot.peaks())
            raise e
        return index_shifted_pivot
    else:
        raise ValueError(f"Unrecognized pivot type {type(pivot)}")

def _create_augmented_gaps(
    augmented_spectrum: np.ndarray,
    augmented_pivot: Pivot,
    offset: int,
    gap_params: GapSearchParameters,
    nonviable_gaps: list[tuple[int,int]]
):
    annotated_augmented_spectrum, augmented_gap_results = find_gaps(gap_params, augmented_spectrum)
    augmented_gaps = collapse_second_order_list(r.get_index_pairs() for r in augmented_gap_results)
    augmented_gaps = [(i, j) for (i, j) in augmented_gaps if (i, j) not in nonviable_gaps]
    augmented_gaps = list(set(augmented_gaps))
    augmented_gaps.sort(key = lambda x: x[0] + x[1] / 1000)
    return augmented_gaps

#=============================================================================#
# interface

class Boundary:
    """Construct the augmented peak list, pivot, and gaps by restricting the spectrum to a window 
    around the left and right ions, and forcing symmetry by mirroring each boundary around the 
    pivot center.

    Implements get_residues, get_offset, get_augmented peaks, get_augmented pivot, get_augmented gaps."""

    def __init__(self,
        spectrum: np.ndarray,
        pivot: Pivot,
        gap_params: GapSearchParameters,
        boundary_pair: tuple[tuple[int, float],tuple[int, float]],
        valid_terminal_residues: list,
        padding: int = 3,
    ):
        # stored fields
        self._spectrum = spectrum
        self._pivot = pivot
        self._gap_params = deepcopy(gap_params)
        self._gap_params.charges = np.array([])
        self._valid_terminal_residues = valid_terminal_residues
        self._tolerance = gap_params.tolerance
        self._padding = padding
        
        # unpack and store individually
        b_boundary, y_boundary = boundary_pair
        self._b_idx, self._b_res = b_boundary
        self._y_idx, self._y_res = y_boundary

        # create augmented data
        self._augmented_spectrum , self._offset, self._boundary_values, self._boundary_indices = _create_augmented_spectrum(
            self._spectrum,
            self._pivot,
            self._b_idx,
            self._y_idx,
            self._tolerance,
            self._padding,
        )

        self._augmented_pivot = _create_augmented_pivot(
            self._augmented_spectrum,
            self._offset,
            self._pivot,
        )
        
        self._augmented_gaps = _create_augmented_gaps(
            self._augmented_spectrum,
            self._augmented_pivot,
            self._offset,
            self._gap_params,
            self._augmented_pivot.negative_index_pairs(),
        )

    def get_residues(self):
        "Returns the b and y residues of the boundary."
        return self._b_res, self._y_res
    
    def get_boundary_indices(self):
        return self._boundary_indices

    def get_boundary_values(self):
        return self._boundary_values
    
    def get_augmented_peaks(self):
        "Augments the peak array by restricting to the boundary index range and adding symmetric boundary conditions."
        return self._augmented_spectrum
    
    def get_offset(self):
        """The integer by which pivot index values are offset to match the augmented peak array:
        for each i in pivot.indices(), original_spectrum[i] = augmented_spectrum[i + offset]."""
        return self._offset
    
    def get_augmented_pivot(self):
        """The pivot created by translating its pivot set by the offset value, so that the
        mz values relative to the augmented peaks are the same."""
        return self._augmented_pivot
    
    def get_augmented_gaps(self):
        """Finds all gaps in the augmented peaks, collecting and collapsing GapResult objects 
        into a flat list of integer 2-tuples."""
        return self._augmented_gaps
    
    def __repr__(self):
        return f"""Boundary(
    indices = {self.get_boundary_indices()}
    peaks = {self.get_boundary_values()}
    residues = {self.get_residues()}
)"""
    
def find_and_create_boundaries(
    spectrum: np.ndarray,
    terminal_y_ions: list[tuple[int, float]],
    pivot: Pivot,
    gap_params: GapSearchParameters,
    valid_terminal_residues: list,
    padding: int = 3,
):
    """Given the terminal y ion(s), find the initial b ion(s), and for each element in their product, construct a Boundary object.
    
    :spectrum: a sorted array of floats (peak mz values).
    :pivot: a Pivot object.
    :target_space: a TargetSpace object.
    :valid_terminal_residues: list of valid residues for the y boundary ions. For tryptic peptides, these are K and R.
    :tolerance: float, the threshold for equating two gaps.
    :padding: integer, the number of peaks outside of the boundary to include in the augmented spectrum."""
    initial_b_ions = list(find_initial_b_ion(spectrum, pivot.outer_right(), len(spectrum), pivot.center()))
    boundaries = [Boundary(spectrum, pivot, gap_params, boundary_pair, valid_terminal_residues, padding) 
        for boundary_pair in product(initial_b_ions, terminal_y_ions)]
    return boundaries, initial_b_ions