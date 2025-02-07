#

from sortedcontainers import SortedList
import numpy as np

from .util import reflect, residue_lookup, find_initial_b_ion, find_terminal_y_ion
from .types import Gap
from .pivots import Pivot

#=============================================================================#

def find_boundary_peaks(
    spectrum: np.ndarray,
    pivot: Pivot,
    valid_terminal_residues,
):
    putative_b_ions = find_initial_b_ion(spectrum, pivot.outer_right(), len(spectrum), pivot.center())
    putative_y_ions = filter(lambda y: y[1] in valid_terminal_residues, find_terminal_y_ion(spectrum, pivot.outer_left()))
    return list(putative_b_ions), list(putative_y_ions)

#=============================================================================#

def create_augmented_spectrum(
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
    augments = []
    for val in [b_mz, y_mz]:
        reflected_val = reflect(val, center)
        for aug_peak in [val, reflected_val]:
            min_dif = np.inf
            for peak in subspectrum:  
                dif = abs(peak - aug_peak)
                if dif < min_dif:
                    min_dif = dif
            if min_dif > tolerance:
                augmented_spectrum.add(aug_peak)
                augments.append(aug_peak)
    augmented_spectrum = np.array(augmented_spectrum)

    # shift the pivot
    pivot_left = pivot.outer_left()
    pivot_left_mz = spectrum[pivot_left]
    new_left = len([val for val in augmented_spectrum if val < pivot_left_mz])
    offset = new_left - pivot_left
    
    return augmented_spectrum, offset

#=============================================================================#

def create_augmented_pivot(
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

def create_augmented_gaps(
    augmented_spectrum: np.ndarray,
    augmented_pivot: Pivot,
    offset: int,
    original_gaps: list[Gap]
):
    return list(_enumerate_augmented_gaps(
        len(augmented_spectrum), 
        offset, 
        augmented_pivot.negative_index_pairs(),
        original_gaps))

def _enumerate_augmented_gaps(
    size: int,
    offset: int,
    nonviable_gaps: list[Gap],
    original_gaps: list[Gap],
):
    for (original_i, original_j) in original_gaps:
        i = original_i + offset
        j = original_j + offset
        inbounds = (0 <= i <= j < size)
        viable = ((i, j) not in nonviable_gaps)
        if inbounds and viable:
            yield (i,j)
    
