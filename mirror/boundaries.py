#

from sortedcontainers import SortedList
import numpy as np

from .util import reflect, residue_lookup, find_initial_b_ion, find_terminal_y_ion
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
):
    # reflect the boundaries
    center = pivot.center()
    b_mz = spectrum[b_idx]
    y_mz = spectrum[y_idx]
    
    # augment the spectrum
    n = len(spectrum)
    subspectrum = spectrum[max(0,y_idx - 3):min(n - 1,b_idx + 4)]
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
    if type(pivot) == Pivot:
        index_pairs = pivot.index_pairs()
        new_index_pairs = [(i + offset, j + offset) for (i, j) in index_pairs]
        new_peak_pairs = [(augmented_spectrum[i], augmented_spectrum[j]) for (i,j) in new_index_pairs]
        index_shifted_pivot = Pivot(*new_peak_pairs, *new_index_pairs)
        assert index_shifted_pivot.peaks() == pivot.peaks()
    #elif type(pivot) == VirtualPivot:
    #    indices = pivot.indices()
    #    new_indices = (indices[0] + offset, indices[1] + offset)
    #    index_shifted_pivot = VirtualPivot(new_indices, pivot.center())
    else:
        raise ValueError(f"Unrecognized pivot type {type(pivot)}")
    
    return augmented_spectrum, index_shifted_pivot

#=============================================================================#

def find_augmented_gaps(
    augmented_spectrum,
    augmented_pivot,
    target_groups,
    tolerance,
):
    # TODO: this is just a wrapper for the method in gaps with a filter tacked on which removes gaps that intersect the pivot
    pass