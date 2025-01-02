from .pivot import Pivot, VirtualPivot
from .util import reflect, residue_lookup, ION_OFFSET_LOOKUP, TERMINAL_RESIDUES, GAP_TOLERANCE

import numpy as np
from sortedcontainers import SortedList

def _create_symmetric_augmented_spectra(
    spectrum: np.ndarray,
    pivot: Pivot,
    valid_terminal_residues = TERMINAL_RESIDUES
):
    # find and reflect boundary ions
    for (y_idx, y_res) in pivot.terminal_y(spectrum):
        if y_res in valid_terminal_residues:
            for (b_idx, b_res) in pivot.initial_b(spectrum):
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
                        if min_dif > GAP_TOLERANCE:
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
                elif type(pivot) == VirtualPivot:
                    indices = pivot.indices()
                    new_indices = (indices[0] + offset, indices[1] + offset)
                    index_shifted_pivot = VirtualPivot(new_indices, pivot.center())
                else:
                    raise ValueError(f"Unrecognized pivot type {type(pivot)}")
                
                yield b_res, y_res, augmented_spectrum, index_shifted_pivot

def create_symmetric_augmented_spectra(
    spectrum: np.ndarray,
    pivot: Pivot
):
    return list(_create_symmetric_augmented_spectra(spectrum, pivot))