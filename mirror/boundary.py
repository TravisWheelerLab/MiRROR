from .pivot import Pivot, VirtualPivot
from .util import reflect, residue_lookup, ION_OFFSET_LOOKUP, TERMINAL_RESIDUES

import numpy as np
from sortedcontainers import SortedList

def _create_symmetric_augmented_spectra(
    spectrum: np.ndarray,
    pivot: Pivot,
    valid_terminal_residues = TERMINAL_RESIDUES
):
    # find and reflect boundary ions
    for (y_idx, y_res) in pivot.terminal_y(spectrum):
        if y_res in TERMINAL_RESIDUES:
            for (b_idx, b_res) in pivot.initial_b(spectrum):
                # reflect the boundaries
                center = pivot.center()
                b_mz = spectrum[b_idx]
                y_mz = spectrum[y_idx]
                
                # augment the spectrum
                subspectrum = SortedList(spectrum[y_idx:b_idx + 1])
                augments = []
                for val in [b_mz, y_mz]:
                    reflected_val = reflect(val, center)
                    if not(val in subspectrum):
                        subspectrum.add(val)
                        augments.append(val)
                    if not(reflected_val in subspectrum):
                        subspectrum.add(reflected_val)
                        augments.append(reflected_val)
                augmented_spectrum = np.array(subspectrum)

                # shift the pivot
                pivot_left = pivot.peaks()[0]
                offset = len([val for val in augments if val < pivot_left])
                if type(pivot) == Pivot:
                    index_pairs = pivot.index_pairs()
                    new_index_pairs = [(i + offset, j + offset) for (i, j) in index_pairs]
                    index_shifted_pivot = Pivot(*pivot.peak_pairs(), *new_index_pairs)
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