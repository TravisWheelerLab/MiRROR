from .pivot import Pivot, VirtualPivot
from .util import reflect, residue_lookup, ION_OFFSET_LOOKUP

import numpy as np

def create_symmetric_boundary(
    spectrum,
    pivot: Pivot
):
    # find and reflect boundary ions
    init_b, b_res = pivot.initial_b_ion(spectrum)
    term_y, y_res = pivot.terminal_y_ion(spectrum)
    center = pivot.center()
    reflected_b_mz = reflect(spectrum[init_b], center)
    reflected_y_mz = reflect(spectrum[term_y], center)
    # TODO - create the symmetric spectrum more efficiently.
    spectrum_list = list(spectrum)
    boundary_symmetric_spectrum = np.array(sorted(set(list(map(lambda x: round(x, 6), spectrum_list + [reflected_b_mz, reflected_y_mz])))))
    # shift the pivot
    n_peaks = len(spectrum)
    n_sym_peaks = len(boundary_symmetric_spectrum)
    if type(pivot) == Pivot:
        index_pairs = pivot.index_pairs()
        new_index_pairs = [(i + 1, j + 1) for (i, j) in index_pairs]
        index_shifted_pivot = Pivot(*pivot.peak_pairs(), *new_index_pairs)
    elif type(pivot) == VirtualPivot:
        indices = pivot.indices()
        new_indices = (indices[0] + 1, indices[1] + 1)
        index_shifted_pivot = VirtualPivot(new_indices, pivot.center())
    else:
        raise ValueError(f"Unrecognized pivot type {type(pivot)}")
    return boundary_symmetric_spectrum, index_shifted_pivot