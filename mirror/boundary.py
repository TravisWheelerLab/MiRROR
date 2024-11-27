from .pivot import Pivot, VirtualPivot
from .util import reflect, residue_lookup, ION_OFFSET_LOOKUP

import numpy as np

def find_initial_b_ion(
    spectrum, 
    pivot: Pivot
):
    center = pivot.center()
    # starting at the pivot, scan the upper half of the spectrum
    for i in range(pivot.outer_right(), len(spectrum)):
        corrected_mass = reflect(spectrum[i], center) - ION_OFFSET_LOOKUP['b']
        residue = residue_lookup(corrected_mass)
        if residue != 'X':
            return i, residue

def find_terminal_y_ion(
    spectrum, 
    pivot: Pivot
):
    # starting at the pivot, scan the lower half of the spectrum
    for i in range(pivot.outer_left(), -1, -1):
        corrected_mass = spectrum[i] - ION_OFFSET_LOOKUP['y']
        residue = residue_lookup(corrected_mass)
        if residue != 'X':
            return i, residue

def create_symmetric_boundary(
    spectrum,
    pivot: Pivot
):
    init_b, b_res = find_initial_b_ion(spectrum, pivot)
    term_y, y_res = find_terminal_y_ion(spectrum, pivot)
    center = pivot.center()
    reflected_b_mz = reflect(spectrum[init_b], center)
    reflected_y_mz = reflect(spectrum[term_y], center)
    spectrum_list = list(spectrum)
    n_peaks = len(spectrum_list)
    boundary_symmetric_spectrum = np.array(sorted(set(list(map(lambda x: round(x, 6), spectrum_list + [reflected_b_mz, reflected_y_mz])))))
    n = len(boundary_symmetric_spectrum)
    for i in range(n - 1):
        L = boundary_symmetric_spectrum[i]
        R = boundary_symmetric_spectrum[i + 1]
        if L < center < R:
            pivot_data = boundary_symmetric_spectrum[i - 1: i + 3]
            pivot_indices = (i - 1, i, i + 1, i + 2)
    boundary_symmetric_pivot = VirtualPivot(pivot_data, pivot_indices)
    return boundary_symmetric_spectrum, boundary_symmetric_pivot