from dataclasses import dataclass
from copy import deepcopy

from sortedcontainers import SortedList
import numpy as np

from .types import Iterator
from .util import product, collapse_second_order_list, reflect, find_initial_b_ion
from .gaps import GapSearchParameters, GapResult, find_gaps
from .pivots import Pivot

#=============================================================================#
# helper functions for augmentations

def create_augmented_spectrum(
    spectrum: np.ndarray,
    center: float,
    terminal_y: tuple[int, float],
    initial_b: tuple[int, float],
    tolerance: float,
    padding: int = 3
):
    # unpack the boundaries
    y_idx, y_res = terminal_y
    b_idx, b_res = initial_b
    # define the subspectrum range from the ion boundaries
    n = len(spectrum)
    bound_lo = max(0, y_idx - padding)
    bound_hi = min(n - 1, b_idx + padding + 1)
    # augment the spectrum with mirrored boundary ions
    y_mz = spectrum[y_idx]
    b_mz = spectrum[b_idx]
    augmented_spectrum = SortedList(spectrum[bound_lo: bound_hi])
    #print("peak window", augmented_spectrum)
    boundary_values = []
    boundary_indices = []
    boundary_residues = []
    for (boundary_peak, boundary_residue) in [(b_mz, b_res), (y_mz, y_res)]:
        # construct the mirrored peak and try to locate it in the spectrum
        reflected_peak = reflect(boundary_peak, center)
        for query_peak in [boundary_peak, reflected_peak]:
            left_idx = augmented_spectrum.bisect_left(query_peak - tolerance)
            right_idx = augmented_spectrum.bisect_right(query_peak + tolerance)
            #print("query", query_peak)
            #print("bisect", (left_idx, right_idx))
            # determine whether the mirrored peak exists in the spectrum
            min_dif = np.inf
            closest_peak = (None, None)
            for (local_idx, peak) in enumerate(augmented_spectrum[left_idx: right_idx + 1]):
                dif = abs(peak - query_peak)
                if dif < min_dif:
                    min_dif = dif
                    idx = left_idx + local_idx
                    closest_peak = (idx, peak)

            boundary_residues.append(boundary_residue)
            if min_dif <= tolerance:
                # the reflected peak is in the spectrum; record it and its position.
                boundary_values.append(closest_peak[1])
            else:
                # the reflected peak is not in the spectrum; add it!
                augmented_spectrum.add(query_peak)
                boundary_values.append(query_peak)
    # re-index boundary peaks
    for peak in boundary_values:
        idx = augmented_spectrum.index(peak)
        #print("boundary:", peak, idx)
        boundary_indices.append(idx)

    return augmented_spectrum, boundary_values, boundary_indices, boundary_residues

def create_augmented_pivot(
    augmented_peak_list: SortedList,
    pivot: Pivot,
) -> Pivot:
    # re-index the peaks of the pivot
    peak_pairs = pivot.peak_pairs()
    shifted_index_pairs = [(augmented_peak_list.index(mz1), augmented_peak_list.index(mz2)) for (mz1, mz2) in peak_pairs]
    return Pivot(*peak_pairs, *shifted_index_pairs)

def create_augmented_gaps(
    augmented_spectrum: np.ndarray,
    gap_params: GapSearchParameters
):
    # collect all gaps identified over the augmented spectrum
    annotated_augmented_spectrum, augmented_gap_results = find_gaps(gap_params, augmented_spectrum)
    augmented_gaps = collapse_second_order_list(r.get_index_pairs() for r in augmented_gap_results)
    augmented_gaps = list(set(augmented_gaps))
    augmented_gaps.sort(key = lambda x: x[0] + x[1] / 1000)
    print(f"\taugmented_spectrum {augmented_spectrum}\naugmented_gaps {augmented_gaps}")
    return augmented_gaps

#=============================================================================#
# interface

class BoundedSpectrum:
    """Precompute the augmented peaks and gaps, given the input spectrum and a center of symmetry.
    Implements `get_augmented_spectrum` to retrieve the bounded spectrum with mirrored boundary ions;
    `get_augmented_gaps` to retrieve the gap set over the augmented spectrum, and `augment_pivot` to
    map a Pivot object to its counterpart with indices shifted to match those of the augmented spectrum."""

    def __init__(self,
        spectrum: np.ndarray,
        boundary_pair: tuple[tuple[int, float],tuple[int, float]],
        center: float,
        precision: int,
        gap_params: GapSearchParameters,
        padding: int = 3,
    ):
        self._gap_params = deepcopy(gap_params)
        self._gap_params.charges = np.array([])
        self._center = center
        self._precision = precision
        self._epsilon = 10**-precision
        terminal_y, initial_b = boundary_pair
        #print("terminal y", terminal_y)
        #print("initial b", initial_b)
        (self._augmented_peak_list, 
            self._augmented_boundary_values, 
            self._augmented_boundary_indices, 
            self._augmented_boundary_residues
        ) = create_augmented_spectrum(
            spectrum,
            center,
            terminal_y,
            initial_b,
            gap_params.tolerance,
            padding = padding,
        )
        
        self._augmented_spectrum = np.array(self._augmented_peak_list)

        self._augmented_gaps = create_augmented_gaps(
            self._augmented_spectrum,
            self._gap_params,
        )
    
    def get_augmented_spectrum(self):
        return self._augmented_spectrum
    
    def get_augmented_boundary_indices(self):
        return self._augmented_boundary_indices
    
    def get_augmented_boundary_values(self):
        return self._augmented_boundary_values

    def get_augmented_gaps(self):
        return self._augmented_gaps

    def augmentable(self, pivot: Pivot):
        aug_idx = self.get_augmented_boundary_indices()
        #print("aug indx", aug_idx)
        #print("pivot idx", pivot.indices())
        L = min(aug_idx) <= pivot.outer_left()
        R = max(aug_idx) >= pivot.outer_right()
        C = abs(pivot.center() - self._center) < self._epsilon
        #print(f"L {L} R {R} C {C}")
        return (L and R and C)

    def augment_pivot(self, pivot: Pivot):
        if not self.augmentable(pivot):
            raise ValueError("pivot is incompatible with the boundary!")
        else:
            augmented_pivot = create_augmented_pivot(self._augmented_peak_list, pivot)
            return augmented_pivot
    
    def construct_augmented_data(self, pivot: Pivot):
        return AugmentedData(
            spectrum = self.get_augmented_spectrum(),
            boundary = self.get_augmented_boundary_indices(),
            boundary_residues = self._augmented_boundary_residues,
            gaps = self.get_augmented_gaps(),
            pivot = self.augment_pivot(pivot),
        )

@dataclass
class AugmentedData:
    """Stores the augmented spectrum, boundary indices, augmented gaps, and augmented pivot as constructed by
    the combination a BoundedSpectrum and a Pivot."""

    spectrum: np.ndarray
    boundary: list[int]
    boundary_residues: list[str]
    gaps: list[tuple[int,int]]
    pivot: Pivot

    def __iter__(self):
        return iter((self.spectrum, self.boundary, self.gaps, self.pivot))

    def get_boundary_residues(self):
        return list(set(self.boundary_residues))

def create_augmented_data(
    spectrum: np.ndarray,
    terminal_y_ions: list[tuple[int, float]],
    initial_b_ions: list[tuple[int, float]],
    center: float,
    precision: int,
    pivots: list[Pivot],
    gap_params: GapSearchParameters,
    padding: int = 3,
) -> Iterator[AugmentedData]:
    """"""
    for boundary_pair in product(terminal_y_ions, initial_b_ions):
        #print("boundary pair", boundary_pair)
        bounded_spectrum = BoundedSpectrum(
            spectrum,
            boundary_pair,
            center,
            precision,
            gap_params,
            padding = padding,
        )
        for pivot in pivots:
            if bounded_spectrum.augmentable(pivot):
                yield bounded_spectrum.construct_augmented_data(pivot)