import itertools as it

from ..spectra.types import PeakList
from .types import Pivot, PivotParams

def _find_overlapping_pivots(
    spectrum,
    gap_indices,
    tolerance,
) -> Iterator[Pivot]:
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

def _find_disjoint_pivots(
    spectrum,
    gap_indices,
    tolerance,
) -> Iterator[Pivot]:
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

def _find_adjacent_pivots(
    spectrum,
    tolerance,
) -> Iterator[Pivot]:
    n = len(spectrum)
    for i in range(n - 4):
        i2 = i + 1
        i_gap = spectrum[i2] - spectrum[i]
        j = i + 2
        j2 = j + 1
        j_gap = spectrum[j2] - spectrum[j]
        if abs(i_gap - j_gap) < tolerance:
            yield Pivot((spectrum[i], spectrum[i2]), (spectrum[j], spectrum[j2]), (i, i2), (j, j2))

def find_overlap_pivots(
    spectrum: PeakList,
    gap_indices: Iterator[tuple[int, int]],
    params: PivotParams, 
) -> Iterator[Pivot]:
    return _find_overlap_pivots(
        spectrum = spectrum,
        gap_indices = gap_indices,
        tolerance = params.tolerance)

def find_adjacent_pivots(
    spectrum: PeakList,
    params: PivotParams, 
) -> Iterator[Pivot]:
    return _find_adjacent_pivots(
        spectrum = spectrum,
        tolerance = params.tolerance)

def find_connected_pivots(
    spectrum: PeakList,
    gap_indices: Iterator[tuple[int, int]],
    params: PivotParams, 
) -> Iterator[Pivot]:
    return it.chain(
        find_adjacent_pivots(spectrum, params),
        find_overlap_pivots(spectrum, gap_indices, params))

def find_disjoint_pivots(
    spectrum: PeakList,
    gap_indices: Iterator[tuple[int, int]],
    params: PivotParams, 
) -> Iterator[Pivot]:
    return _find_disjoint_pivots(
        spectrum = spectrum,
        gap_indices = gap_indices,
        tolerance = params.tolerance)

def find_mirror_symmetries(params: PivotParams, *args, **kwargs) -> Iterator[Pivot]:
    pass