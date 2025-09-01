from typing import Self, Iterator, Callable, Any, Iterable
from abc import ABC, abstractmethod
from enum import Enum
from math import log
import functools as ft
import itertools as it
import dataclasses

import numpy as np
import numba

from .. import util
from ..spectra.types import PeakList
from .pairs import PairedFragments

@dataclasses.dataclass(slots=True)
class Pivot():
    pivot_point: float
    score: float
    
    def rescore(self,
        new_score: float
    ) -> Self:
        data = dataclasses.asdict(self)
        data["score"] = new_score
        return type(self)(**data)

@dataclasses.dataclass(slots=True)
class OverlapPivot(Pivot):
    """A pivot structure composed of two fragment mass intervals which overlap."""
    pivot_point: float
    score: float
    indices: tuple[int,int,int,int]
    masses: tuple[float,float,float,float]
    
    @classmethod
    def from_indices(cls,
        indices: tuple[int,int,int,int],
        fragment_masses: Iterable[float],
    ) -> Self:
        """Construct an OverlapPivot given its four indices and the fragment mass spectrum."""
        if list(indices) == sorted(indices):
            masses = tuple([fragment_masses[i] for i in indices])
            if list(masses) == sorted(masses):
                return OverlapPivot(
                    pivot_point = np.mean(masses),
                    indices = indices,
                    masses = masses,
                    score = 0.)
        raise ValueError(f"pivot could not be formed on indices {indices}")

    @classmethod
    def from_dict(cls,
        data: dict[str,Any],
    ) -> Self:
        masses = tuple(data["masses"])
        return cls(
            pivot_point = np.mean(masses),
            indices = tuple(data["indices"]),
            masses = masses,
            score = data["score"])

@dataclasses.dataclass(slots=True)
class VirtualPivot(Pivot):
    """A pivot structure that is not composed of distinct PairedFragments objects, but is rather discovered from statistical properties of the space of PairedFragments."""
    pivot_point: float
    score: float
    frequency: int

    @classmethod
    def from_dict(cls,
        data: dict[str,Any],
    ) -> Self:
        return cls(
            pivot_point = data["pivot_point"],
            frequency = data["frequency"],
            score = data["score"])

def _find_virtual_pivots(
    midpoints: Iterable[float],
    bin_width: float,
) -> Iterator[tuple[float,float]]:
    # restrict the potential pivots to the most frequent values
    # the point(s) that are most frequent are likely the ones that induce the greatest mirror symmetry.
    num_bins = int((midpoints.max() - midpoints.min()) / bin_width)
    if num_bins == 0:
        return []
    # print(num_bins)
    bin_counts, bin_edges = np.histogram(
        midpoints,
        bins = num_bins)
    # print("bin edges", bin_edges)
    bin_values = (bin_edges[:-1] + bin_edges[1:]) / 2
    # print("bin values", bin_values)
    frequencies = sorted(set(bin_counts))
    mask = bin_counts > frequencies[int(len(frequencies) * 0.75)]
    maximal_bin_values = bin_values[mask]
    maximal_bin_counts = bin_counts[mask]
    # return the bin values and the normalized bin counts
    return zip(maximal_bin_values, maximal_bin_counts / sum(maximal_bin_counts))

def _construct_midpoints(
    pairs: list[PairedFragments],
) -> Iterable[float]:
    n = len(pairs)
    centers = np.array([np.sum(p.fragment_masses()) / 2 for p in pairs])
    midpoints = (centers.reshape(n, 1) + centers.reshape(1, n))[np.triu_indices(n)] / 2
    return midpoints

def find_virtual_pivots(
    pairs: list[PairedFragments],
    bin_width: float,
) -> Iterator[VirtualPivot]:
    # bin the pairs by their amino acid.
    _, bins = util.binsort(
        pairs,
        key = lambda x: x.residue.amino_id)
    # within each bin, construct midpoints, and concatenate them into one list.
    midpoints = np.concatenate([_construct_midpoints(pair_bin) for pair_bin in bins])
    # find the most common midpoints and cast them as virtual pivots
    pivot_params = _find_virtual_pivots(midpoints, bin_width)
    return [VirtualPivot(
            pivot_point = pivot_point, 
            score = 0.,
            frequency = frequency) 
        for (pivot_point, frequency) in pivot_params]

@numba.jit
def _find_overlap_pivots(
    spectrum: Iterable[float],
    pairs: Iterable[tuple[int,int]],
    tolerance: float,
) -> Iterator[tuple[int,int]]:
    n = len(spectrum)
    for (i, i2) in pairs:
        mass_i = spectrum[i2] - spectrum[i]
        for j in range(i + 1, i2):
            for j2 in range(i2 + 1, n):
                mass_j = spectrum[j2] - spectrum[j]
                dif = abs(mass_i - mass_j)
                if dif < tolerance:
                    yield (i,j,i2,j2)
                elif mass_j > mass_i:
                    break

def find_overlap_pivots(
    peaks: PeakList,
    pairs: list[PairedFragments],
    tolerance: float,
) -> list[OverlapPivot]:
    spectrum = peaks.mz
    pairs = np.unique([p.peak_indices() for p in pairs], axis = 0)
    pivot_ind = _find_overlap_pivots(spectrum, pairs, tolerance)
    return [OverlapPivot.from_indices(ind, spectrum) for ind in pivot_ind]

def _find_pivots(
    peaks: PeakList,
    pairs: list[PairedFragments],
    tolerance: float,
    find_overlap = True,
    find_virtual = False,
) -> list[Pivot]:
    pivots = []
    if find_overlap:
        pivots.extend(find_overlap_pivots(
            peaks,
            pairs,
            tolerance))
    if find_virtual:
        pivots.extend(find_virtual_pivots(
            pairs,
            tolerance / 2))
    return pivots

def _bin_pivots(
    pivots: list[Pivot],
    precision: int,
) -> tuple[np.array,np.array]:
    return np.unique_inverse([np.round(p.pivot_point, precision) for p in pivots])

def _screen_pivots(
    peaks: np.array,
    pivots: np.array,
    points: np.array,
    idx_arr: np.array,
    tolerance: float,
) -> Iterable[Pivot]:
    # calibrate tolerance to the minimum pairwise distance between peaks
    tolerance = min(tolerance, min([y - x for (x, y) in it.pairwise(peaks)]) / 2)
    
    # mask points by their symmetry score
    point_scores = np.array([util.measure_mirror_symmetry(peaks, pt, tolerance) for pt in points])
    points_mask = point_scores > (point_scores.max() / 2)
    pivots_mask = points_mask[idx_arr]
    
    # apply mask to pivots/points/idx_arr and cast to python types.
    new_pivots = [p.rescore(s) for (p, s) in zip(
        pivots[pivots_mask],
        point_scores[idx_arr][pivots_mask].tolist())]
    new_points = points[points_mask].tolist()

    # reindex the pivot_idx -> point_idx map.
    old_pivot_idx = np.arange(len(pivots))[pivots_mask]
    point_reindexer = np.arange(len(points))
    new_point_idx = np.arange(len(new_points))
    point_reindexer[points_mask] = new_point_idx
   ## one-liner - test_annotation in 44s
   ## new_idx_arr = point_reindexer[idx_arr[old_pivot_idx[pivots_mask]]].tolist()
   ## comprehension - test_annotation in 44s
    new_idx_arr = np.zeros_like(new_pivots)
    for new_pvt_idx, old_pvt_idx in enumerate(old_pivot_idx):
        old_pt_idx = idx_arr[old_pvt_idx]
        new_pt_idx = point_reindexer[old_pt_idx]
        new_idx_arr[new_pvt_idx] = new_pt_idx.tolist()
    new_idx_arr = new_idx_arr.tolist()

    return new_pivots, new_points, new_idx_arr
    
def find_pivots(
    peaks: PeakList,
    pairs: list[PairedFragments],
    comparison_tolerance: float,
    symmetry_tolerance: float,
    find_overlap = True,
    find_virtual = False,
) -> tuple[list[Pivot],list[float],list[int]]:
    if not(find_overlap or find_virtual):
        raise ValueError("both find_overlap and find_virtual are False. there are no other ways to find pivots.")
    # generate pivots
    pivots = _find_pivots(
        peaks,
        pairs,
        comparison_tolerance,
        find_overlap,
        find_virtual)

    # bin by pivot point
    points, idx_arr = _bin_pivots(
        pivots,
        precision=2)

    # screen pivot points
    pivots, points, idx_arr = _screen_pivots(
        peaks.mz,
        np.array(pivots),
        points,
        idx_arr,
        symmetry_tolerance)

    return pivots, points, idx_arr
