from typing import Self, Iterator, Callable, Any, Iterable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from math import log
import functools as ft
import itertools as it

import numpy as np
import numba

from .. import util
from ..spectra.types import PeakList
from .pairs import PairedFragments

class AbstractPivot(ABC):
    """The top-level abstraction to pivot data. Declares get_pivot_point, get_score, and rescore."""

    @abstractmethod
    def get_pivot_point(self) -> float:
        """The center point of the pivot; an approximate point of mirror symmetry for peaks and fragment pairs."""

    @abstractmethod
    def get_pivot_indices(self) -> Any:
        """Return either a sorted 4-tuple of integers, in the case of observed pivots, or None, in the case of virtual pivots."""

    @abstractmethod
    def get_score(self) -> float:
        """Return a heuristic for the quality of the pivot. Lower is better."""

    @abstractmethod
    def set_score(self, *args, **kwargs) -> Self:
        """Returns a pivot with the new score value; assume that this mutates the old pivot."""

@dataclass
class OverlapPivot(AbstractPivot):
    """A pivot structure composed of two fragment mass intervals which overlap."""
    indices: tuple[int,int,int,int]
    masses: tuple[float,float,float,float]
    score: float = 0

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
                    indices = indices,
                    masses = masses)
        raise ValueError(f"pivot could not be formed on indices {indices}")

    @classmethod
    def from_dict(cls,
        data: dict[str,Any],
    ) -> Self:
        return cls(
            indices = tuple(data["indices"]),
            masses = tuple(data["masses"]),
            score = data["score"])
    
    def get_pivot_point(self) -> float:
        return sum(self.masses) / 4

    def get_pivot_indices(self) -> tuple[int,int,int,int]:
        """Return the indices of the left and right pairs comprising the pivot."""
        return self.indices        

    def get_score(self) -> float:
        return self.score

    def set_score(self, score) -> Self:
        self.score = score
        return self

@dataclass
class VirtualPivot(AbstractPivot):
    """A pivot structure that is not composed of distinct PairedFragments objects, but is rather discovered from statistical properties of the space of PairedFragments."""
    pivot_point: float
    frequency: int
    score: float = 0

    @classmethod
    def from_dict(cls,
        data: dict[str,Any],
    ) -> Self:
        return cls(
            pivot_point = data["pivot_point"],
            frequency = data["frequency"],
            score = data["score"])
    
    def get_pivot_point(self) -> float:
        return self.pivot_point

    def get_pivot_indices(self) -> None:
        """Virtual pivots do not have indices, so this method returns None."""
        return None

    def get_score(self) -> float:
        return self.score

    def set_score(self, symmetry_score: int) -> Self:
        self.score = score
        return self

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
    return [VirtualPivot(pivot_point, frequency) for (pivot_point, frequency) in zip(maximal_bin_values, maximal_bin_counts / sum(maximal_bin_counts))]

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
    return _find_virtual_pivots(midpoints, bin_width)

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
) -> Iterator[OverlapPivot]:
    spectrum = peaks.mz
    pairs = np.unique([p.peak_indices() for p in pairs], axis = 0)
    return [OverlapPivot.from_indices(ind, spectrum)
        for ind in _find_overlap_pivots(spectrum, pairs, tolerance)]
