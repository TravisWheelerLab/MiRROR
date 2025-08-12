from typing import Self, Iterator, Callable, Any, Iterable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from math import log
import functools as ft
import itertools as it

import numpy as np

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
    """A pivot structure composed of two PairedFragments objects whose fragment mass intervals overlap."""
    indices: tuple[int,int,int,int]
    left_pairs: list[PairedFragments]
    right_pairs: list[PairedFragments]
    pivot_point: float
    score: float = 0

    @classmethod
    def from_pairs(cls,
        bin_idx_pair: tuple[int,int],
        paired_fragments_bins: list[list[PairedFragments]],
    ) -> Self:
        """Construct a FragmentPivot object from two PairedFragments objects whose fragment mass intervals intersect."""
        left_bin_idx, right_bin_idx = bin_idx_pair
        left_pairs = paired_fragments_bins[left_bin_idx]
        left_idx_pair = left_pairs[0].peak_indices()
        right_pairs = paired_fragments_bins[right_bin_idx]
        right_idx_pair = right_pairs[0].peak_indices()
        if (left_idx_pair[0] < right_idx_pair[0] < left_idx_pair[1] < right_idx_pair[1]):
            return cls(
                indices = (left_idx_pair[0], right_idx_pair[0], left_idx_pair[1], right_idx_pair[1]),
                left_pairs = left_pairs,
                right_pairs = right_pairs,
                pivot_point = (sum(left_pairs[0].fragment_masses()) + sum(right_pairs[0].fragment_masses())) / 4)

    def get_pivot_point(self) -> float:
        return self.pivot_point

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
    upper_quartile = int(len(frequencies) * 0.75)
    uq_mask = bin_counts > frequencies[upper_quartile]
    maximal_bin_values = bin_values[uq_mask]
    maximal_bin_counts = bin_counts[uq_mask]
    # return the bin values and the normalized bin counts
    return [VirtualPivot(pivot_point, frequency) for (pivot_point, frequency) in zip(maximal_bin_values, maximal_bin_counts / sum(maximal_bin_counts))]

def _construct_midpoints(
    pairs: list[PairedFragments],
) -> Iterable[float]:
    n = len(pairs)
    pair_centers = np.array([sum(p.fragment_masses()) / 2 for p in pairs])
    return (pair_centers.reshape(n,1) + pair_centers.reshape(1,n)).flatten() / 2.

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

def _find_overlap_pivots(
    pairs: list[PairedFragments],
) -> Iterator[OverlapPivot]:
    # bin the pairs by their peak indices
    pair_indices, bins = util.binsort(
        pairs,
        key = lambda x: str(x.peak_indices()))
    # reorder bins by pair_indices
    pair_ind_order = np.argsort(pair_indices)
    pair_indices = pair_indices[pair_ind_order]
    bins = [bins[i] for i in pair_ind_order]
    # construct pivots
    n = len(pair_indices)
    for i in range(n):
        left_i, right_i = bins[i][0].fragment_masses()
        for j in range(i + 1, n):
            left_j, right_j = bins[j][0].fragment_masses()
            if left_j >= right_i:
                break
            elif (left_i < left_j < right_i < right_j):
                yield OverlapPivot.from_pairs(
                    (i,j),
                    bins)

def find_overlap_pivots(
    pairs: list[PairedFragments],
) -> Iterator[OverlapPivot]:
    # bin the pairs by their amino acid.
    _, bins = util.binsort(
        pairs,
        key = lambda x: x.residue.amino_id)
    # within each bin, search for pivots. chain results into one iterator
    return it.chain.from_iterable(
        _find_overlap_pivots(pair_bin) for pair_bin in bins)
