from typing import Self, Iterator, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import functools as ft
import itertools as it

import numpy as np

from ..spectra.types import PeakList
from .solvers import ResidueStateSpace
from .pairs import PairedFragments

class AbstractPivot(ABC):
    """The top-level abstraction to pivot data. Declares get_pivot_point, get_score, and rescore."""

    @abstractmethod
    def get_pivot_point(self) -> float:
        """The center point of the pivot; an approximate point of mirror symmetry for peaks and fragment pairs."""

    @abstractmethod
    def get_score(self) -> float:
        """The score of the pivot. Typically an aggregate of component scores, and, after rescoring, a measure of the symmetry of the reflection parametized by the pivot point."""

    @abstractmethod
    def rescore(self, *args, **kwargs) -> Self:
        """Aggregate scoring data to produce a new Pivot."""

@dataclass
class OverlapPivot(AbstractPivot):
    """A pivot structure composed of two PairedFragments objects whose fragment mass intervals overlap."""
    left_pair: PairedFragments
    right_pair: PairedFragments
    residue_mass_delta: float

    @classmethod
    def from_pairs(cls,
        index_pair: tuple[int,int],
        paired_fragments: list[PairedFragments],
    ) -> Self:
        """Construct a FragmentPivot object from two PairedFragments objects whose fragment mass intervals intersect."""
        left_pair, right_pair = sorted(
            (paired_fragments[index_pair[0]], paired_fragments[index_pair[1]]),
            key = lambda pair: pair.left_fragment.fragment_mass)
        ll, lr = left_pair.fragment_masses()
        rl, rr = right_pair.fragment_masses()
        if ll < rl < lr < rr:
            return cls(
                left_pair = left_pair,
                right_pair = right_pair,
                residue_mass_delta = abs(pairs[0].residue.residue_mass - pairs[1].residue.residue_mass))
        else:
            raise ValueError(f"Fragment mass intervals [{ll}, {lr}], [{rl}, {rr}] do not intersect!")

@dataclass
class VirtualPivot(AbstractPivot):
    """A pivot structure that is not composed of distinct PairedFragments objects, but is rather discovered from statistical properties of the space of PairedFragments."""
    pivot_point: float        

class PivotSearchMode(Enum):
    OVERLAP = 1
    VIRTUAL = 2

@dataclass
class PivotSearchParams:
    tolerance: float
    mode: PivotSearchMode

def _find_overlap_pivots(
    pairs: list[tuple[float,float]],
) -> Iterator[tuple[int,int]]:
    n = len(pairs)
    for i in range(n):
        i_left, i_right = pairs[i]
        for j in range(i + 1, n):
            j_left, j_right = pairs[j]
            if j_left >= i_right:
                break
            else:
                yield (i,j)

def _find_virtual_pivots(
    pairs: list[tuple[float,float]],
    tolerance: float
) -> Iterator[float]:
    n = len(pairs)
    # reduce pairs to their average fragment mass
    midpoints = np.array(list(map(sum, pairs))) / 2.
    # construct potential pivots from midpoint half-sums.
    # for any given pair (x, y), their center h = (x + 2) / 2 is the point of mirror symmetry.
    midpoint_pivots = (midpoints.reshape(n,1) + midpoints.reshape(1,n)).flatten() / 2.
    # restrict the potential pivots to the most frequent values
    # the point(s) that are most frequent are likely the ones that induce the greatest mirror symmetry.
    bin_counts, bin_edges = np.histogram(
        midpoint_pivots,
        bins = int((midpoints.max() - midpoints.min()) / tolerance))
    bin_values = (bin_edges[:-1] + bin_edges[1:]) / 2
    frequencies = sorted(set(bin_counts))
    maximal_bin_values = bin_values[bin_counts > frequencies[int(len(frequencies) * 0.75)]]
    return maximal_bin_values

def _find_pivots(
    pairs: list[PairedFragments],
    search_strategy: PivotSearchParams,
) -> Iterator[AbstractPivot]:
    tolerance = search_strategy.tolerance
    pair_masses = [pair.fragment_masses() for pair in pairs]
    if search_strategy.mode is PivotSearchMode.OVERLAP:
        return map(
            ft.partial(
                OverlapPivot.from_pairs,
                paired_fragments = pairs),
            _find_overlap_pivots(pair_masses))
    elif search_strategy.mode is PivotSearchMode.VIRTUAL:
        return map(
            VirtualPivot,
            _find_disjoint_pivots(pair_masses, tolerance))
    else:
        raise ValueError(f"Search mode {search_strategy.mode} could not be recognized!")

def find_pivots(
    pairs: list[PairedFragments],
    search_strategy: PivotSearchParams,
    residue_state_space: ResidueStateSpace,
) -> Iterator[AbstractPivot]:
    return it.chain.from_iterable(map(
        ft.partial( # 3. within each bin, search for pivots.
            _find_pivots,
            peaks = peaks,
            search_strategy = search_strategy),
        map(
            lambda bin: sorted( # 2. sort bins by fragment mass interval.
                bin, 
                key = lambda x: x.fragment_masses()),
            util.binsort( # 1. bin the pairs by their amino acid.
                pairs,
                bins = len(residue_state_space.n_aminos()),
                key = lambda x: x.residue.amino_id))))
