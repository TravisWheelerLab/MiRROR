from typing import Iterable
from dataclasses import dataclass
from multiprocessing import Pool
import functools as ft
import itertools as it

import numpy as np

from . import util
from .spectra.types import PeakList, BenchmarkPeakList
from .fragments import FragmentState, FragmentStateSpace, ResidueState, ResidueStateSpace, PairedFragments, AbstractPivot, BoundaryFragment, ReflectedBoundaryFragment, find_pairs, find_overlap_pivots, find_virtual_pivots, find_left_boundaries, find_right_boundaries, rescore_pivots

@dataclass
class AnnotationParams:
    fragment_search_tolerance: float
    fragment_state_space: FragmentStateSpace
    residue_state_space: ResidueStateSpace
    pivot_symmetry_tolerance: float
    pivot_score_threshold_factor: float

class AnnotationResult:
    def __init__(self,
        pairs: list[PairedFragments],
        left_boundaries: list[BoundaryFragment],
        right_boundaries: list[list[ReflectedBoundaryFragment]],
        pivots: list[AbstractPivot],
        pivot_index: list[int],
    ):
        self._n_pairs = len(pairs)
        self._pairs = pairs

        self._n_left_boundaries = len(left_boundaries)
        self._left_boundaries = left_boundaries

        self._n_pivots = len(pivots)
        self._n_right_boundaries = len(right_boundaries)
        assert self._n_pivots == len(pivot_index)
        assert self._n_pivots <= self._n_right_boundaries
        self._pivots = pivots
        self._pivot_index = pivot_index
        self._right_boundaries = right_boundaries

    def get_pairs(self) -> list[PairedFragments]:
        return self._pairs

    def get_left_boundaries(self) -> list[BoundaryFragment]:
        return self._left_boundaries

    def get_pivots(self) -> list[AbstractPivot]:
        return self._pivots

    def get_right_boundaries(self, pivot_id: int) -> list[ReflectedBoundaryFragment]:
        return self._right_boundaries[self._pivot_index[pivot_id]]

def reindex_by_fragment_masses(
    pairs: list[PairedFragments],
    fragment_state_space: FragmentStateSpace,
    precision: int = 4,
) -> tuple[list[PairedFragments],Iterable[float]]:
    # project to fragment masses
    fragment_masses = list(it.chain.from_iterable(
        np.round(p.fragment_masses(), precision) for p in pairs))
    fragment_masses, reindexer = np.unique_inverse(fragment_masses)
    # reindex the pairs
    reindexed_pairs = [PairedFragments.from_solution((
        FragmentState.from_index(
            peak_idx = reindexer[2 * i],
            fragment_mass = pair.left_fragment.fragment_mass,
            loss_id = pair.left_fragment.loss_id,
            charge = 1, # the peak idx now points to the decharged fragment mass in the new fragment_masses array
            state_space = fragment_state_space),
        FragmentState.from_index(
            peak_idx = reindexer[(2 * i) + 1],
            fragment_mass = pair.right_fragment.fragment_mass,
            loss_id = pair.right_fragment.loss_id,
            charge = 1, # the peak idx now points to the decharged fragment mass in the new fragment_masses array
            state_space = fragment_state_space),
        pair.residue)) for (i, pair) in enumerate(pairs)]
    return (
        reindexed_pairs,
        fragment_masses)

def annotate(
    peaks: PeakList,
    params: AnnotationParams,
) -> AnnotationResult:
    # find pairs of peaks whose m/z difference is solvable as a (FragmentState,FragmentState,ResidueStateSpace) tuple.
    peak_pairs = list(find_pairs(
        peaks = peaks,
        tolerance = params.fragment_search_tolerance,
        residue_state_space = params.residue_state_space,
        fragment_state_space = params.fragment_state_space))
    # # create fragment masses and reindexed pairs
    # fragment_pairs, fragment_masses = reindex_into_fragment_masses(
    #     pairs = peak_pairs,
    #     fragment_state_space = params.fragment_state_space)
    # find structures of pairs that reflect about a common point of symmetry.
    overlap_pivots = list(find_overlap_pivots(pairs = fragment_pairs))
    virtual_pivots = list(find_virtual_pivots(
        pairs = fragment_pairs,
        bin_width = params.fragment_search_tolerance))
    pivots = overlap_pivots + virtual_pivots
    # find boundary peaks. 
    ## BoundaryPeaks have m/z that is within a shift transformation of a (FragmentState,ResidueState) solution.
    left_boundaries = find_left_boundaries(
        peaks = peaks,
        tolerance = params.fragment_search_tolerance,
        residue_state_space = params.residue_state_space,
        fragment_state_space = params.fragment_state_space)
    ## ReflectedBoundaryPeaks have m/z that is within a reflection and shift of a (FragmentState,ResidueState) solution.
    ## the reflection is parametized by a pivot, so right_boundaries is a nested list that shares its index with pivots.
    right_boundaries = find_right_boundaries(
        pivots = pivots,
        peaks = peaks,
        match_threshold = params.fragment_search_tolerance,
        residue_state_space = params.residue_state_space,
        fragment_state_space = params.fragment_state_space)
    # score and filter pivots according to their symmetry and right boundary quality.
    expected_num_symmetries = 2 # need a better way to set this
    pivots, pivot_index = rescore_pivots(
        pivots = pivots,
        left_boundaries = left_boundaries,
        right_boundaries = right_boundaries,
        symmetry_threshold = params.pivot_symmetry_tolerance,
        score_threshold = expected_num_symmetries * params.pivot_score_threshold_factor)
    # wrap everything up as an AnnotationResult object
    return AnnotationResult(
        pairs = peak_pairs,
        left_boundaries = left_boundaries,
        right_boundaries = right_boundaries,
        pivots = pivots,
        pivot_index = pivot_index)
