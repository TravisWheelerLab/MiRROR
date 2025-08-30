from typing import Iterable, Self
from dataclasses import dataclass, asdict
from multiprocessing import Pool
from time import time
import itertools as it
import functools as ft
import itertools as itA
import json

import numpy as np

from . import util
from .spectra.types import PeakList, BenchmarkPeakList
from .fragments import FragmentState, FragmentStateSpace, ResidueState, ResidueStateSpace, PairedFragments, Pivot, OverlapPivot, VirtualPivot, BoundaryFragment, ReflectedBoundaryFragment, find_pairs, find_pivots, find_left_boundaries, find_right_boundaries, rescore_pivots

@dataclass
class AnnotationParams:
    fragment_search_tolerance: float
    fragment_space: FragmentStateSpace
    extremal_fragment_space: FragmentStateSpace
    residue_space: ResidueStateSpace
    pivot_symmetry_tolerance: float
    pivot_score_threshold_factor: float

@dataclass
class AnnotationResult:
    pairs: list[PairedFragments]
    left_boundaries: list[BoundaryFragment]
    right_boundaries: list[list[ReflectedBoundaryFragment]]
    pivots: list[Pivot]
    pivot_index: list[int]
    _profile: dict[str,float] = None

    def __post_init__(self):
        assert len(self.pivots) == len(self.pivot_index)
        assert len(set(self.pivot_index)) == len(self.right_boundaries)

    @classmethod
    def read(cls, filepath: str) -> Self:
        with open(filepath, 'r') as f:
            anno = json.load(f)
            return cls(
                pairs = [
                    PairedFragments.from_dict(x) for x in anno['pairs']],
                left_boundaries = [
                    BoundaryFragment.from_dict(x) for x in anno['left_boundaries']],
                right_boundaries = [
                    [ReflectedBoundaryFragment.from_dict(x) for x in X] for X in anno['right_boundaries']],
                pivots = [
                    OverlapPivot.from_dict(x) if ('indices' in x) else VirtualPivot.from_dict(x) for x in anno['pivots']],
                pivot_index = anno['pivot_index'],
                _profile = anno['_profile'])
        
    def write(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent = 4)

    def get_pairs(self) -> list[PairedFragments]:
        return self._pairs

    def get_left_boundaries(self) -> list[BoundaryFragment]:
        return self._left_boundaries

    def get_pivots(self) -> list[Pivot]:
        return self._pivots

    def get_right_boundaries(self, pivot_id: int) -> list[ReflectedBoundaryFragment]:
        return self._right_boundaries[self._pivot_index[pivot_id]]

def _localize_into_ideal_masses(
    pairs: list[PairedFragments],
    fragment_space: FragmentStateSpace,
    precision: int = 4,
) -> tuple[list[PairedFragments],Iterable[float]]:
    # project to fragment masses
    fragment_masses = list(it.chain.from_iterable(
        np.round(
            p.fragment_masses(), 
            precision) 
        for p in pairs))
    fragment_masses, reindexer = np.unique_inverse(fragment_masses)
    fragment_masses = fragment_masses.tolist()
    reindexer = reindexer.tolist()
    
    # reindex the pairs
    reindexed_pairs = [PairedFragments.from_solution((
        FragmentState.from_index(
            peak_idx = reindexer[2 * i],
            fragment_mass = pair.left_fragment.fragment_mass,
            loss_id = pair.left_fragment.loss_id,
            charge = 1, # the peak idx now points to the decharged fragment mass in the new fragment_masses array
            state_space = fragment_space),
        FragmentState.from_index(
            peak_idx = reindexer[(2 * i) + 1],
            fragment_mass = pair.right_fragment.fragment_mass,
            loss_id = pair.right_fragment.loss_id,
            charge = 1, # the peak idx now points to the decharged fragment mass in the new fragment_masses array
            state_space = fragment_space),
        pair.residue)) for (i, pair) in enumerate(pairs)]
    return (
        reindexed_pairs,
        PeakList(fragment_masses,np.ones_like(fragment_masses)))

def annotate(
    peaks: PeakList,
    params: AnnotationParams,
) -> AnnotationResult:
    # find pairs of peaks whose m/z difference is solvable as a (FragmentState,FragmentState,ResidueStateSpace) tuple.
    pair_time = time()
    fragment_pairs = list(find_pairs(
        peaks = peaks,
        tolerance = params.fragment_search_tolerance,
        residue_space = params.residue_space,
        fragment_space = params.fragment_space))
    pair_time = time() - pair_time
    
    # create fragment masses and reindexed pairs
    localize_time = time()
    n_peaks = len(peaks)
    fragment_pairs, peaks = _localize_into_ideal_masses(
        pairs = fragment_pairs,
        fragment_space = params.fragment_space)
    localize_time = time() - localize_time

    # find structures of pairs that reflect about a common point of symmetry.
    pivot_time = time()
    pivots, pivot_points, indexer = list(find_pivots(
        peaks = peaks,
        pairs = fragment_pairs,
        comparison_tolerance = 2 * params.fragment_search_tolerance,
        symmetry_tolerance = params.pivot_symmetry_tolerance))
    pivot_time = time() - pivot_time

    # BoundaryPeaks have m/z that is within a shift transformation of a (FragmentState,ResidueState) solution.
    lb_time = time()
    left_boundaries = list(find_left_boundaries(
        peaks = peaks,
        tolerance = 0.1,#params.fragment_search_tolerance,
        residue_space = params.residue_space,
        fragment_space = params.extremal_fragment_space))
    lb_time = time() - lb_time

    # ReflectedBoundaryPeaks have m/z that is within a reflection and shift of a (FragmentState,ResidueState) solution. the reflection is parametized by a pivot, so right_boundaries is a nested list that shares its index with pivots.
    rb_time = time()
    right_boundaries = list(find_right_boundaries(
        pivot_points = pivot_points,
        peaks = peaks,
        tolerance = params.fragment_search_tolerance,
        residue_space = params.residue_space,
        fragment_space = params.extremal_fragment_space))
    rb_time = time() - rb_time

    # score and filter pivots according to their symmetry and right boundary quality.
    rescore_time = time()
    # expected_num_symmetries = 2 # need a better way to set this
    # rescored_pivots, pivot_index = rescore_pivots(
    #     pivots = pivots,
    #     left_boundaries = left_boundaries,
    #     right_boundaries = right_boundaries,
    #     peaks = peaks,
    #     symmetry_tolerance = params.pivot_symmetry_tolerance,
    #     score_threshold = expected_num_symmetries * params.pivot_score_threshold_factor)
    rescore_time = time() - rescore_time
    
    # wrap everything up as an AnnotationResult object
    return AnnotationResult(
        pairs = fragment_pairs,
        left_boundaries = left_boundaries,
        right_boundaries = right_boundaries,
        pivots = pivots,#rescored_pivots
        pivot_index = indexer,#pivot_index,
        _profile = {"pair": pair_time, "localize": localize_time, "pivot": pivot_time, "lb": lb_time, "rb": rb_time, "rescore": rescore_time})
