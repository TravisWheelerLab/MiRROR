from dataclasses import dataclass
from multiprocessing import Pool
import functools as ft
#import itertool as it

from .fragments import ResidueState, FragmentState, FragmentStateSpace
from .pairs import PairedFragment
from .pivots import PivotSearchParams, Pivot
from .boundaries import LeftBoundaryFragment, RightBoundaryFragment

@dataclass
class AnnotationParams:
    match_threshold: float
    residue_state_space: ResidueStateSpace
    difference_threshold: float
    pivot_search_strategies: list[PivotSearchParams]
    pivot_symmetry_threshold: float
    pivot_score_threshold: float

class AnnotationResult:
    def __init__(self,
        pairs: list[PairedFragment],
        left_boundaries: list[LeftBoundaryFragment],
        right_boundaries: list[list[RightBoundaryFragment]],
        pivots: list[Pivot],
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

    def get_pairs(self):
        return self._pairs

    def get_left_boundaries(self):
        return self._left_boundaries

    def get_pivots(self):
        return self._pivots

    def get_right_boundaries(self, pivot_id: int):
        return self._right_boundaries[self._pivot_index[pivot_id]]

def _annotate(
    peaks: PeakList,
    params: AnnotationParams,
) -> AnnotationResult:
    # find pairs of peaks whose m/z difference has a ResidueState in the ResidueStateSpace.
    peak_pairs = find_pairs(
        peaks = peaks,
        match_threshold = params.match_threshold,
        residue_state_space = params.residue_state_space)
    # find pairs of pairs (or equivalent four-peak structures) that reflect about a common point of symmetry.
    pivots = find_pivots(
        pairs = peak_pairs,
        peaks = peaks,
        difference_threshold = params.difference_threshold,
        search_strategies = params.pivot_search_strategies)
    # find boundary peaks. 
    ## LeftBoundaryPeaks have m/z that is within a shift transformation of a ResidueState.    
    left_boundaries = find_left_boundaries(
        peaks = peaks,
        match_threshold = params.match_threshold,
        residue_state_space = params.residue_state_space)
    ## RightBoundaryPeaks have m/z that is within a reflection and shift of a ResidueState.
    ## the reflection is parametized by a pivot, so right_boundaries is a second-order collection.
    right_boundaries = find_right_boundaries(
        pivots = pivots,
        peaks = peaks,
        match_threshold = params.match_threshold,
        residue_state_space = params.residue_state_space)
    # score and filter pivots according to their symmetry and right boundary quality.
    pivots, pivot_index = rescore_pivots(
        pivots = pivots,
        left_boundaries = left_boundaries,
        right_boundaries = right_boundaries,
        symmetry_threshold = params.pivot_symmetry_threshold,
        score_threshold = params.pivot_score_threshold)
    # wrap everything up as an AnnotationResult object
    return AnnotationResult(
        pairs = peak_pairs,
        left_boundaries = left_boundaries,
        right_boundaries = right_boundaries,
        pivots = pivots,
        pivot_index = pivot_index)

def annotate(
    peak_lists: Iterator[PeakList],
    params: AnnotationParams,
) -> Iterator[AnnotationResult]:
    with Pool() as pool:
        return pool.map(
            ft.partial(_annotate, params = params),
            peak_lists)
