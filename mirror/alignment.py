from typing import Self
from dataclasses import dataclass
from multiprocessing import Pool
import functools as ft
import itertools as it

from .pivots import Pivot
from .graphs import construct_spectrum_graphs, align_spectrum_graphs, pair_alignments, LocalCostModel, AbstractAlignment
from .annotation import AnnotationResult

@dataclass
class AlignmentParams:
    cost_threshold: float
    cost_model: LocalCostModel
    aggregate_score_threshold: float

class AlignmentResult:
#    def __init__(self,
#        pivots: list[Pivots],
#        alignments: list[list[AbstractAlignment]],
#        alignment_pairs: list[list[tuple[int, int, int]]]
#    ):
#        super(AlignmentResult, self).__init__(paired_alignments)
#
#    @classmethod
#    def from_pivots(cls, pivots: list[Pivot]):
#        return cls(
#            pivots = pivots,
#            alignments = it.repeat(None, len(pivots)),
#            alignment_pairs = it.repeat(None, len(pivots)))

def _align(
    annotation: AnnotationResult,
    params: AlignmentParams,
) -> AlignmentResult:
    alignment_result = AlignmentResult.from_size(annotation.get_pivots())
    pairs = annotation.get_pairs()
    left_boundaries = annotation.get_left_boundaries()
    for (i, pivot) in enumerate(annotation.get_pivots()):
        right_boundaries = annotation.get_right_boundaries(i)
        spectrum_graph_pair = construct_spectrum_graphs(
            pairs = pairs,
            left_boundaries = left_boundaries,
            right_boundaries = right_boundaries,
            pivot = pivot)
        alignments = list(align_spectrum_graphs(
            graphs = spectrum_graph_pair,
            cost_model = params.cost_model,
            cost_threshold = params.cost_threshold))
        alignment_pairs = pair_alignments(
            pivot = pivot,
            alignments = alignments,
            score_threshold = params.aggregate_score_threshold)
#        alignment_result[i] = (alignments, alignment_pairs)
        # how is this data getting stored? we don't need to pass all of the alignment data to `filtration`, just
    return alignment_result

def align(
    annotations: Iterator[AnnotationResult],
    params: AlignmentParams,
) -> Iterator[AlignmentResult]:
    with Pool() as pool:
        return pool.map(
            ft.partial(_align, params = params),
            annotations)
