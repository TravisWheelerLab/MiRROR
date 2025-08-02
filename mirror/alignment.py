from typing import Self
from dataclasses import dataclass
from multiprocessing import Pool
import functools as ft
import itertools as it

from .pivots import AbstractPivot
from .graphs import construct_spectrum_graphs, align_spectrum_graphs, pair_alignments, SpectrumGraph, LocalCostModel, AbstractAlignment
from .sequences import SuffixArray, AffixPair

from .annotation import AnnotationResult

@dataclass
class AlignmentParams:
    cost_threshold: float
    cost_model: LocalCostModel
    suffix_array: SuffixArray
    aggregate_score_threshold: float

class AlignmentResult:
    def __init__(self,
        pivot: AbstractPivot,
        alignments: list[AbstractAlignment],
        affix_pairs: list[tuple[int, int]],
    ):
        self._pivot = pivot
        self._alignments = alignments
        self._affix_pairs = affix_pairs
        self._n = len(affix_pairs)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self,
        i: int,
    ) -> AffixPair:
        l, r = self._affix_pairs[i]
        return AffixPair(
            pivot = pivot,
            prefix = self._alignments[l],
            suffix = self._alignments[r])

    def __iter__(self) -> Iterator[AffixPair]:
        return map(
            self.__getitem__,
            range(len(self)))

def compose(
    annotation: AnnotationResult,
    params: AlignmentParams,
) -> Iterator[tuple[SpectrumGraph,SpectrumGraph]]:
    pairs = annotation.get_pairs()
    left_boundaries = annotation.get_left_boundaries()
    for (i, pivot) in enumerate(annotation.get_pivots()):
        right_boundaries = annotation.get_right_boundaries(i)
        spectrum_graph_pair = construct_spectrum_graphs(
            pairs = pairs,
            left_boundaries = left_boundaries,
            right_boundaries = right_boundaries,
            pivot = pivot)
        yield pivot, spectrum_graph_pair

def align(
    pivot: AbstractPivot,
    spectrum_graphs: tuple[SpectrumGraph,SpectrumGraph],
    params: AlignmentParams,
) -> AlignmentResult:
    alignments = list(align_spectrum_graphs(
        graphs = spectrum_graphs,
        cost_model = params.cost_model,
        cost_threshold = params.cost_threshold))
    alignment_pairs = pair_alignments(
        pivot = pivot,
        alignments = alignments,
        score_threshold = params.aggregate_score_threshold)
    return AlignmentResult(
        pivot = pivot,
        alignments = alignments,
        alignment_pairs = alignment_pair)
