from typing import Self
from dataclasses import dataclass
from multiprocessing import Pool
import functools as ft
import itertools as it

from .pivots import AbstractPivot
from .graphs import construct_spectrum_graphs, align_spectrum_graphs, pair_alignments, SpectrumGraphPair, LocalCostModel, AbstractAlignment
from .sequences import SuffixArray

from .annotation import AnnotationResult

@dataclass
class AlignmentParams:
    cost_threshold: float
    cost_model: LocalCostModel
    suffix_array: SuffixArray
    reverse_suffix_array: SuffixArray
    aggregate_score_threshold: float

class AlignmentResult:
    pass

def compose(
    annotation: AnnotationResult,
    params: AlignmentParams,
) -> Iterator[SpectrumGraphPair]:
    pairs = annotation.get_pairs()
    left_boundaries = annotation.get_left_boundaries()
    for (i, pivot_cluster) in enumerate(anno.get_pivot_clusters()):
        yield SpectrumGraphPair.from_annotation(
            pairs = pairs,
            pivot_point = annotation.get_pivot_point(i),
            pivots = pivot_cluster,
            left_boundaries = left_boundaries,
            right_boundaries = annotation.get_right_boundaries(i))

def align(
    spectrum_graphs: Iterator[SpectrumGraphPair],
    params: AlignmentParams,
) -> AlignmentResult:
    offset = 0
    alignments = []
    alignment_pairs = []
    # each SpectrumGraphPair corresponds to a pivot cluster.
    for graph_pair in spectrum_graphs:
        # each prefix source is a (b_lo, y_hi) product node.
        # each suffix source is a (b_hi, y_lo) product node.
        # consider all combinations of sources.
        for (pfx_src, sfx_src) in product(graph_pair.prefix_sources, graph_pair.suffix_sources):
            # each sink product pair is an overlap pivot;
            # if there are no overlap pivots, it's all-to-all.
            for (pfx_snk, sfx_snk) in graph_pair.related_sinks:
                pfx_aln = align_spectrum_graphs(
                    topology = graph_pair.strong_product,
                    cost_model = params.cost_model,
                    cost_threshold = params.cost_threshold,
                    sources = pfx_src,
                    sinks = pfx_snk,
                    suffix_array = params.suffix_array)
                sfx_aln = align_spectrum_graphs(
                    topology = graph_pair.strong_product,
                    cost_model = params.cost_model,
                    cost_threshold = params.cost_threshold,
                    sources = sfx_src,
                    sinks = sfx_snk,
                    suffix_array = params.reverse_suffix_array)
                aln, aln_pairs = pair_disjoint_alignments(
                    prefixes = pfx_aln,
                    suffixes = sfx_aln)
                alignment_pairs.extend(aln_pairs)
                alignment.extend(aln)
                offset += len(aln)
