# python dependencies
from typing import Iterator, Iterable, Self
import dataclasses
import itertools as it
# horizontal dependencies
from .graph_types import ReindexedDAG, ProductDAG, BoxProductDAG, DirectProductDAG, StrongProductDAG
from .align_types import FuzzyLocalCostModel, CostFunction, AbstractAlignment, LocalAlignment, AlignmentFilter
from .align import align
from .ensemble_types import EnsembleAlignment
from .ensemble import assemble_alignments
from .concatenation_types import ConcatenationAlignment
from .concatenation import concatenate_ensembles
# module dependencies
from ..util import binsort, disjoint_pairs
from ..fragments import PairedFragments, BoundaryFragment, ReflectedBoundaryFragment, Pivot, OverlapPivot, VirtualPivot
from ..sequences.suffix_array import SuffixArray

def partition_pairs(
    pivot_point: float,
    pairs: list[PairedFragments],
    left_boundaries: list[BoundaryFragment],
    right_boundaries: list[ReflectedBoundaryFragment],
) -> tuple[list[PairedFragments],list[PairedFragments],list[PairedFragments]]:
    """Construct spectrum graphs over PairedFragments as edges, *BoundaryFragments as sources, and Pivot-induced sinks."""
    # partition pairs into left, right, and cut based on their position relative to the pivot point and min/max left/right boundaries.
    left_bound = min(lb.fragment.fragment_mass for lb in left_boundaries)
    right_bound = max(2 * pivot_point - rb.fragment.fragment_mass for rb in right_boundaries)
    left_pairs = []
    right_pairs = []
    cut_pairs = []
    for p in pairs:
        left_idx, right_idx = p.peak_indices()
        left_mass, right_mass = p.fragment_masses()
        if left_mass < pivot_point < right_mass:
            cut_pairs.append(p)
        elif left_bound <= right_mass < pivot_point:
            left_pairs.append(p)
        elif pivot_point < left_mass <= right_bound:
            right_pairs.append(p)
    return (
        left_pairs,
        right_pairs,
        cut_pairs)

def construct_graph(
    pairs: list[PairedFragments],
    boundaries: list[BoundaryFragment],
    reverse: bool,
) -> ReindexedDAG:
    edges = [p.peak_indices() for p in pairs]
    data = [(p.left_fragment, p.right_fragment, p.residue, p.cost()) for p in pairs]
    dag = ReindexedDAG.from_edges(
        edges = [(j, i) if reverse else (i, j) for (i, j) in edges],
        weights = data,
        weight_key = 'w')
    b_boundaries = []
    y_boundaries = []
    unk_boundaries = []
    for x in boundaries:
        if x.fragment.peak_idx in dag:
            series = x.series
            if series == 'b':
                b_boundaries.append(x)
            elif series == 'y':
                y_boundaries.append(x)
            else:
                unk_boundaries.append(x)
    return (
        dag,
        b_boundaries,
        y_boundaries,
        unk_boundaries)

@dataclasses.dataclass(slots=True)
class SpectrumGraphPair:
    """Product graph types, prefix and suffix partitions of the source node product, and sink node products related by pivot structures."""
    pivot_point: float
    left: ReindexedDAG
    right: ReindexedDAG
    strong_product: StrongProductDAG
    box_product: BoxProductDAG
    direct_product: DirectProductDAG
    prefix_sources: list[int]
    suffix_sources: list[int]
    related_sinks: list[tuple[int,int]]

    @classmethod
    def from_annotation(cls,
        pairs: list[PairedFragments],
        pivot_point: float,
        pivots: list[Pivot],
        left_boundaries: list[BoundaryFragment],
        right_boundaries: list[ReflectedBoundaryFragment],
    ) -> Self:
        """Compose annotated spectrum data (pairs, pivots, boundaries) into a pair of spectrum graphs represented as a strong product. Partition the product of boundaries into prefix and suffix sources. Construct a subset of the sink product and a relation over it induced by the overlap pivots."""
        left_pairs, right_pairs, cut_pairs = partition_pairs(
            pivot_point,
            pairs,
            left_boundaries,
            right_boundaries)
        left, left_b_src, left_y_src, _ = construct_graph(
            left_pairs,
            left_boundaries,
            reverse=False)
        right, right_b_src, right_y_src, __ = construct_graph(
            right_pairs,
            right_boundaries,
            reverse=True)
        # prefix sources take the form (left b boundary, right y boundary).
        pfx_src_prod = [(
                left.get_node_idx(b.fragment.peak_idx),
                right.get_node_idx(y.fragment.peak_idx)) 
            for (b, y) in it.product(left_b_src, right_y_src)]
        # suffix sources take the form (left y boundary, right b boundary).
        sfx_src_prod = [(
                left.get_node_idx(y.fragment.peak_idx),
                right.get_node_idx(b.fragment.peak_idx)) 
            for (y, b) in it.product(left_y_src, right_b_src)]
        # sinks are dual to overlap pivots: if the pivot is formed from two edges e = (e1, e2), f = (f1, f2) such that e1 < f1 < pivot_point < e2 < f2, then the corresponding pair of prefix, suffix sinks is (e1, f2), (f1, e2).
        overlap_pivots = [p for p in pivots if isinstance(p,OverlapPivot)]
        rel_snk_prod = [(
                (
                    left.get_node_idx(p.indices[0]),
                    right.get_node_idx(p.indices[3])),
                (
                    left.get_node_idx(p.indices[1]),
                    right.get_node_idx(p.indices[2])))
            for p in overlap_pivots if all(x in left for x in p.indices[:2]) and all(x in right for x in p.indices[2:])]
        # TODO: if there are no overlap pivots, construct the sink products from the cut pairs.
        strong_product = StrongProductDAG(left, right)
        return cls(
            pivot_point = pivot_point,
            left = left,
            right = right,
            strong_product = strong_product,
            box_product = strong_product.subgraph_box,
            direct_product = strong_product.subgraph_direct,
            prefix_sources = [strong_product.ravel(*x) for x in pfx_src_prod],
            suffix_sources = [strong_product.ravel(*x) for x in sfx_src_prod],
            related_sinks = [(
                    strong_product.ravel(*outer_anti_edge),
                    strong_product.ravel(*inner_anti_edge))
                for (outer_anti_edge, inner_anti_edge) in rel_snk_prod])

@dataclasses.dataclass
class SpectrumGraphCostModel:
    graph_pair: SpectrumGraphPair
    sources: list[int]
    sinks: list[int]
    match: float = 0
    skip: float = 0.
    substitution: float = 1.
    gap: float = 2.
    tolerance: float = 0.01

    def __call__(
        self, 
        product_graph: ProductDAG,
    ) -> CostFunction:
        first_sources, second_sources = zip(
            *[product_graph.unravel(s) for s in self.sources])
        first_sinks, second_sinks = zip(
            *[product_graph.unravel(t) for t in self.sinks])
        first_extremities = set(first_sources + first_sinks)
        second_extremities = set(second_sources + second_sinks)
        def cost_function(edge: tuple, weight: tuple):
            first_edge, second_edge = zip(
                *[product_graph.unravel(v) for v in edge])
            is_first_stationary = first_edge[0] == first_edge[1]
            is_second_stationary = second_edge[0] == second_edge[1]
            first_weight, second_weight = weight
            cost = 0.
            if not (is_first_stationary or is_second_stationary):
                _, first_fragment, first_residue, first_cost = first_weight
                __, second_fragment, second_residue, second_cost = second_weight
                cost += first_cost + second_cost
                if first_residue == second_residue:
                    cost += self.match
                else:
                    cost += self.substitution
            elif is_first_stationary:
                __, second_fragment, second_residue, second_cost = second_weight
                cost += second_cost
                if first_edge[0] in first_extremities:
                    cost += self.skip
                else:
                    cost += self.gap
            elif is_second_stationary:
                _, first_fragment, first_residue, first_cost = first_weight
                cost += first_cost
                if second_edge[0] in second_extremities:
                    cost += self.skip
                else:
                    cost += self.gap
            else:
                raise ValueError(f"both edges are stationary! {edge} {weight}")
            return cost
        return cost_function

class SuffixArrayAlignmentFilter(AlignmentFilter):
    def __init__(
        self,
        suffix_array,
    ):
        pass

def align_spectrum_graphs(
    topology: ProductDAG,
    cost_model: FuzzyLocalCostModel,
    cost_threshold: float,
    sources: list[int],
    sinks: list[int],
    suffix_array: SuffixArray,
) -> list[AbstractAlignment]:
    return align(
        product_graph = topology,
        cost_model = cost_model,
        threshold = cost_threshold,
        path_filter = None if (suffix_array is None) else SuffixArrayAlignmentFilter(suffix_array),
        sources = sources,
        sinks = sinks)

def pair_alignments(
    alignments: tuple[list[AbstractAlignment],list[AbstractAlignment]],
    graphs: SpectrumGraphPair,
    cut_pairs: list[PairedFragments],
    pivot: Pivot,
    score_threshold: float,
) -> Iterator[tuple[AbstractAlignment,AbstractAlignment]]:
    prefixes, suffixes = alignments
    reindexed_target = lambda aln: (
        graphs[0].get_node_label(aln.first_target()),
        graphs[1].get_node_label(aln.second_target()))
    prefix_sinks, prefixes = binsort(prefixes, key = reindexed_target)
    suffix_sinks, suffixes = binsort(suffixes, key = reindexed_target)
    # create a bipartite graph from the cut pairs
    pair_indices = [p.peak_indices() for p in cut_pairs]
    pair_tab = {}
    for (p,s) in pair_indices:
        if p in pair_tab:
            pair_tab[p].append(s)
        else:
            pair_tab[p] = [s]
    # map from suffix sinks (node tuples) to their index w.r.t. the suffixes
    suf_snk_tab = {
        snk: idx for (idx, snk) in enumerate(suffix_sinks)}
    # generate all viable affix pairs
    for (pre_idx, pre_snk) in enumerate(prefix_sinks):        
        pre_bin = prefixes[pre_idx]
        for suf_snk in zip(pair_tab[pre_snk[0]], pair_tab[pre_sink[1]]):
            suf_idx = suf_snk_tab[suf_snk]
            suf_bin = suffixes[suf_idx]
            for (i, j) in pairwise_disjoint(pre_bin, suf_bin):
                yield (pre_bin[i], suf_bin[j])
