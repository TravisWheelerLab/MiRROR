# python dependencies
from typing import Iterator, Iterable, Self
from dataclasses import dataclass
import itertools as it
# horizontal dependencies
from .graph_types import ReindexedDAG
from .align_types import FuzzyLocalCostModel, AbstractAlignment, LocalAlignment, AlignmentFilter
from .align import align
from .ensemble_types import EnsembleAlignment
from .ensemble import assemble_alignments
from .concatenation_types import ConcatenationAlignment
from .concatenation import concatenate_ensembles
# module dependencies
from ..util import binsort, disjoint_pairs
from ..fragments import PairedFragments, BoundaryFragment, ReflectedBoundaryFragment, Pivot, OverlapPivot, VirtualPivot
from ..sequences.suffix_array import SuffixArray

class SuffixArrayAlignmentFilter(AlignmentFilter):
    def __init__(
        self,        
    ):
        pass

@dataclass
class SpectrumGraph:
    """A wrapper for NodeLabeledDAG implementing the from_pairs constructor."""
    dag: ReindexedDAG
    b_boundaries: list[int]
    y_boundaries: list[int]
    unk_boundaries: list[int]

    @classmethod
    def from_pairs(cls,
        pairs: Iterator[PairedFragments],
        boundaries: list[BoundaryFragment],
        reverse: bool,
        weight_key: str = "state",
    ) -> Self:
        """Construct a graph over the edges given by the peak indices of `pairs` weighted by the right fragment and residue. Peak indices are swapped if `reverse` is True."""
        edges = [p.peak_indices() for p in pairs]
        if reverse:
            edges = [(j,i) for (i,j) in edges]
        dag = ReindexedDAG.from_edges(
            edges = edges,
            weights = [(p.right_fragment, p.residue) for p in pairs],
            weight_key = weight_key)
        b_boundaries = [dag.get_node_idx(x.fragment.peak_idx) for x in boundaries if x.series == 'b']
        y_boundaries = [dag.get_node_idx(x.fragment.peak_idx) for x in boundaries if x.series == 'y']
        unk_boundaries = [dag.get_node_idx(x.fragment.peak_idx) for x in boundaries if x.series not in ['b','y']]
        return cls(dag, b_boundaries, y_boundaries, unk_boundaries)

def partition_pairs(
    pivot_point: float,
    pairs: list[PairedFragments],
    left_boundaries: list[BoundaryFragment],
    right_boundaries: list[ReflectedBoundaryFragment],
) -> tuple[SpectrumGraph,SpectrumGraph,list[PairedFragments]]:
    """Construct spectrum graphs over PairedFragments as edges, *BoundaryFragments as sources, and Pivot-induced sinks."""
    # partition pairs into left, right, and cut.
    left_pairs = []
    right_pairs = []
    cut_pairs = []
    for p in pairs:
        left_idx, right_idx = p.peak_indices()
        left_mass, right_mass = p.fragment_masses()
        if (right_mass < pivot_point):
            left_pairs.append(p)
        if (left_mass > pivot_point):
            right_pairs.append(p)
        if (left_mass < pivot_point < right_mass):
            cut_pairs.append(p)
    # remove singletons from boundaries
    left_nodes = set(it.chain.from_iterable(p.peak_indices() for p in left_pairs))
    left_boundaries = [x for x in left_boundaries if x.fragment.peak_idx in left_nodes]
    right_nodes = set(it.chain.from_iterable(p.peak_indices() for p in right_pairs))
    right_boundaries = [x for x in right_boundaries if x.fragment.peak_idx in right_nodes]

    return (
        (left_pairs, left_boundaries),
        (right_pairs, right_boundaries),
        cut_pairs)

def _align_spectrum_graphs(
    graphs: tuple[SpectrumGraph,SpectrumGraph],
    cost_model_type: FuzzyLocalCostModel,
    cost_threshold: float,
    suffix_array: SuffixArray,
) -> list[AbstractAlignment]:
    pass

def align_prefixes(
    graphs: tuple[SpectrumGraph,SpectrumGraph],
    cost_model_type: FuzzyLocalCostModel,
    cost_threshold: float,
    suffix_array: SuffixArray,
) -> list[AbstractAlignment]:
    pass 

def align_suffixes(
    graphs: tuple[SpectrumGraph,SpectrumGraph],
    cost_model_type: FuzzyLocalCostModel,
    cost_threshold: float,
    suffix_array: SuffixArray,
) -> list[AbstractAlignment]:
    pass 

def align_spectrum_graphs(
    graphs: tuple[SpectrumGraph,SpectrumGraph],
    cost_model_type: FuzzyLocalCostModel,
    cost_threshold: float,
) -> tuple[list[AbstractAlignment],list[AbstractAlignment]]:
    # construct the product
    product_graph = StrongProductDAG(*graphs)
    
    # propagate boundary annotations
    # TODO
    
    # align for prefixes and suffixes
    prefix_sources = zip(
            graphs[0].get_b_sources(),
            graphs[1].get_y_sources())
    prefix_alignments = align(
        *graphs,
        cost_model = cost_model,
        cost_threshold = cost_threshold,
        sources = prefix_sources)
    
    suffix_sources = zip(
            graphs[0].get_y_sources(),
            graphs[1].get_b_sources())
    prefix_alignments = align(
        *graphs,
        cost_model = cost_model,
        cost_threshold = cost_threshold,
        sources = suffix_sources)

    return (
        prefix_alignments,
        suffix_alignments)

def pair_alignments(
    alignments: tuple[list[AbstractAlignment],list[AbstractAlignment]],
    graphs: tuple[SpectrumGraph,SpectrumGraph],
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
