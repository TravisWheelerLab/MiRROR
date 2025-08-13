from typing import Iterator
from .minimal_nodes import propagate
from .minimal_paths import backtrace
from .align_types import LocalAlignment, LocalCostModel
from .ensemble_types import AlignmentIntersectionGraph, AlignmentPairGraph, EnsembleAlignment
from networkx import is_bipartite, connected_components

def solve_alignment_pair_chains(
    alignment_intersection_graph: AlignmentIntersectionGraph,
    cost_model: LocalCostModel,
    threshold: float,
):
    alignment_pair_chains = []
    for component in connected_components(alignment_intersection_graph):
        aln_pair = AlignmentPairGraph(
            alignment_itx = alignment_intersection_graph,
            component = component,
            cost_model = cost_model)
        sources = aln_pair.sources()
        sinks = aln_pair.sinks()
        # the weights of aln_pair are their own cost
        identity_cost = lambda _, x: x
        for src in sources:
            nc = propagate(
                topology = aln_pair,
                cost = identity_cost,
                threshold = threshold,
                source = src,
            )
            for snk in sinks:
                minimal_paths = backtrace(
                    topology = aln_pair,
                    cost = identity_cost,
                    node_cost = nc,
                    threshold = threshold,
                    source = src,
                    sink = snk,
                )
                alignment_pair_chains.extend(minimal_paths)
    return alignment_pair_chains

def assemble_alignments(
    alignments: list[LocalAlignment],
    cost_model: LocalCostModel,
    threshold: float,
) -> list[EnsembleAlignment]:
    # construct the alignment intersection
    aln_itx = AlignmentIntersectionGraph(
        alignments = alignments)
    assert is_bipartite(aln_itx)
    # find sequences of alignment pairs that can be concatenated to
    # form longer alignments.
    # parametize with `threshold` to filter out low-quality chains.
    # use alignment pair chains to guide construction of new alignments
    alignment_pair_chains = solve_alignment_pair_chains(
        alignment_intersection_graph = aln_itx,
        cost_model = cost_model,
        threshold = threshold)
    # construct the EnsembleAlignments, associating each alignment pair chain
    # to a single alignment object.
    return list(map(
        lambda x: EnsembleAlignment(
            score = x[0],
            alignment_chain = [alignments[i] for i in x[1]]),
        alignment_pair_chains))
