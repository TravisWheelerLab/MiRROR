from typing import Iterator
from .minimal_nodes import propagate
from .minimal_paths import backtrace
from .align_types import AlignedPath, CostModel
from .fragment_types import FragmentIntersectionGraph, FragmentPairGraph, FragmentChain
from networkx import is_bipartite, connected_components

def chain_fragment_pairs(
    fragment_intersection_graph: FragmentIntersectionGraph,
    cost_model: CostModel,
    threshold: float,
):
    fragment_pair_chains = []
    for component in connected_components(fragment_intersection_graph):
        frag_pair = FragmentPairGraph(
            fragment_itx = fragment_intersection_graph,
            component = component,
            cost_model = cost_model,
        )
        sources = frag_pair.sources()
        sinks = frag_pair.sinks()
        identity_cost = lambda _, x: x
        for src in sources:
            nc = propagate(
                topology = frag_pair,
                cost = identity_cost,
                threshold = threshold,
                source = src,
            )
            for snk in sinks:
                minimal_paths = backtrace(
                    topology = frag_pair,
                    cost = identity_cost,
                    node_cost = nc,
                    threshold = threshold,
                    source = src,
                    sink = snk,
                )
                fragment_pair_chains.extend(minimal_paths)
    return fragment_pair_chains

def collate_fragments(
    alignments: Iterator[AlignedPath],
    cost_model: CostModel,
    threshold: float,
) -> list[FragmentChain]:
    # construct the fragment intersection
    frag_itx = FragmentIntersectionGraph(
        alignments = alignments,
    )
    assert is_bipartite(frag_itx)
    # find sequences of fragment pairs that can be concatenated to
    # form longer alignments.
    # parametize with `threshold` to filter out low-quality chains.
    # use fragment pair chains to guide construction of new alignments
    fragment_pair_chains = chain_fragment_pairs(
        fragment_intersection_graph = frag_itx,
        cost_model = cost_model,
        threshold = threshold,
    )
    # construct the FragmentChain objects, associating each fragment pair chain
    # to a single alignment-like structure.
    return list(map(
        lambda x: FragmentChain(
            score = x[0],
            alignment_chain = [alignments[i] for i in x[1]]),
        fragment_pair_chains))