from typing import Iterator, Any
from itertools import pairwise, chain
from functools import reduce
from operator import eq, ne

from .graph_types import ProductDAG, MultiDAG, BipartiteGraph
from .consensus_types import *

def _filter_loops(
    edges: Iterator[tuple[Any, Any]],
):
    return filter(
        lambda x: ne(*x),
        edges)

def _collect_edges(
    paths: Iterator[list[Any]]
) -> Iterator[tuple[Any, Any]]:
    """Given an iterator of paths, collect all 
    the edges in each path as a flat iterator."""
    return chain.from_iterable(map(
        lambda x: pairwise(x),
        paths))

def _path_projections(
    aligned_paths: Iterator[list[tuple[int, int]]],
) -> tuple[Iterator[list[int]],Iterator[list[int]]]:
    """Given an iterator of aligned paths, where each node is a 2-tuple,
    collect the projections of each path into its first and second components."""
    first_paths, second_paths = zip(*map(
        lambda x: zip(*x),
        aligned_paths))
    return first_paths, second_paths

def consensus_graphs(
    aligned_paths: Iterator[list[tuple[int, int]]]
):
    """A pair of graphs with edges weighted by their occurrence in `aligned_paths`."""
    first_paths, second_paths = _path_projections(aligned_paths)
    first_edges = list(_filter_loops(_collect_edges(first_paths)))
    second_edges = list(_filter_loops(_collect_edges(second_paths)))
    return (
        MultiDAG(
            edges = first_edges),
        MultiDAG(
            edges = second_edges))

def associate_matched_edges(
    product_graph: ProductDAG,
    aligned_paths: Iterator[list[tuple[int, int]]]
):
    """Associate edges in the first component of `product_graph` to edges in 
    the second component if they are matched in `aligned_paths`."""
    aligned_edges = list(_collect_edges(aligned_paths))
    matched_edges = list(filter(
        lambda e: reduce(
            eq,
            product_graph.weight_out(
                product_graph.ravel(*e[0]),
                product_graph.ravel(*e[1]))),
        aligned_edges))
    projected_edge_pairs = map(
        lambda x: (
            ("first", x[0][0], x[1][0]),
            ("second", x[0][1], x[1][1])),
        matched_edges)
    return BipartiteGraph(
        edges = projected_edge_pairs)

def maximal_associated_consensus_paths(
    product_graph: ProductDAG,
    aligned_paths: Iterator[list[tuple[int, int]]],
):
    first_consensus, second_consensus = consensus_graphs(aligned_paths)
    associated_edges = associate_matched_edges(product_graph, aligned_paths)
