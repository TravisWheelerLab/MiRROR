from itertools import pairwise

from .align_types import AbstractAlignment, LocalAlignment, AbstractCostModel, LocalCostModel
from .graph_types import DAG, StrongProductDAG
from .minimal_nodes import propagate
from .minimal_paths import backtrace

from networkx import DiGraph
from numpy import inf

def _all_skips(fragment: list[tuple[int, int]]):
    fragment_edges = pairwise(fragment)
    for ((x1, x2), (y1, y2)) in fragment_edges:
        if x1 != y1 and x2 != y2:
            return False
    return True

def align(
    product_graph: StrongProductDAG,
    cost_model: AbstractCostModel,
    threshold = inf,
    precision = 10,
    path_filter = lambda x: True,
) -> list[AbstractAlignment]:
    # set up; alignment type, cost function, 
    if isinstance(cost_model, LocalCostModel):
        alignment_type = LocalAlignment
    else:
        raise ValueError(f"unsupported cost model: {type(cost_model)}")
    cost_fn = cost_model(product_graph)
    # enumerate alignments
    sources = list(product_graph.sources())
    sinks = list(product_graph.sinks())
    aligned_paths = []
    for source in sources:
        node_cost = propagate(
            topology = product_graph,
            cost = cost_fn,
            threshold = threshold,
            source = source)
        for sink in sinks:
            aligned_paths.extend(backtrace(
                topology = product_graph,
                cost = cost_fn,
                node_cost = node_cost,
                threshold = threshold,
                source = source,
                sink = sink,
                path_filter = path_filter))
    # filter alignments
    aligned_paths = [a for a in aligned_paths if not _all_skips([product_graph.unravel(v) for v in a[1]])]
    return list(map(
        lambda x: alignment_type(
            score = round(x[0], precision), 
            alignment_nodes = [product_graph.unravel(v) for v in x[1]],
            alignment_weights = [product_graph.weight_out(v, w) for (v, w) in pairwise(x[1])],
            cost_model = cost_model),
        aligned_paths))
