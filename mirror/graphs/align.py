from itertools import pairwise

from .align_types import AlignedPath, CostModel
from .graph_types import DAG, StrongProductDAG
from .minimal_nodes import propagate
from .minimal_paths import backtrace

from networkx import DiGraph
from numpy import inf

def align(
    product_graph: StrongProductDAG,
    cost_model: CostModel,
    threshold = inf,
    precision = 10,
) -> list[AlignedPath]:
    sources = list(product_graph.sources())
    sinks = list(product_graph.sinks())
    cost_fn = cost_model(product_graph)
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
                sink = sink))
    return list(map(
        lambda x: AlignedPath(
            round(x[0], precision), 
            [product_graph.unravel(v) for v in x[1]],
            [product_graph.weight_out(v, w) for (v, w) in pairwise(x[1])]),
        aligned_paths))