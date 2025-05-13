from typing import Iterator, Any
from .graph_types import DiGraph, DAG
from networkx import is_bipartite

class MultiGraph(DiGraph):
    """A digraph that allows multiple edges between a pair
    of nodes. A multi-edge comprising `n` edges is represented as a single 
    edge with integer weight `n`. Constructed from an iterable of edges.
    To change the weight key, set the `count_weight_key` kwarg."""
    def __init__(self,
        edges: Iterator[tuple[Any, Any]],
        count_weight_key = "count"
    ):
        multigraph = DiGraph()
        for (i, j) in edges:
            multigraph.add_edge(i,j)
            if count_weight_key not in multigraph[i][j]:
                multigraph[i][j][count_weight_key] = 0
            multigraph[i][j][count_weight_key] += 1
        super(MultiGraph, self).__init__(incoming_graph_data = multigraph)

class BipartiteGraph(DiGraph):
    def __init__(self,
        edges = Iterator[tuple[Any, Any]],
    ):
        graph = MultiGraph(edges)
        if is_bipartite(graph):
            super(BipartiteGraph, self).__init__(incoming_graph_data = graph)
        else:
            raise ValueError("not a bipartite graph!")

class MultiDAG(DAG):
    """A directed acyclic graph that allows multiple edges between a pair
    of nodes. A multi-edge comprising `n` edges is represented as a single 
    edge with integer weight `n`. Constructed from an iterable of edges.
    To change the weight key, set the `count_weight_key` kwarg."""
    def __init__(self,
        edges: Iterator[tuple[Any, Any]],
        count_weight_key = "count"
    ):
        super(MultiDAG, self).__init__(
            graph = MultiGraph(edges),
            weight_key = count_weight_key,
        )

