from typing import Iterator, Any
from itertools import pairwise, chain
from functools import reduce
from operator import eq, ne

from .graph_types import StrongProductDAG
from .align_types import CostModel

from numpy import inf
from networkx import Graph, is_bipartite, connected_components

unique = lambda x: list(map(
    list, 
    set(map(
        tuple, 
        x))))
    
def edge_to_path_table(dag, paths, offset = 0):
    edge_to_paths = [dict() for _ in range(dag.order())]
    for path_idx, path in enumerate(paths):
        print(path_idx, path_weights(dag, path))
        for (src, tgt) in pairwise(path):
            if ne(src, tgt):
                if not(tgt in edge_to_paths[src]):
                    edge_to_paths[src][tgt] = []
                edge_to_paths[src][tgt].append(path_idx)
    return edge_to_paths

def path_weights(dag, path):
    return [dag.weight_out(x, y) for (x, y) in pairwise(path)]

def realign_fragments(
    aligned_fragments: Iterator[list[tuple[int, int]]],
    product_graph: StrongProductDAG,
    cost_model: CostModel,
    threshold = inf,
    precision = 10,
):
    # unzip alignments, reduce each projection to a unique set of paths.
    first_fragments, second_fragments = map(unique, zip(*map(lambda x: zip(*x), aligned_fragments)))    
    n = len(first_fragments)

    # construct the fragment intersection graph
    first_dag = product_graph.first_graph
    second_dag = product_graph.second_graph
    ## construct the edge match graph
    match_graph = Graph()
    for algn_frag in aligned_fragments:
        for ((x1, x2), (y1, y2)) in pairwise(algn_frag):
            w1 = first_dag.weight_out(x1, y1)
            w2 = second_dag.weight_out(x2, y2)
            if w1 == w2:
                print(x1, y1, w1, '\t', x2, y2, w2)
                match_graph.add_edge(('1', x1, y1), ('2', x2, y2))
    assert is_bipartite(match_graph)
    print("match", match_graph.edges)
    ## construct edge -> path tables
    ### table 1
    first_edge_to_paths = edge_to_path_table(first_dag, first_fragments)
    ### table 2
    second_edge_to_paths = edge_to_path_table(second_dag, second_fragments)
    ## loop matched edges and connect all the paths supported by each edge
    fragment_itx = Graph()
    for ((_, x1, y1), (__, x2, y2)) in match_graph.edges():
        for i in first_edge_to_paths[x1][y1]:
            for j in second_edge_to_paths[x2][y2]:
                print((i, j), (i, n + j), (path_weights(first_dag, first_fragments[i]), path_weights(second_dag, second_fragments[j])))
                print('\t', first_fragments[i], second_fragments[j])
                fragment_itx.add_edge(i, n + j)
    assert is_bipartite(fragment_itx)
    print("fragment", fragment_itx.edges)
    # from each component in the intersection graph
    # construct a topological order
    #fragment_orders = map(
    #    construct_fragment_preorder,
    #    connected_components(fragment_itx))
    
    # find the maximal paths of each topological order
    