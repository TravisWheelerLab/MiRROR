import enum
from typing import Iterator

from ..heapq_numba import heappop as heappop_numba, heappush  as heappush_numba
from .types import SparseWeightedProductAdj, _index_ty, _label_ty, _cost_ty

import numpy as np
import numba

class EdgeType(enum.Enum):
    DIRECT = 1
    VERTICAL = 2
    HORIZONTAL = 3

@numba.jit(nopython=True)
def cost_fn(t: EdgeType, cost_model: tuple[float,float,float]):
    if t == EdgeType.DIRECT:
        return cost_model[0]
    elif t == EdgeType.VERTICAL:
        return cost_model[1]
    elif t == EdgeType.HORIZONTAL:
        return cost_model[2]

@numba.jit(nopython=True)
def strong_product_neighbors(
    left_adj: list[list[int]],
    right_adj: list[list[int]],
    node: tuple[int,int],
) -> Iterator[tuple[int,int,EdgeType]]:
    u, w = node
    
    for v in left_adj[u]:
        for x in right_adj[w]:
            yield (v, x, EdgeType.DIRECT)
    # direct edges
    
    for v in left_adj[u]:
        yield (v, w, EdgeType.VERTICAL)
    # vertical box edges
    
    for x in right_adj[w]:
        yield (u, x, EdgeType.HORIZONTAL)
    # horizontal box edges

@numba.jit(f"{_label_ty}({_label_ty},{_label_ty},{_label_ty})",nopython=True)
def ravel(
    i: int,
    j: int,
    n: int,
) -> int:
    return (i * n) + j

@numba.jit(f"Tuple(({_label_ty},{_label_ty}))({_label_ty},{_label_ty})",nopython=True)
def unravel(
    k: int,
    n: int,
) -> tuple[int,int]:
    return (
        k // n,
        n % n,
    )

_pq_state_ty = numba.types.Tuple((_cost_ty,_label_ty,_label_ty))

@numba.jit(nopython=True)
def _propagate_cost(
    left_adj: list[np.ndarray],
    right_adj: list[np.ndarray],
    initial_conditions: list[tuple[float,int,int]],
    pair_cost_table: dict[int,float],
    unpaired_cost: float,
    threshold: float,
    cost_model: tuple[float,float,float],
) -> tuple[
    int,             # number of nodes in the sparse product graph.
    dict[int,int],   # node indexer.
    list[int],       # sparse edge sources. edges reversed relative to inputs.
    list[int],       # sparse edge targets. "                                "
    list[float],     # sparse cost list.
    list[int],       # sparse de-indexer.
]:
    right_order = len(right_adj)

    pq = numba.typed.List.empty_list(_pq_state_ty)
    for entry_state in initial_conditions:
        heappush_numba(pq, entry_state)
    # construct priority queue from initial conditions
    
    n = 0
    node_index = numba.typed.Dict.empty(key_type=_label_ty,value_type=_index_ty)
    sparse_edge_src = numba.typed.List.empty_list(_index_ty)
    sparse_edge_tgt = numba.typed.List.empty_list(_index_ty)
    sparse_cost = numba.typed.List.empty_list(_cost_ty)
    sparse_labels = numba.typed.List.empty_list(_label_ty)
    # return types

    while len(pq) > 0:
        path_cost, prev_node, curr_node = heappop_numba(pq)
        
        if not((path_cost > threshold) or (curr_node in node_index)):
            # terminate paths that either
            # 1. exceed the threshold, or
            # 2. encounter a node already visited (by a cheaper path.)

            node_index[curr_node] = n
            sparse_cost.append(path_cost)
            sparse_labels.append(curr_node)
            n += 1
            # reached a new node; record its index, label, and cost.

            neighbors = strong_product_neighbors(left_adj, right_adj, unravel(curr_node, right_order))
            for (l, r, edge_type) in neighbors:
                next_node = ravel(l, r, right_order)
                edge_cost = cost_fn(edge_type, cost_model)
                pair_cost = pair_cost_table.get(next_node, unpaired_cost)
                new_cost = path_cost + edge_cost + pair_cost
                heappush_numba(pq, (new_cost, curr_node, next_node))
                # record the next step into the graph with cost determined by edge type and predetermined node costs

        if not(prev_node == curr_node):
            sparse_edge_src.append(node_index[curr_node])
            sparse_edge_tgt.append(node_index[prev_node])
        # finally, record the reversed edge curr_node -> prev_node in the sparse adjacency matrix.

    return (
        n,
        node_index,
        sparse_edge_src,
        sparse_edge_tgt,
        sparse_cost,
        sparse_labels,
    )

def make_cost_model(
    cost_model: tuple[float,float,float],
):
    return tuple([_cost_ty(x) for x in cost_model])

def make_pair_cost_table(
    right_order: int,
    paired_nodes: list[tuple[int,int]],
    pair_costs: list[float],
) -> dict[int,float]:
    table = numba.typed.Dict.empty(_label_ty,_cost_ty)
    for ((u, w), cost) in zip(paired_nodes, pair_costs):
        pair = _label_ty(ravel(u, w, right_order))
        table[pair] = _cost_ty(cost)
    return table

def make_initial_conditions(
    right_order: int,
    left_sources: list[int],
    right_sources: list[int],
    pair_cost_table: dict[int,float],
    unpaired_cost: float,
):
    product_sources = [ravel(u, w, right_order) for u in left_sources for w in right_sources]
    return [(
        _cost_ty(pair_cost_table.get(node, unpaired_cost)),
        _label_ty(node),
        _label_ty(node),
    ) for node in product_sources]

def propagate_cost(
    right: Adj,
    left: Adj,
    paired_nodes: list[tuple[int,int]],
    pair_costs: list[float],
    unpaired_cost: float,
    threshold: float,
    cost_model: tuple[float,float,float,float],
) -> SparseWeightedProductAdj:
    pair_cost_table = make_pair_cost_table(
      right.order,
      paired_nodes,
      pair_costs,
    )
    unpaired_cost = _cost_ty(unpaired_cost)
    initial_conditions = make_initial_conditions(
        right.order,
        left.sources,
        right.sources,
        pair_cost_table,
        unpaired_cost,
    )
    cost_model = make_cost_model(cost_model)
    prop_res = _propagate_cost(
        left.adj,
        right.adj,
        initial_conditions,
        pair_cost_table,
        unpaired_cost,
        threshold,
        cost_model,
    )
    n, node_index, sparse_edge_src, sparse_edge_tgt, sparse_cost, sparse_labels = prop_res
    sparse_adj = [[] for _ in range(n)]
    for (i, j) in zip(sparse_edge_src, sparse_edge_tgt):
        sparse_adj[i].append(j)
    return SparseWeightedProductAdj(
        n,
        node_index,
        sparse_adj,
        sparse_cost,
        sparse_labels,
    )
