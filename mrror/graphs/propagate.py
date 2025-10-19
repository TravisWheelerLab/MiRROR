from typing import Iterator
from heapq import heappush, heappop

from .types import SpectrumGraph, WeightedProductGraph

import numpy as np
import networkx as nx

def strong_product_neighbors(
    left_adj: list[list[int]],
    right_adj: list[list[int]],
    node: tuple[int,int],
    direct: float,
    vgap: float,
    hgap: float,
) -> Iterator[tuple[int,int,float]]:
    u, w = node
    
    for v in left_adj[u]:
        for x in right_adj[w]:
            yield (v, x, direct)
    # direct edges
    
    for v in left_adj[u]:
        yield (v, w, vgap)
    # vertical box edges
    
    for x in right_adj[w]:
        yield (u, x, hgap)
    # horizontal box edges

def ravel(
    i: int,
    j: int,
    n: int,
) -> int:
    return (i * n) + j

def unravel(
    k: int,
    n: int,
) -> tuple[int,int]:
    return (
        k // n,
        k % n,
    )

def _propagate_cost(
    left_adj: list[np.ndarray],
    right_adj: list[np.ndarray],
    right_order: int,
    matched_nodes: set[int],
    initial_conditions: list[tuple[float,int,int]],
    threshold: float,
    match: float,
    sub: float,
    vgap: float,
    hgap: float,
) -> tuple[
    nx.DiGraph,         # product topology on raveled nodes.
    dict[int,float],    # cost on raveled nodes
]:
    # print("\n_propagate")

    pq = []
    for entry_state in initial_conditions:
        heappush(pq, entry_state)
    # print(f"initial conditions:\n{[unravel(x, right_order) for (_,_,x) in initial_conditions]}")
    # construct priority queue from initial conditions
    
    prod = nx.DiGraph()
    cost = dict()
    # return types

    while len(pq) > 0:
        path_cost, prev_node, curr_node = heappop(pq)
        # if prev_node != curr_node:
            # print(f"{path_cost} {unravel(prev_node, right_order)} -> {unravel(curr_node, right_order)}")
        # else:
            # print(f"->{path_cost} {unravel(prev_node, right_order)}")

        if path_cost <= threshold:
            if not(curr_node in prod):
                # terminate paths that either
                # 1. exceed the threshold, or
                # 2. encounter a node already visited (by a cheaper path.)

                # node_index[curr_node] = n
                # sparse_cost.append(path_cost)
                # sparse_labels.append(curr_node)
                # n += 1
                prod.add_node(curr_node)
                cost[int(curr_node)] = path_cost
                # reached a new node; record its index, label, and cost.

                neighbors = strong_product_neighbors(left_adj, right_adj, unravel(curr_node, right_order), 0., vgap, hgap)
                # print(len(neighbors))
                for (l, r, edge_cost) in neighbors:
                    next_node = ravel(l, r, right_order)
                    node_cost = match if next_node in matched_nodes else sub
                    new_cost = path_cost + edge_cost + node_cost
                    # print("\t", (l, r), edge_cost, node_cost)
                    heappush(pq, (new_cost, curr_node, next_node))
                    # record the next step into the graph with cost determined by edge type and predetermined node costs

            if not(prev_node == curr_node):
                prod.add_edge(curr_node, prev_node)
                # sparse_edge_src.append(node_index[curr_node])
                # sparse_edge_tgt.append(node_index[prev_node])
        # finally, record the reversed edge curr_node -> prev_node in the sparse adjacency matrix
        # else:
            # print("terminated.")

    return (
        prod,
        cost,
    )

def propagate_cost(
    left: SpectrumGraph,
    left_sources: Iterator[int],
    right: SpectrumGraph,
    right_sources: Iterator[int],
    matched_nodes: list[tuple[int,int]],
    threshold: float,
    cost_model: tuple[float,float,float,float],
) -> WeightedProductGraph:
    right_order = right.order()
    # print("right order", right_order)
    # print("right num nodes", )
    matched_nodes = set([ravel(u, w, right_order) for (u,w) in matched_nodes])
    # print("sources",[(u,v) for u in left_sources for v in right_sources])
    product_sources = [ravel(u, w, right_order) for u in left_sources for w in right_sources]
    match, sub, vgap, hgap = cost_model
    initial_conditions = [(
        match if v in matched_nodes else sub,
        v,
        v,
    ) for v in product_sources]
    product_graph, cost = _propagate_cost(
        left.graph.adj,
        right.graph.adj,
        right_order,
        matched_nodes,
        initial_conditions,
        threshold,
        match,
        sub,
        vgap,
        hgap,
    )
    return WeightedProductGraph(
        graph = product_graph,
        right_operand_order = right_order,
        weights = cost,
    )
