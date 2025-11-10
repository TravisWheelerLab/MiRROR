import abc
import itertools as it
from typing import Iterator, Callable, Any
from heapq import heappush, heappop

from ..util import ravel, unravel
from .types import SpectrumGraph, WeightedProductGraph

import numpy as np
import networkx as nx

class AbstractNodeCostModel(abc.ABC):

    def __call__(self, node: int) -> float:
        """Calculates the cost of visiting a node."""

class AbstractEdgeCostModel(abc.ABC):

    def __call__(self, curr_node: int, next_node: int) -> float:
        """Calculates the cost of traversing an edge."""

def strong_product_neighbors(
    left_adj: list[list[int]],
    right_adj: list[list[int]],
    curr_left_node: int,
    curr_right_node: int,
) -> Iterator[tuple[int,int,float]]:
    for next_left_node in left_adj[curr_left_node]:
        for next_right_node in right_adj[curr_right_node]:
            yield (next_left_node, next_right_node)
    # direct edges
    for next_left_node in left_adj[curr_left_node]:
        yield (next_left_node, curr_right_node)
    # vertical box edges
    for next_right_node in right_adj[curr_right_node]:
        yield (curr_left_node, next_right_node)
    # horizontal box edges

def _propagate_cost(
    left_adj: list[np.ndarray],
    right_adj: list[np.ndarray],
    right_order: int,
    initial_conditions: list[tuple[float,Any,int,int]],
    threshold: float,
    node_cost_model: Callable[[int],float],
    edge_cost_model: Callable[[int,int],float],
) -> tuple[
    nx.DiGraph,         # product topology on raveled nodes.
    dict[int,float],    # cost on raveled nodes
]:
    pq = []
    for entry_state in initial_conditions:
        heappush(pq, entry_state)
    # construct priority queue from initial conditions
    prod = nx.DiGraph()
    cost = dict()
    comp = dict()
    # product topology, node costs, edge comparisons.
    while len(pq) > 0:
        curr_cost, edge_weight, prev_node, curr_node = heappop(pq)
        if prev_node is not None:
            prod.add_edge(curr_node, prev_node)
            comp[(curr_node,prev_node)] = edge_weight
            # record the reversed edge curr_node -> prev_node
        if curr_cost <= threshold:
            if not(curr_node in prod):
                # terminate paths that either
                # 1. exceed the threshold, or
                # 2. encounter a node already visited (by a cheaper path.)
                prod.add_node(curr_node)
                cost[int(curr_node)] = curr_cost
                # reached a new node; record its index, label, and cost.
                neighbors = strong_product_neighbors(left_adj, right_adj, *unravel(curr_node, right_order))
                for (l, r) in neighbors:
                    next_node = ravel(l, r, right_order)
                    node_cost = node_cost_model(next_node)
                    edge_cost, edge_weight = edge_cost_model(curr_node, next_node)
                    next_cost = curr_cost + edge_cost + node_cost
                    heappush(pq, (
                        next_cost,
                        edge_weight,
                        curr_node,
                        next_node,
                    ))
                    # record the next step into the graph with cost given by node and edge cost models.
    return (
        prod,
        cost,
        comp,
    )

def propagate_cost(
    left: SpectrumGraph,
    left_sources: Iterator[int],
    right: SpectrumGraph,
    right_sources: Iterator[int],
    threshold: float,
    node_cost_model: AbstractNodeCostModel,
    edge_cost_model: AbstractEdgeCostModel,
) -> WeightedProductGraph:
    right_order = right.order()
    product_sources = [ravel(u, w, right_order) for (u,w) in it.product(left_sources,right_sources)]
    initial_conditions = [(node_cost_model(v), None, None, v) for v in product_sources]
    sparse_product, propagated_costs, compared_edges = _propagate_cost(
        left.graph.adj,
        right.graph.adj,
        right_order,
        initial_conditions,
        threshold,
        node_cost_model,
        edge_cost_model,
    )
    return WeightedProductGraph(
        graph = sparse_product,
        right_operand_order = right_order,
        node_weights = propagated_costs,
        edge_weights = compared_edges,
    )
