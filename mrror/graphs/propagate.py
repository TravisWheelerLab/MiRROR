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
    lower_adj: list[list[int]],
    upper_adj: list[list[int]],
    curr_lower_node: int,
    curr_upper_node: int,
) -> Iterator[tuple[int,int,float]]:
    for next_lower_node in lower_adj[curr_lower_node]:
        for next_upper_node in upper_adj[curr_upper_node]:
            yield (next_lower_node, next_upper_node)
    # direct edges
    for next_lower_node in lower_adj[curr_lower_node]:
        yield (next_lower_node, curr_upper_node)
    # vertical box edges
    for next_upper_node in upper_adj[curr_upper_node]:
        yield (curr_lower_node, next_upper_node)
    # horizontal box edges

def _propagate_cost(
    lower_adj: list[np.ndarray],
    upper_adj: list[np.ndarray],
    upper_order: int,
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
        if curr_cost <= threshold:
            if not(curr_node in prod):
                # terminate paths that either
                # 1. exceed the threshold, or
                # 2. encounter a node already visited (by a cheaper path.)
                prod.add_node(curr_node)
                cost[int(curr_node)] = curr_cost
                # reached a new node; record its index, label, and cost.
                neighbors = list(strong_product_neighbors(lower_adj, upper_adj, *unravel(curr_node, upper_order)))
                for (l, r) in neighbors:
                    next_node = ravel(l, r, upper_order)
                    node_cost = node_cost_model(next_node)
                    edge_cost, next_edge_weight = edge_cost_model(curr_node, next_node)
                    next_cost = curr_cost + edge_cost + node_cost
                    heappush(pq, (
                        next_cost,
                        next_edge_weight,
                        curr_node,
                        next_node,
                    ))
                    # record the next step into the graph with cost given by node and edge cost models.
            if prev_node is not None:
                prod.add_edge(curr_node, prev_node)
                if not(curr_node in comp):
                    comp[int(curr_node)] = dict()
                comp[int(curr_node)][int(prev_node)] = edge_weight
                # record the reversed edge curr_node -> prev_node
    return (
        prod,
        cost,
        comp,
    )

def propagate_cost(
    left: SpectrumGraph,
    lower_sources: Iterator[int],
    right: SpectrumGraph,
    upper_sources: Iterator[int],
    threshold: float,
    node_cost_model: AbstractNodeCostModel,
    edge_cost_model: AbstractEdgeCostModel,
) -> WeightedProductGraph:
    upper_order = int(right.order())
    product_sources = [ravel(u, w, upper_order) for (u,w) in it.product(lower_sources,upper_sources)]
    initial_conditions = [(node_cost_model(v), None, None, v) for v in product_sources]
    sparse_product, propagated_costs, compared_edges = _propagate_cost(
        left.graph.adj,
        right.graph.adj,
        upper_order,
        initial_conditions,
        threshold,
        node_cost_model,
        edge_cost_model,
    )
    return WeightedProductGraph(
        graph = sparse_product,
        upper_operand_order = upper_order,
        node_weights = propagated_costs,
        edge_weights = compared_edges,
    )
