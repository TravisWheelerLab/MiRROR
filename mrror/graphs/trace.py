from typing import Iterator, Any
from collections import deque

from .types import WeightedProductGraph
from ..costmodel import AbstractPathCostModel

import numpy as np

def _trace(
    adj,
    node_cost: dict[int,float],
    initial_states: list[tuple[float,list[int],Any,int]],
    cost_model: PathCostModel,
) -> Iterator[tuple[float,list[int],Any]]:
    q = deque(initial_states)
    while len(q) > 0:
        curr_cost, curr_cost_state, curr_path, curr_node = q.pop()
        if curr_cost > threshold:
            continue
        else:
            next_path = curr_path + [curr_node,]
            neighbors = list(adj[curr_node])
            if len(neighbors) == 0:
                yield (curr_cost, curr_cost_state, next_path)
            else:
                for next_node in neighbors:
                    delta_cost, next_cost_state = cost_model.next_state(curr_cost_state, curr_node, next_node)
                    next_cost = max(curr_cost + delta_cost, node_cost[next_node])

def trace(
    prop_graph: WeightedProductGraph,
    sources: Iterator[int],
    cost_model: AbstractPathCostModel,
    threshold: float,
) -> list[tuple[float,list[int],Any]]:
    """"""
    adj = prop_graph.graph.adj
    node_cost = prop_graph.node_weights
    initial_states = [
        (*cost_model.initial_state(x), [], x)
        for x in sources
    ]
    return _trace(
        adj,
        node_cost,
        initial_states,
        cost_model,
        threshold,
    )
