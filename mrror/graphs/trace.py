from typing import Iterator, Any
from collections import deque

from .types import WeightedProductGraph
from ..costmodels import AbstractPathCostModel

import numpy as np

def _trace(
    adj,
    initial_states: list[tuple[float,list[int],Any,int]],
    cost_model: AbstractPathCostModel,
    threshold: float,
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
                yield ((curr_cost, curr_cost_state), next_path)
            else:
                for next_node in neighbors:
                    delta_cost, next_cost_state = cost_model.next_state(curr_cost_state, curr_node, next_node)
                    next_cost = curr_cost + delta_cost
                    q.append((next_cost, next_cost_state, next_path, next_node))

def trace(
    prop_graph: WeightedProductGraph,
    sources: Iterator[int],
    cost_model: AbstractPathCostModel,
    threshold: float,
) -> list[tuple[float,list[int],Any]]:
    """"""
    adj = prop_graph.graph.adj
    initial_states = [
        (*cost_model.initial_state(x), [], x)
        for x in sources
    ]
    return list(_trace(
        adj,
        initial_states,
        cost_model,
        threshold,
    ))
