import abc
from typing import Iterator, Any
from collections import deque

from .types import WeightedProductGraph

import numpy as np

class AbstractPathCostModel(abc.ABC):

    @abc.abstractmethod
    def initial_state(self, node: int) -> tuple[float,Any]:
        """Constructs the initial cost and cost state for a given node.
        
        arguments:
        - node: int, the index of a node in the graph.
        
        return:
        - initial_cost: float, typically zero or the cost of the given node.
        - initial_state: T, representing the path with one node."""

    @abc.abstractmethod
    def next_state(self, cost_state, curr_node: int, next_node: int) -> tuple[float,Any]:
        """Constructs the successor cost and cost state for a given cost state and edge.
        
        arguments:
        - cost_state: T, the state of the path.
        - curr_node: int, the last node in the path.
        - next_node: int, a node adjacent to curr_node.
        
        return:
        - delta_cost: float, the cost incurred by traversing the edge (curr_node, next_node) with cost_state.
        - next_state: T, the state of the new path including next_node."""

def _trace(
    adj,
    initial_states: list[tuple[float,list[int],Any,int]],
    threshold: float,
    cost_model: AbstractPathCostModel,
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
    threshold: float,
    cost_model: AbstractPathCostModel,
) -> list[tuple[float,list[int],Any]]:
    return list(_trace(
        prop_graph.graph.adj,
        [(*cost_model.initial_state(x), [], x) for x in sources],
        threshold,
        cost_model,
    ))
