import abc
from typing import Iterator, Any
from collections import deque

from .types import WeightedProductGraph, PathSpace

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

    @classmethod
    @abc.abstractmethod
    def state_type(self) -> type:
        """Returns the primitive type of the state, as expected by numpy's dtype argument as in `np.empty((n,),dtype=cost_model.state_type())`. Typically something like `list` or `tuple`, but in more complex implementations the state type could be custom class."""

def _trace(
    adj,
    initial_states: list[tuple[float,list[int],Any,int]],
    threshold: float,
    cost_model: AbstractPathCostModel,
) -> Iterator[tuple[float,Any,list[int]]]:
    q = deque(initial_states)
    while len(q) > 0:
        curr_cost, curr_cost_state, curr_path, curr_node = q.pop()
        if curr_cost < threshold:
            next_path = curr_path + [curr_node,]
            neighbors = list(adj[curr_node])
            if len(neighbors) == 0:
                yield (curr_cost, curr_cost_state, next_path)
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
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    trace_results = list(_trace(
            prop_graph.graph.adj,
            [(*cost_model.initial_state(x), [], x) for x in sources],
            threshold,
            cost_model,
    ))
    n_paths = len(trace_results)
    costs = np.empty((n_paths,), dtype=float)
    states = np.empty((n_paths,), dtype=cost_model.state_type())
    paths = np.empty((n_paths,), dtype=list)
    for (i,res) in enumerate(trace_results):
        costs[i] = res[0]
        states[i] = res[1]
        paths[i] = res[2]
    order = np.argsort(costs)
    costs = costs[order]
    states = states[order]
    paths = paths[order]
    return PathSpace(
        path = np.concat(paths),
        offset = np.cumsum([0,] + [len(x) for x in paths]),
        cost = costs,
        state = states,
    )
