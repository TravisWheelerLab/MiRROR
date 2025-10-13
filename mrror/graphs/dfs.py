from typing import Iterator, Callable
from collections import deque

from .types import WeightedProductGraph

import numpy as np

def _dfs(
    adj: list[np.ndarray],
    cost: dict[int,float],
    threshold: float,
    initial_states: list[tuple[float,int,list[int]]],
) -> Iterator[tuple[float,list[int]]]:
    q = deque(initial_states)
    while len(q) > 0:
        path_cost, current_node, path_nodes = q.pop()
        current_cost = path_cost + cost[current_node]
        if current_cost <= threshold:
            # terminate paths that exceed the threshold
            degree = len(adj[current_node])
            if degree == 0:
                yield (current_cost, path_nodes + [current_node])
                # yield paths that reach sinks
            else:
                for next_node in adj[current_node]:
                    q.append((
                        current_cost,
                        next_node.item(),
                        path_nodes + [current_node],
                    ))

def _filtered_dfs(*args, **kwargs):
    pass

def dfs(
    topology: WeightedProductGraph,
    sources: Iterator[int],
    threshold: float,
    filter: Callable = None,
) -> list[list[int]]:
    if filter:
        return _filtered_dfs(
        )
    else:
        return list(_dfs(
            adj = topology.graph.adj,
            cost = topology.weights,
            threshold = threshold,
            initial_states = [(0., x, []) for x in sources],
        ))
