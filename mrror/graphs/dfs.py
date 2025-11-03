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
        prev_cost, curr_node, prev_path = q.pop()
        curr_cost = prev_cost + cost[curr_node]
        curr_path = prev_path + [curr_node,]
        if curr_cost <= threshold:
            # terminate paths that exceed the threshold
            degree = len(adj[curr_node])
            if degree == 0:
                yield (curr_cost, curr_path)
                # yield paths that reach sinks
            else:
                for next_node in adj[curr_node]:
                    q.append((
                        curr_cost,
                        next_node.item(),
                        curr_path,
                    ))

def dfs(
    topology: WeightedProductGraph,
    sources: Iterator[int],
    threshold: float,
    filter = None,
) -> list[tuple]:
    if filter:
        return []
    else:
        return list(_dfs(
            topology.graph.adj,
            topology.node_weights,
            threshold,
            [(0., x, []) for x in sources],
        ))
