from .graph_types import DAG
from typing import Iterator, Callable
from collections import deque

def backtrace(
    topology: DAG,
    cost: Callable,
    node_cost: dict[int, float],
    threshold: float,
    source: int,
    sink: int,
) -> Iterator[list[int]]:
    initial_state = (
        sink,   # node
        0.,     # score
        [],     # path
    )
    q = deque([initial_state])
    while len(q):
        current_node, path_cost, path_nodes = q.pop()
        if current_node in node_cost:
            current_cost = node_cost[current_node]
            minimum_potential_score = current_cost + path_cost
            if minimum_potential_score > threshold:
                continue
            elif current_node == source:
                yield path_cost, path_nodes + [current_node]
            else:
                for preceeding_node in topology.adj_in(current_node):
                    edge = (current_node, preceeding_node)
                    edge_weight = topology.weight_in(current_node, preceeding_node)
                    preceeding_cost = path_cost + cost(edge, edge_weight)
                    q.append((
                        preceeding_node,
                        preceeding_cost,
                        path_nodes + [current_node],
                    ))