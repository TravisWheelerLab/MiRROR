from .graph_types import DAG
from .align_types import CostFunction
from typing import Iterator, Callable
from collections import deque

def backtrace(
    topology: DAG,
    cost: CostFunction,
    node_cost: dict[int, float],
    threshold: float,
    source: int,
    sink: int,
    path_filter = lambda x: True,
) -> Iterator[list[int]]:
    initial_state = (
        sink,   # node
        0.,     # cost
        [],     # path
    )
    q = deque([initial_state])
    while len(q):
        current_node, path_cost, path_nodes = q.pop()
        if current_node in node_cost:
            current_cost = node_cost[current_node]
            minimum_potential_cost = current_cost + path_cost
            if minimum_potential_cost > threshold or not path_filter(list(reversed(path_nodes))):
                continue
            elif current_node == source:
                yield path_cost, list(reversed(path_nodes + [current_node]) )
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