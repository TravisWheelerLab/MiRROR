from .graph_types import DAG
from .align_types import CostFunction
from typing import Callable
from heapq import heappop, heappush

def propagate(
    topology: DAG,
    cost: CostFunction,
    threshold: float,
    source: int,
) -> dict[int, float]:
    node_cost = dict()
    pq = [(
        0,      # score
        source, # current node
    )]
    while len(pq) > 0:
        current_cost, current_node = heappop(pq)
        if current_cost > threshold:
            break
        elif current_node in node_cost:
            continue
        else:
            node_cost[current_node] = current_cost
            for succeeding_node in topology.adj_out(current_node):
                edge = (current_node, succeeding_node)
                edge_weight = topology.weight_out(current_node, succeeding_node)
                succeeding_cost = current_cost + cost(edge, edge_weight)
                heappush(pq, (
                    succeeding_cost, 
                    succeeding_node,
                ))
    return node_cost