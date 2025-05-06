from dataclasses import dataclass
from typing import Callable

from .graph_types import ProductDAG

CostFunction = Callable[[tuple, tuple], float]
CostModel = Callable[[ProductDAG], CostFunction]

@dataclass
class LocalCostModel(CostModel):
    match: float = 0
    skip: float = 0.
    substitution: float = 1.
    gap: float = 2.

    def __call__(
        self, 
        product_graph: ProductDAG,
    ) -> CostFunction:
        first_sources, second_sources = zip(
            *[product_graph.unravel(s) for s in product_graph.sources()])
        first_sinks, second_sinks = zip(
            *[product_graph.unravel(t) for t in product_graph.sinks()])
        first_extremities = set(first_sources + first_sinks)
        second_extremities = set(second_sources + second_sinks)
        def cost_function(edge: tuple, weight: tuple):
            first_edge, second_edge = zip(
                *[product_graph.unravel(v) for v in edge])
            is_first_stationary = first_edge[0] == first_edge[1]
            is_second_stationary = second_edge[0] == second_edge[1]
            if not (is_first_stationary or is_second_stationary):
                if weight[0] == weight[1]:
                    return self.match
                else:
                    return self.substitution
            elif is_first_stationary:
                if first_edge[0] in first_extremities:
                    return self.skip
                else:
                    return self.gap
            elif is_second_stationary:
                if second_edge[0] in second_extremities:
                    return self.skip
                else:
                    return self.gap
            else:
                raise ValueError(f"both edges are stationary! {edge} {weight}")
        return cost_function