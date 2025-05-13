from dataclasses import dataclass
from typing import Callable, Any

from .graph_types import ProductDAG

class AlignedPath(list[tuple[Any, Any]]):
    @classmethod
    def _not_none_interval(cls, seq: list[Any]):
        """very naiive implementation to find the interval of non-none values.
        not optimized for long sequences!"""
        none_seq = [s is None for s in seq]
        return none_seq.index(False), len(seq) - 1 - none_seq[::-1].index(False)
    
    @classmethod
    def _remove_none_and_stationary(cls, seq: list[Any]):
        """also naiive and unoptimized. does what the name says. removes None, and removes repeated values."""
        prev_char = ''
        for char in seq:
            if (char == None) or (char == prev_char):
                continue
            else:
                yield char
            prev_char = char

    def __init__(self, score: float, alignment: list[tuple[Any, Any]]):
        # private fields
        self._first_sequence, self._second_sequence = zip(*alignment)
        self._first_interval = self._not_none_interval(self._first_sequence)
        self._second_interval = self._not_none_interval(self._second_sequence)
        self._first_fragment = list(self._remove_none_and_stationary(self._first_sequence))
        self._second_fragment = list(self._remove_none_and_stationary(self._second_sequence))
        # public fields
        self.score = score
        self.alignment = alignment

        super(AlignedPath, self).__init__(alignment)
    
    def first_interval(self):
        return self._first_interval
    
    def second_interval(self):
        return self._second_interval

    def first_fragment(self):
        return self._first_fragment
    
    def second_fragment(self):
        return self._second_fragment

    def fragments(self):
        return (
            self.first_fragment(), 
            self.second_fragment())


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