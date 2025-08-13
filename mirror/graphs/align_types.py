from dataclasses import dataclass
from typing import Iterator, Iterable, Callable, Any
from itertools import chain, pairwise
from math import floor, ceil
from abc import ABC, abstractmethod

from .graph_types import ProductDAG
from ..sequences.suffix_array import SuffixArray
from ..util import print_alignment

CostFunction = Callable[[tuple, tuple], float]

@dataclass
class AbstractCostModel(ABC, Callable):
    match: float = 0
    skip: float = 0.
    substitution: float = 1.
    gap: float = 2.

    @abstractmethod
    def __call__(
        self,
        product_graph: ProductDAG,
    ) -> CostFunction:
        """Return a cost function parametized by the product graph."""
        
@dataclass
class LocalCostModel(AbstractCostModel):
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

@dataclass
class FuzzyLocalCostModel(LocalCostModel):
    match: float = 0
    skip: float = 0.
    substitution: float = 1.
    gap: float = 2.
    tolerance: float = 0.01

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
                if abs(weight[0] - weight[1]) < self.tolerance:
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

class ProductPathWeightFilter(Callable[[Iterable[int]], bool]):
    def __init__(self,
        weight_sequence_filter: Callable[[list[Any]], bool],
        graph: ProductDAG,
    ):
        self._filter = weight_sequence_filter
        self._graph = graph
    
    def _weight_sequences(self, path_nodes: Iterable[int]):
        first_weights, second_weights = zip(*map(
            lambda edge: self._graph.weight_out(*edge),
            pairwise(path_nodes)))
        return first_weights, second_weights
    
    def __call__(self, path_nodes: Iterable[int]):
        path_nodes = list(path_nodes)
        if len(path_nodes) < 2:
            return True
        else:
            first_weights, second_weights = self._weight_sequences(path_nodes)
            first_filter = self._filter(first_weights)
            second_filter = self._filter(second_weights)
            return first_filter or second_filter

class AbstractAlignment(ABC, list[tuple[Any, Any]]):
    # edges
    @abstractmethod
    def edges(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Edge sequence in the product graph."""
    @abstractmethod
    def first_edges(self) -> list[tuple[int, int]]:
        """Edge sequence in the first graph, including stationary edges."""
    @abstractmethod
    def second_edges(self) -> list[tuple[int, int]]:
        """Edge sequence in the second graph, including stationary edges."""
    @abstractmethod
    def first_aligned_edges(self) -> list[tuple[int, int]]:
        """First edges without stationary edges."""
    @abstractmethod
    def second_aligned_edges(self) -> list[tuple[int, int]]:
        """Second edges without stationary edges."""
    # source and sink
    @abstractmethod
    def source(self) -> tuple[int, int]:
        """Origin vertex in the product graph."""
    @abstractmethod
    def first_source(self) -> tuple[int, int]:
        """Origin vertex in the first graph."""
    @abstractmethod
    def second_source(self) -> tuple[int, int]:
        """Origin vertex in the second graph."""
    @abstractmethod
    def target(self) -> tuple[int, int]:
        """Destination vertex in the product graph."""
    @abstractmethod
    def first_target(self) -> tuple[int, int]:
        """Destination vertex in the first graph."""
    @abstractmethod
    def second_target(self) -> tuple[int, int]:
        """Destination vertex in the second graph."""
    # weights
    @abstractmethod
    def weights(self) -> list[tuple[Any, Any]]:
        """Weight sequence in the product graph."""
    @abstractmethod
    def first_weights(self) -> list[Any]:
        """Weight sequence in the first graph, including None weights."""
    @abstractmethod
    def second_weights(self) -> list[Any]:
        """Weight sequence in the second graph, including None weights."""
    @abstractmethod
    def first_aligned_weights(self) -> list[Any]:
        """Second weights without None (≔ gap or skip) weights."""
    @abstractmethod
    def second_aligned_weights(self) -> list[Any]:
        """Second weights without None (≔ gap or skip) weights."""
    # score
    @abstractmethod
    def score(self) -> float:
        """The score of the alignment, determined by the cost model."""

class LocalAlignment(AbstractAlignment):

    @staticmethod
    def _not_none_interval(seq: list[Any]):
        is_none = list(map(
            lambda x: x is None,
            seq))
        reversed_is_none = is_none[::-1]
        return (
            is_none.index(False),
            len(seq) - 1 - reversed_is_none.index(False))

    @staticmethod
    def _remove_stationary(seq: Iterator) -> Iterator:
        prev_char = ''
        for char in seq:
            if char == prev_char:
                continue
            else:
                yield char
                prev_char = char
    
    @staticmethod
    def _remove_none(seq: Iterator) -> Iterator:
        return filter(
            lambda x: x is not None,
            seq)

    def __init__(self,
        score: float,
        alignment_nodes: list[tuple[Any, Any]],
        alignment_weights: list[tuple[Any, Any]],
        cost_model: LocalCostModel,
    ):
        # primary data
        self.cost_model = cost_model
        self.alignment_nodes = alignment_nodes
        self.alignment_weights = alignment_weights
        self._score = score
        # decomposed node sequence data
        self._first_nodes, self._second_nodes = zip(*alignment_nodes)
        self._first_component = list(self._remove_stationary(self._first_nodes))
        self._second_component = list(self._remove_stationary(self._second_nodes))
        # decomposed edge sequence data
        self._first_weights, self._second_weights = map(list, zip(*alignment_weights))
        self._first_aligned_weights = list(self._remove_none(self._first_weights))
        self._second_aligned_weights = list(self._remove_none(self._second_weights))
        self._first_interval = self._not_none_interval(self._first_weights)
        self._second_interval = self._not_none_interval(self._second_weights)
        # initialize as a list
        super(LocalAlignment, self).__init__(alignment_nodes)
    
    # nodes and intervals
    def first_component(self) -> list[Any]:
        return self._first_component

    def second_component(self) -> list[Any]:
        return self._second_component
    
    def components(self) -> tuple[list[Any], list[Any]]:
        return (
            self.first_component(),
            self.second_component())

    def first_interval(self) -> tuple[int, int]:
        """Position of first and last non-None weight in the first weight sequence."""
        return self._first_interval

    def second_interval(self) -> tuple[int, int]:
        """Position of first and last non-None weight in the second weight sequence."""
        return self._second_interval
    
    # edges
    def edges(self):
        return pairwise(self.alignment_nodes)

    def first_edges(self):
        return pairwise(self._first_nodes)

    def second_edges(self):
        return pairwise(self._second_nodes)

    def first_aligned_edges(self):
        return pairwise(self._first_component)

    def second_aligned_edges(self):
        return pairwise(self._second_component)
    
    # source and sink
    def first_source(self):
        return self._first_component[0]
    
    def second_source(self):
        return self._second_component[0]
    
    def source(self):
        return (self.first_source(), self.second_source())

    def first_target(self):
        return self._first_component[-1]
    
    def second_target(self):
        return self._second_component[-1]
    
    def target(self):
        return (self.first_target(), self.second_target())
    
    # weights
    def weights(self):
        return self.alignment_weights
    
    def first_weights(self):
        return self._first_weights
    
    def second_weights(self):
        return self._second_weights
    
    def first_aligned_weights(self):
        return self._first_aligned_weights
    
    def second_aligned_weights(self):
        return self._second_aligned_weights

    # score
    
    def score(self):
        """The score of the alignment, determined by the cost model."""
        return self._score

    def subscore(self,
        left_sub_interval: tuple[int, int] = None,
        right_sub_interval: tuple[int, int] = None,
    ) -> float:
        """The score of a sub-alignment created by restricting either the left or the right sub interval, but not both."""
        l_none = left_sub_interval is None
        r_none = right_sub_interval is None
        cost_model = self.cost_model
        first_weights = self.first_weights()
        second_weights = self.second_weights()
        # omg, so many error cases.
        if l_none and r_none:
            raise ValueError("neither `left_sub_interval`, `right_sub_interval` were passed.")
        elif not(l_none or r_none):
            raise ValueError("only one of `left_sub_interval`, `right_sub_interval` may be passed at a time.")
        elif not l_none:
            left_interval = self.first_interval()
            if left_sub_interval[0] < left_interval[0] or left_sub_interval[1] > left_interval[1]:
                raise ValueError("subinterval out of bounds")
            else:
                left_subscore = 0.0
                for i in range(left_sub_interval[0], left_sub_interval[1] + 1):
                    # because the sub interval is contained within the interval,
                    # we are guaranteed to never encounter a 'skip' state,
                    # so we don't have to care about the parametization w.r.t.
                    # whatever product graph from which the alignment was made.
                    left_weight = first_weights[i]
                    right_weight = second_weights[i]
                    if left_weight == right_weight:
                        left_subscore += cost_model.match
                    elif left_weight is None or right_weight is None:
                        left_subscore += cost_model.gap
                    else:
                        left_subscore += cost_model.substitution
                return left_subscore
        elif not r_none:
            # equivalent to the left case, replacing `left` => `right`, `first` => `second`.
            right_interval = self.second_interval()
            if right_sub_interval[0] < right_interval[0] or right_sub_interval[1] > right_interval[1]:
                raise ValueError("subinterval out of bounds")
            else:
                right_subscore = 0.0
                for i in range(right_sub_interval[0], right_sub_interval[1] + 1):
                    # as in the left case, we don't have to worry about skip states.
                    left_weight = first_weights[i]
                    right_weight = second_weights[i]
                    if left_weight == right_weight:
                        right_subscore += cost_model.match
                    elif left_weight is None or right_weight is None:
                        right_subscore += cost_model.gap
                    else:
                        right_subscore += cost_model.substitution
                return right_subscore
    
    def __repr__(self):
        return print_alignment(
            score = self.score(),
            alignment_edges = self.edges(),
            alignment_weights = self.weights(),
            name = "LocalAlignment")
