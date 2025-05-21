from dataclasses import dataclass
from typing import Iterable, Callable, Any
from itertools import chain, pairwise
from math import floor, ceil
from abc import ABC, abstractmethod

from .graph_types import ProductDAG

CostFunction = Callable[[tuple, tuple], float]

@dataclass
class CostModel(ABC, Callable):
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

@dataclass
class FuzzyLocalCostModel(CostModel):
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

    def __init__(self, 
        score: float, 
        alignment: list[tuple[Any, Any]], 
        aligned_weights: list[tuple[Any, Any]],
        cost_model: CostModel,
    ):
        # private fields
        ## node sequence
        self._first_nodes, self._second_nodes = zip(*alignment)
        self._first_fragment = list(self._remove_none_and_stationary(self._first_nodes))
        self._second_fragment = list(self._remove_none_and_stationary(self._second_nodes))
        ## weight sequence
        self._first_weights, self._second_weights = map(list, zip(*aligned_weights))
        self._first_interval = self._not_none_interval(self._first_weights)
        self._second_interval = self._not_none_interval(self._second_weights)
        # public fields
        self.score = score
        self.alignment = alignment
        self.aligned_weights = aligned_weights
        self.cost_model = cost_model

        super(AlignedPath, self).__init__(alignment)
    
    def __repr__(self):
        header = f"\nscore: {self.score}\n"
        # construct the alignment representation
        first_nodes = []
        first_weights = []
        horizontal_separator = []
        second_weights = []
        second_nodes = []
        for (((first_node_left, second_node_left), (first_node_right, second_node_right)), (first_weight, second_weight)) in zip(pairwise(self.alignment), self.aligned_weights):
            if first_weight is None:
                first_nodes.append('...')
                first_weights.append('')
                horizontal_separator.append('')
                second_weights.append(str(second_weight))
                second_nodes.append(f"{second_node_left}-{second_node_right}")
            elif second_weight is None:
                first_nodes.append(f"{first_node_left}-{first_node_right}")
                first_weights.append(str(first_weight))
                horizontal_separator.append('')
                second_weights.append('')
                second_nodes.append('...')
            else:
                first_nodes.append(f"{first_node_left}-{first_node_right}")
                first_weights.append(str(first_weight))
                horizontal_separator.append('|')
                second_weights.append(str(second_weight))
                second_nodes.append(f"{second_node_left}-{second_node_right}")
        # left-justify the alignment representation 
        pad_len = max(map(
            len, 
            chain(first_nodes, first_weights, horizontal_separator, second_weights, second_nodes)))
        left_pad_len = ceil(pad_len / 2)
        fill_char = ' '
        first_nodes, second_nodes = map(
                lambda symbols: map(
                    lambda s: s.ljust(pad_len, fill_char),
                    symbols),
                [first_nodes, second_nodes])
        first_weights, horizontal_separator, second_weights = map(
                lambda symbols: map(
                    lambda s: s.rjust(left_pad_len, fill_char).ljust(pad_len, fill_char),
                    symbols),
                [first_weights, horizontal_separator, second_weights])
        # set the footer
        footer = '\n' + ((len(self) + 1) * pad_len * '-')
        # done 
        return header + '\n'.join(map(
            lambda symbols: ' '.join(symbols),
            [first_nodes, first_weights, horizontal_separator, second_weights, second_nodes]
        )) + footer
    
    def subscore(self, 
        left_sub_interval: tuple[int, int] = None,
        right_sub_interval: tuple[int, int] = None,
    ) -> float:
        l_none = left_sub_interval is None
        r_none = right_sub_interval is None
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
                    left_weight = self._first_weights[i]
                    right_weight = self._second_weights[i]
                    if left_weight == right_weight:
                        left_subscore += self.cost_model.match
                    elif left_weight is None or right_weight is None:
                        left_subscore += self.cost_model.gap
                    else:
                        left_subscore += self.cost_model.substitution
                return left_subscore
        elif not r_none:
            # identical to the preceding case, replacing `left` => `right`, `first` => `second`.
            right_interval = self.second_interval()
            if right_sub_interval[0] < right_interval[0] or right_sub_interval[1] > right_interval[1]:
                raise ValueError("subinterval out of bounds")
            else:
                right_subscore = 0.0
                for i in range(right_sub_interval[0], right_sub_interval[1] + 1):
                    # because the sub interval is contained within the interval,
                    # we are guaranteed to never encounter a 'skip' state,
                    # so we don't have to care about the parametization w.r.t.
                    # whatever product graph from which the alignment was made.
                    left_weight = self._first_weights[i]
                    right_weight = self._second_weights[i]
                    if left_weight == right_weight:
                        right_subscore += self.cost_model.match
                    elif left_weight is None or right_weight is None:
                        right_subscore += self.cost_model.gap
                    else:
                        right_subscore += self.cost_model.substitution
                return right_subscore

    def first_interval(self):
        return self._first_interval
    
    def second_interval(self):
        return self._second_interval

    def first_fragment(self):
        return self._first_fragment
    
    def first_weights(self):
        return self._first_weights
    
    def first_aligned_weights(self):
        return list(filter(lambda x: x is not None, self._first_weights))
    
    def second_fragment(self):
        return self._second_fragment

    def second_weights(self):
        return self._second_weights
    
    def second_aligned_weights(self):
        return list(filter(lambda x: x is not None, self._second_weights))

    def fragments(self):
        return (
            self.first_fragment(), 
            self.second_fragment())

