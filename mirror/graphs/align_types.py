from dataclasses import dataclass
from typing import Callable, Any
from itertools import chain, pairwise
from math import floor, ceil

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

    def __init__(self, score: float, alignment: list[tuple[Any, Any]], aligned_weights: list[tuple[Any, Any]]):
        # private fields
        ## node sequence
        self._first_nodes, self._second_nodes = zip(*alignment)
        self._first_fragment = list(self._remove_none_and_stationary(self._first_nodes))
        self._second_fragment = list(self._remove_none_and_stationary(self._second_nodes))
        ## weight sequence
        self._first_weights, self._second_weights = zip(*aligned_weights)
        self._first_interval = self._not_none_interval(self._first_weights)
        self._second_interval = self._not_none_interval(self._second_weights)
        # public fields
        self.score = score
        self.alignment = alignment
        self.aligned_weights = aligned_weights

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