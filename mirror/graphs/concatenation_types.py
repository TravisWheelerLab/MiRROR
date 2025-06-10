import itertools as it
import functools as ft
import operator as op
from typing import Iterator, Any, Callable

from .graph_types import DiGraph, DAG
from .align_types import AbstractAlignment, LocalAlignment#, ContinousCostModel
from .ensemble_types import EnsembleAlignment
from ..util import print_alignment


class DualIntervalDAG(DAG):

    def __init__(self,
        first_intervals: list[tuple[float, float]],
        second_intervals: list[tuple[float, float]],
        #continuous_cost_model: ContinousCostModel,
        cost_key = "cost",
    ):
        n = len(first_intervals)
        if n != len(second_intervals):
            raise ValueError("alignments, first_intervals, and second_intervals must all be the same length.")
        for (l, r) in first_intervals + second_intervals:
            if l >= r:
                raise ValueError("all intervals must take the form [l, r] where l < r.")
        dual_interval_succession_graph = DiGraph()
        for i in range(n):
            fi_outer = first_intervals[i]
            si_outer = second_intervals[i]
            for j in range(i + 1, n):
                fi_inner = first_intervals[j]
                si_inner = second_intervals[j]
                si_gap = si_outer[0] - si_inner[1]
                fi_gap = fi_inner[0] - fi_outer[1]
                min_gap = min(si_gap, fi_gap)
                if min_gap > 0:
                    dual_interval_succession_graph.add_edge(i, j) # edges are oriented outward to inward
                    dual_interval_succession_graph[i][j][cost_key] = min_gap#continous_cost_model.gap(min_gap)
        super(DualIntervalDAG, self).__init__(graph = dual_interval_succession_graph, weight_key = cost_key)

class ConcatenationAlignment(AbstractAlignment):

    @staticmethod
    def _concat_ensemble_edges(
        ensemble_sequence: list[EnsembleAlignment],
        get_edges: Callable,
        get_source: Callable,
        get_target: Callable,
    ) -> Iterator[tuple[Any, Any]]:
        edge_sequences = map(
            get_edges, 
            ensemble_sequence)
        edge_concatenators = map(
            lambda ensemble_pair: [(get_target(ensemble_pair[0]), get_source(ensemble_pair[1])),],
            it.pairwise(ensemble_sequence))
        return ft.reduce(
            op.concat,
            list(it.chain(*it.zip_longest(edge_sequences, edge_concatenators)))[:-1])

    @staticmethod
    def _concat_ensemble_weights(
        ensemble_sequence: list[EnsembleAlignment],
        get_weights: Callable,
        concatenation_weight_symbol: str,
    ):
        weight_sequences = map(get_weights, ensemble_sequence)
        weight_concatenators = [[concatenation_weight_symbol]] * (len(ensemble_sequence) - 1)
        return ft.reduce(
            op.concat,
            list(it.chain(*it.zip_longest(weight_sequences, weight_concatenators)))[:-1])

    def __init__(self,
        score: float,
        ensemble_sequence: list[EnsembleAlignment],
        concatenation_weight_symbol: str = "∘"
    ):
        self.concatenation_weight_symbol = concatenation_weight_symbol
        self.ensemble_sequence = ensemble_sequence
        self._score = score
        # concatenate ensemble data:
        ## join edge sequences with pseudo-edges (prev_target, next_source)
        self._left_edges = self._concat_ensemble_edges(
            ensemble_sequence = self.ensemble_sequence,
            get_edges = lambda ensemble: ensemble.first_edges(),
            get_source = lambda ensemble: ensemble.first_source(),
            get_target = lambda ensemble: ensemble.first_target())

        self._right_edges = self._concat_ensemble_edges(
            ensemble_sequence = self.ensemble_sequence,
            get_edges = lambda ensemble: ensemble.second_edges(),
            get_source = lambda ensemble: ensemble.second_source(),
            get_target = lambda ensemble: ensemble.second_target())
        ## join weight sequences with `concatenation_weight_symbol`
        self._left_weights = self._concat_ensemble_weights(
            ensemble_sequence = self.ensemble_sequence,
            get_weights = lambda ensemble: ensemble.first_weights(),
            concatenation_weight_symbol = concatenation_weight_symbol)

        self._right_weights = self._concat_ensemble_weights(
            ensemble_sequence = self.ensemble_sequence,
            get_weights = lambda ensemble: ensemble.second_weights(),
            concatenation_weight_symbol = concatenation_weight_symbol)
        
    # edges
    def first_edges(self) -> list[tuple[int, int]]:
        """Edge sequence in the first graph, including stationary edges."""
        return self._left_edges

    def second_edges(self) -> list[tuple[int, int]]:
        """Edge sequence in the second graph, including stationary edges."""
        return self._right_edges

    def edges(self) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Edge sequence in the product graph."""
        return list(zip(self.first_edges(), self.second_edges()))

    def first_aligned_edges(self):
        """First edges without stationary edges."""
        return list(filter(lambda x: x[0] != x[1], self.first_edges()))
    
    def second_aligned_edges(self):
        """Second edges without stationary edges."""
        return list(filter(lambda x: x[0] != x[1], self.second_edges()))

    # source and sink
    def source(self) -> tuple[int, int]:
        """Origin vertex in the product graph."""
        return self.ensemble_sequence[0].source()

    def first_source(self) -> tuple[int, int]:
        """Origin vertex in the first graph."""
        return self.ensemble_sequence[0].first_source()

    def second_source(self) -> tuple[int, int]:
        """Origin vertex in the second graph."""
        return self.ensemble_sequence[0].second_source()

    def target(self) -> tuple[int, int]:
        """Destination vertex in the product graph."""
        return self.ensemble_sequence[0].target()

    def first_target(self) -> tuple[int, int]:
        """Destination vertex in the first graph."""
        return self.ensemble_sequence[0].first_target()

    def second_target(self) -> tuple[int, int]:
        """Destination vertex in the second graph."""
        return self.ensemble_sequence[0].second_target()
    
    # weights
    def first_weights(self):
        """Weight sequence in the first graph, including None weights."""
        return self._left_weights

    def second_weights(self):
        """Weight sequence in the second graph, including None weights."""
        return self._right_weights

    def weights(self):
        """Weight sequence in the product graph."""
        return list(zip(self.first_weights(), self.second_weights()))
    
    def first_aligned_weights(self):
        """Second weights without None (≔ gap or skip) weights."""
        return list(filter(lambda x: x is not None, self.first_weights()))
    
    def second_aligned_weights(self):
        """Second weights without None (≔ gap or skip) weights."""
        return list(filter(lambda x: x is not None, self.second_weights()))
    
    #score
    def score(self):
        return self._score

    def __repr__(self):
        return print_alignment(
            score = self.score(),
            alignment_edges = self.edges(),
            alignment_weights = self.weights(),
            name = "ConcatenationAlignment")