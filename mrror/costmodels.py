import abc
from typing import Union, Any, Iterator

import numpy as np

from .util import merge_compare_exact_unique, ravel, unravel
from .sequences.suffix_array import SuffixArray, BisectResult
from .fragments import TargetMassStateSpace, BoundaryResult, PairResult
from .graphs.types import SpectrumGraph, ProductEdgeWeight, WeightedProductGraph
from .graphs.propagate import AbstractNodeCostModel, AbstractEdgeCostModel
from .graphs.trace import AbstractPathCostModel

N_FEATURES = 4
FEATURE_WEIGHT_TENSOR = np.ones(N_FEATURES).reshape(1,1,N_FEATURES)

MISMATCH_SEPARATOR = '/'

class MatchedNodeCostModel(AbstractNodeCostModel):
    """Assigns costs to nodes as match or mismatch according to membership  in a set of matched nodes."""

    def __init__(
        self,
        matched_nodes: Iterator[int],
        node_match_cost: float,
        node_mismatch_cost: float,
    ):
        self._set = set(list(matched_nodes))
        self._match = node_match_cost
        self._mismatch = node_mismatch_cost

    def __call__(self, node: int) -> float:
        if node in self._set:
            return self._match
        else:
            return self._mismatch

class AnnotatedProductEdgeCostModel(AbstractEdgeCostModel):
    """Assigns costs to edges in an implicit product graph by comparing annotated features of component edges in the component graphs."""

    def __init__(
        self,
        left_graph: SpectrumGraph,
        right_graph: SpectrumGraph,
        weight_key: str,
        pairs: PairResult,
        left_boundaries: BoundaryResult,
        right_boundaries: BoundaryResult,
        edge_match_cost: float,
        edge_mismatch_cost: float,
        edge_gap_cost: float,
    ):
        self._left = left_graph
        self._right = right_graph
        self._right_order = right_graph.order()
        self._weight_key = weight_key
        self._pairs = pairs
        self._left_boundaries = left_boundaries
        self._right_boundaries = right_boundaries
        self._match = edge_match_cost
        self._mismatch = edge_mismatch_cost
        self._gap = edge_gap_cost

    def _retrieve_annotation(
        self,
        topology: SpectrumGraph,
        boundaries: BoundaryResult,
        curr_node: int,
        next_node: int,
    ) -> tuple[np.ndarray,np.ndarray]:
        if curr_node == next_node:
            return (
                np.array([0.,]),
                None,
            )
            # denote gaps with zero cost, None for residues and features.
            # (gap cost is added during comparison)
        else:
            idx = topology.graph[curr_node][next_node][self._weight_key]
            if curr_node == topology.boundary_source:
                idx, k = idx
                return boundaries.get_annotation(idx)
            else:
                return self._pairs.get_annotation(idx)

    def _compare_annotations(
        self,
        left_costs: list[np.ndarray],
        left_anno: list[np.ndarray],
        right_costs: list[np.ndarray],
        right_anno: list[np.ndarray],
        match: float,
        mismatch: float,
        gap: float,
        feature_weights: np.ndarray = FEATURE_WEIGHT_TENSOR,
        n_features: int = N_FEATURES,
    ) -> ProductEdgeWeight:
        if left_anno is None:
            return ProductEdgeWeight.from_left_gap(
                right_costs + gap,
                right_anno,
            )
            # LEFT GAP
        elif right_anno is None:
            return ProductEdgeWeight.from_right_gap(
                left_costs + gap,
                left_anno,
            )
            # RIGHT GAP
        else:
            n_left = left_costs.size
            n_right = right_costs.size
            left_anno_tensor = left_anno.reshape(1,*left_anno.shape)
            right_anno_tensor = right_anno.reshape(right_anno.shape[0],1,*right_anno.shape[1:])
            anno_eq = left_anno_tensor == right_anno_tensor
            anno_match_cost = match * (feature_weights * anno_eq).sum(axis=2)
            anno_neq = np.logical_not(anno_eq)
            anno_mismatch_cost = mismatch * (feature_weights * anno_neq).sum(axis=2)
            anno_complexity_cost = left_costs.reshape(1,n_left) + right_costs.reshape(n_right,1)
            comparison_costs = anno_complexity_cost + anno_mismatch_cost + anno_match_cost
            comparison_costs = comparison_costs.flatten()
            # construct the comparison tensors and calculate costs
            compared_left_anno = np.concat([left_anno] * n_right)
            compared_right_anno = np.empty((n_right * n_left,n_features),dtype=right_anno.dtype)
            for i in range(n_left):
                compared_right_anno[i::n_left] = right_anno
            # duplicate annotations. this is a terrible way to compute and store this data.
            return ProductEdgeWeight(
                comparison_costs,
                compared_left_anno,
                compared_right_anno,
            )
            # MATCH

    def __call__(self, curr_node: int, next_node: int) -> tuple[float,ProductEdgeWeight]:
        left_curr_node, right_curr_node = unravel(curr_node, self._right_order)
        left_next_node, right_next_node = unravel(next_node, self._right_order)
        left_anno = self._retrieve_annotation(self._left, self._left_boundaries, left_curr_node, left_next_node)
        right_anno = self._retrieve_annotation(self._right, self._right_boundaries, right_curr_node, right_next_node)
        edge_weight = self._compare_annotations(*left_anno, *right_anno, self._match, self._mismatch, self._gap)
        return (
            edge_weight.costs.min(),
            edge_weight,
        )

OrderedResiduePathState = list[tuple[np.ndarray,np.ndarray]] # each tuple weights an edge in the path with cost and residue annotation data.

class OrderedResiduePathCostModel(AbstractPathCostModel):
    """Associates paths in a WeightedProductGraph to an implicitly-constructed set of annotated sequences. Path cost delta is the edge cost. Path state is the sequence of lists of residues - ordered according to their granular cost - generated from edge annotations."""

    def __init__(
        self,
        product_graph: WeightedProductGraph,
        left_graph: SpectrumGraph,
        right_graph: SpectrumGraph,
        target_space: TargetMassStateSpace,
        mismatch_separator: str = MISMATCH_SEPARATOR,
    ):
        self._graph = product_graph
        self._targets = target_space 
        self._left_boundary = left_graph.boundary_source
        self._right_boundary = right_graph.boundary_source
        self._sep = mismatch_separator

    def _symbolize_annotation(
        self,
        anno: np.ndarray,
        node: int,
        boundary: int,
    ) -> np.ndarray:
        if anno is None:
            return np.array([''])
        elif np.all(anno == -1):
            return np.full_like(anno, '', dtype=str)
        elif node == boundary:
            return self._targets.symbolize_boundaries(anno)
        else:
            return self._targets.symbolize_pairs(anno)

    def _disjoint_sum_str(
        self,
        x: str,
        y: str,
    ):
        if x == y:
            return x
        else:
            return x + self._sep + y

    def _combine_symbols(
        self,
        left_symbols: np.ndarray,
        right_symbols: np.ndarray,
    ):
        assert left_symbols.shape == right_symbols.shape
        return np.array([
            self._disjoint_sum_str(x,y) for (x,y) in zip(
                left_symbols.flatten(),
                right_symbols.flatten(),
            )]).reshape(*left_symbols.shape)

    def initial_state(
        self,
        node: int,
    ) -> tuple[float,OrderedResiduePathState]:
        return (
            0.,
            [],
        )

    def next_state(
        self,
        curr_state: OrderedResiduePathState,
        curr_node: int,
        next_node: int,
    ) -> tuple[float,OrderedResiduePathState]:
        """Append to cost_state the symbolic data associated to the edge from curr_node to next_node. Returns the minimum cost of the edge and the cost state with the edge data appended."""
        left_node, right_node = self._graph.unravel(next_node)
        edge_annotation = self._graph.edge_weights[curr_node][next_node]
        anno_costs = edge_annotation.costs
        left_symbols = self._symbolize_annotation(edge_annotation.left_annotation, left_node, self._left_boundary)
        right_symbols = self._symbolize_annotation(edge_annotation.right_annotation, right_node, self._right_boundary)
        combined_symbols = self._combine_symbols(left_symbols, right_symbols)
        order = np.argsort(anno_costs)
        next_edge_state = (
                anno_costs[order],
                # (n,) array of floats
                combined_symbols[order],
                # (n,4) array of str
        )
        next_state = curr_state + [next_edge_state,]
        return (
            anno_costs.min(),
            next_state,
        )

    def state_type(self) -> type:
        return list

SuffixArrayPathState = tuple[
    list[BisectResult],         # one bisect result: one sequence.
    OrderedResiduePathState,    # per-position annotations. see above.
]

class SuffixArrayPathCostModel(OrderedResiduePathCostModel):
    """Filters out paths whose sequence class does not occur in the suffix array. The state is a collection of bisect results, which can be used to look up full sequences. Otherwise indistinguishable from EnumerationPathCostModel."""

    def __init__(
        self,
        suffix_array: SuffixArray,
        *args,
        **kwargs,
    ):
        self._suf_arr = suffix_array
        super(SuffixArrayPathCostModel, self).__init__(*args, **kwargs)

    def initial_state(
        self,
        node: int,
    ) -> tuple[float,OrderedResiduePathState]:
        return (
            [],
            super(SuffixArrayPathCostModel, self).initial_state(node)[1],
        )

    def next_state(
        self,
        curr_state: SuffixArrayPathState,
        curr_node: int,
        next_node: int,
    ) -> tuple[float,SuffixArrayPathState]:
        costs, annotation_symbols = super(SuffixArrayPathCostModel, self).next_state([], curr_node, next_node)[1]
        annotation_residues = annotation_symbols[:,0]
        # retrieve symbolic annotation for the edge
        curr_prefixes = curr_state[0]
        flat_residues, reindexer = np.unique_inverse(sum(
            (x.split(self._sep) for x in annotation_residues),
            [],
        ))
        bisect_results = self._suf_arr.bisect(
            flat_residues,
            prefix=curr_prefixes,
        )
        counts = np.array([x.count for x in bisect_results])
        occurring = counts > 0
        next_prefixes = bisect_results[occurring]
        # query the suffix array and mask non-occurring sequences.
        curr_annotation_sequence = curr_state[1]
        annotation_mask = occurring[reindexer]
        next_costs = costs[annotation_mask]
        next_annotation = annotation_symbols[annotation_mask]
        next_annotation_sequence = curr_annotation_sequence + [(next_costs,next_annotation),]
        # apply occurrence mask to symbolic annotation
        return (
            next_costs.min(),
            (
                next_prefixes,
                next_annotation_sequence,
            ),
        )

    @classmethod
    def state_type(self) -> type:
        return tuple
