import abc
from typing import Union, Any, Iterator

import numpy as np

from .util import merge_compare_exact_unique, ravel, unravel
from .sequences.suffix_array import SuffixArray, BisectResult
from .fragments import TargetMassStateSpace, BoundaryResult, PairResult
from .graphs.types import SpectrumGraph, ProductEdgeWeight, WeightedProductGraph
from .graphs.propagate import AbstractNodeCostModel, AbstractEdgeCostModel
from .graphs.trace import AbstractPathCostModel

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

N_FEATURES = 4
FEATURE_WEIGHT_TENSOR = np.ones(N_FEATURES).reshape(1,1,N_FEATURES)
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
                # costs, annotation = boundaries.get_annotation(idx)
            else:
                return self._pairs.get_annotation(idx)
                # costs, annotation = self._pairs.get_annotation(idx)
        #     # retrieve annotations 
        # residues = annotation[:,0]
        # features = annotation[:,1:]
        # # retrieve annotation data
        # unique_res, reindexer = np.unique_inverse(residues)
        # feature_clusters = [[] for _ in unique_res]
        # cost_clusters = [[] for _ in unique_res]
        # for (i,j) in enumerate(reindexer):
        #     feature_clusters[j].append(features[i])
        #     cost_clusters[j].append(costs[i])
        # # cluster features by residue
        # return (
        #     unique_res,
        #     [np.array(x) for x in cost_clusters],
        #     [np.hstack(x) for x in feature_clusters],
        # )

    def _compare_annotations(
        self,
        left_costs: list[np.ndarray],
        left_anno: list[np.ndarray],
        right_costs: list[np.ndarray],
        right_anno: list[np.ndarray],
        match: float,
        mismatch: float,
        gap: float,
        feature_weights: np.ndarray = FEATURE_WEIGHT_TENSOR, # TODO
        n_features: int = N_FEATURES,
    ) -> ProductEdgeWeight:
        print(left_costs, left_anno, right_costs, right_anno)
        if left_anno is None:
            return ProductEdgeWeight.from_left_gap(
                left_costs + gap,
                left_anno,
            )
            # LEFT GAP
        elif right_anno is None:
            return ProductEdgeWeight.from_right_gap(
                right_costs + gap,
                right_anno,
            )
            # RIGHT GAP
        else:
            n_left = left_costs.size
            n_right = right_costs.size
            left_anno_tensor = left_anno.reshape(1,*left_anno.shape)
            right_anno_tensor = right_anno.reshape(right_anno.shape[0],1,*right_anno.shape[1:])
            anno_eq = left_anno_tensor == right_anno_tensor
            anno_match_cost = match * (feature_weights * anno_eq).sum(axis=2)
            anno_neq = -1 * anno_eq
            anno_mismatch_cost = mismatch * (feature_weights * anno_neq).sum(axis=2)
            anno_complexity_cost = left_costs.reshape(1,n_left) + right_costs.reshape(n_right,1)
            comparison_costs = anno_complexity_cost + anno_mismatch_cost + anno_match_cost
            comparison_costs = comparison_costs.flatten()
            # construct the comparison tensors and calculate costs
            compared_right_anno = np.concat([right_anno] * n_left)
            compared_left_anno = np.empty((n_left * n_right,n_features),dtype=left_anno.dtype)
            for i in range(n_right):
                compared_left_anno[i::n_right] = left_anno
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

class EnumerationPathCostModel(AbstractPathCostModel):
    """Assigns cost and state to paths according to their edges' costs and features given by an edge weight dictionary, like the edge_weight field of a WeightedProductGraph."""

class SuffixArrayPathCostModel(EnumerationPathCostModel):
    """Filters out paths whose sequence class does not occur in the suffix array. The state is a collection of bisect results, which can be used to look up full sequences. Otherwise indistinguishable from EnumerationPathCostModel."""
