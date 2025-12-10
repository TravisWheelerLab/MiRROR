import abc
from typing import Union, Any, Iterator, Self
import dataclasses

import numpy as np

from ..util import merge_compare_exact_unique, ravel, unravel, combine_symbols, combine_masses
from ..sequences.suffix_array import SuffixArray, BisectResult
from ..fragments import TargetMassStateSpace, BoundaryResult, PairResult
from ..graphs.types import SpectrumGraph, ProductEdgeWeight, WeightedProductGraph
from ..graphs.propagate import AbstractNodeCostModel, AbstractEdgeCostModel
from ..graphs.trace import AbstractPathCostModel

from .pathspaces import AnnotatedResiduePathState, AnnotatedResiduePathSpace, SuffixArrayPathState, SuffixArrayPathSpace

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
        lower_graph: SpectrumGraph,
        upper_graph: SpectrumGraph,
        weight_key: str,
        pairs: PairResult,
        lower_boundaries: BoundaryResult,
        upper_boundaries: BoundaryResult,
        edge_match_cost: float,
        edge_mismatch_cost: float,
        edge_gap_cost: float,
    ):
        self._left = lower_graph
        self._right = upper_graph
        self._upper_order = upper_graph.order()
        self._weight_key = weight_key
        self._pairs = pairs
        self._lower_boundaries = lower_boundaries
        self._upper_boundaries = upper_boundaries
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
        lower_costs: list[np.ndarray],
        lower_anno: list[np.ndarray],
        upper_costs: list[np.ndarray],
        upper_anno: list[np.ndarray],
        match: float,
        mismatch: float,
        gap: float,
        feature_weights: np.ndarray = FEATURE_WEIGHT_TENSOR,
        n_features: int = N_FEATURES,
    ) -> ProductEdgeWeight:
        if lower_anno is None:
            return ProductEdgeWeight.from_lower_gap(
                upper_costs + gap,
                upper_anno,
            )
            # LEFT GAP
        elif upper_anno is None:
            return ProductEdgeWeight.from_upper_gap(
                lower_costs + gap,
                lower_anno,
            )
            # RIGHT GAP
        else:
            n_left = lower_costs.size
            n_right = upper_costs.size
            lower_anno_tensor = lower_anno.reshape(1,*lower_anno.shape)
            upper_anno_tensor = upper_anno.reshape(upper_anno.shape[0],1,*upper_anno.shape[1:])
            anno_eq = lower_anno_tensor == upper_anno_tensor
            anno_match_cost = match * (feature_weights * anno_eq).sum(axis=2)
            anno_neq = np.logical_not(anno_eq)
            anno_mismatch_cost = mismatch * (feature_weights * anno_neq).sum(axis=2)
            anno_complexity_cost = lower_costs.reshape(1,n_left) + upper_costs.reshape(n_right,1)
            comparison_costs = anno_complexity_cost + anno_mismatch_cost + anno_match_cost
            comparison_costs = comparison_costs.flatten()
            # construct the comparison tensors and calculate costs
            compared_lower_anno = np.concat([lower_anno] * n_right)
            compared_upper_anno = np.empty((n_right * n_left,n_features),dtype=upper_anno.dtype)
            for i in range(n_left):
                compared_upper_anno[i::n_left] = upper_anno
            # duplicate annotations. this is a terrible way to compute and store this data.
            return ProductEdgeWeight(
                comparison_costs,
                compared_lower_anno,
                compared_upper_anno,
            )
            # MATCH

    def __call__(self, curr_node: int, next_node: int) -> tuple[float,ProductEdgeWeight]:
        lower_curr_node, upper_curr_node = unravel(curr_node, self._upper_order)
        lower_next_node, upper_next_node = unravel(next_node, self._upper_order)
        lower_anno = self._retrieve_annotation(self._left, self._lower_boundaries, lower_curr_node, lower_next_node)
        upper_anno = self._retrieve_annotation(self._right, self._upper_boundaries, upper_curr_node, upper_next_node)
        edge_weight = self._compare_annotations(*lower_anno, *upper_anno, self._match, self._mismatch, self._gap)
        return (
            edge_weight.costs.min(),
            edge_weight,
        )

class AnnotatedResiduePathCostModel(AbstractPathCostModel):
    """Associates paths in a WeightedProductGraph to an implicitly-constructed set of annotated sequences. Path cost delta is the edge cost. Path state is the sequence of lists of residues - ordered according to their granular cost - generated from edge annotations."""

    _PATH_SPACE = AnnotatedResiduePathSpace

    def __init__(
        self,
        pivot: float,
        product_graph: WeightedProductGraph,
        lower_graph: SpectrumGraph,
        upper_graph: SpectrumGraph,
        target_space: TargetMassStateSpace,
        mismatch_separator: str = MISMATCH_SEPARATOR,
        mass_offset_threshold: float = 0.01, # 1% is very permissive. could be much smaller if the input data is high resolution.
    ):
        self._mass_threshold = (2 + mass_offset_threshold) * pivot
        self._graph = product_graph
        self._targets = target_space 
        self._lower_boundary = lower_graph.boundary_source
        self._upper_boundary = upper_graph.boundary_source
        self._sep = mismatch_separator

    def _symbolize_and_weigh_annotation(
        self,
        anno: np.ndarray,
        node: int,
        boundary: int,
    ) -> tuple[np.ndarray,np.ndarray]:
        if anno is None:
            return (
                np.array([['',],], dtype=str),
                np.array([[0.,],], dtype=float),
            )
        elif np.all(anno == -1):
            return (
                np.full_like(anno, '', dtype=str),
                np.full_like(anno, 0., dtype=float),
            )
        elif node == boundary:
            return (
                self._targets.symbolize_boundaries(anno),
                self._targets.weigh_boundaries(anno),
            )
        else:
            return (
                self._targets.symbolize_pairs(anno),
                self._targets.weigh_pairs(anno),
            )

    def initial_state(
        self,
        node: int,
    ) -> tuple[float,AnnotatedResiduePathState]:
        return (
            0.,
            [],
        )

    def next_state(
        self,
        curr_state: AnnotatedResiduePathState,
        curr_node: int,
        next_node: int,
    ) -> tuple[float,AnnotatedResiduePathState]:
        """Append to cost_state the symbolic and mass data associated to the edge from curr_node to next_node. Returns the minimum cost of the edge and the cost state with the edge data appended."""
        lower_node, upper_node = self._graph.unravel(next_node)
        edge_annotation = self._graph.edge_weights[curr_node][next_node]
        anno_costs = edge_annotation.costs
        lower_symbols, lower_masses = self._symbolize_and_weigh_annotation(edge_annotation.lower_annotation, lower_node, self._lower_boundary)
        upper_symbols, upper_masses = self._symbolize_and_weigh_annotation(edge_annotation.upper_annotation, upper_node, self._upper_boundary)
        combined_symbols = combine_symbols(lower_symbols, upper_symbols, self._sep)
        combined_mass = combine_masses(lower_masses, upper_masses)
        order = np.argsort(anno_costs)
        next_edge_state = (
                anno_costs[order],
                # (n,) array of floats
                combined_symbols[order],
                # (n,4) array of str
                combined_mass[order],
                # (n,2) array of left and right masses.
        )
        next_state = curr_state + [next_edge_state,]
        return (
            anno_costs.min(),
            next_state,
        )

    def path_space_type(self) -> type:
        return self._PATH_SPACE

class SuffixArrayPathCostModel(AnnotatedResiduePathCostModel):
    """Filters out paths whose sequence class does not occur in the suffix array. The state is a collection of bisect results, which can be used to look up full sequences. Otherwise indistinguishable from EnumerationPathCostModel."""

    _PATH_SPACE = SuffixArrayPathSpace

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
    ) -> tuple[float,AnnotatedResiduePathState]:
        init_cost, init_state = super(SuffixArrayPathCostModel, self).initial_state(node)
        return (
            init_cost,
            (
                [],
                init_state,
            ),
        )

    def next_state(
        self,
        curr_state: SuffixArrayPathState,
        curr_node: int,
        next_node: int,
    ) -> tuple[float,SuffixArrayPathState]:
        costs, annotation_symbols, annotation_masses = super(SuffixArrayPathCostModel, self).next_state([], curr_node, next_node)[1][0]
        annotation_residues = annotation_symbols[:,0]
        # retrieve symbolic annotation for the edge

        curr_prefixes = curr_state[0]
        if len(curr_prefixes) == 0:
            curr_prefixes = [None,]
        split_residues = []
        split_indices = []
        for (i,x) in enumerate(annotation_residues):
            anno_res = x.split(self._sep)
            split_residues.extend(anno_res)
            split_indices.extend([i,] * len(anno_res))
        split_indices = np.array(split_indices)
        split_costs = costs[split_indices]
        split_annotation_symbols = annotation_symbols[split_indices]
        split_annotation_masses = annotation_masses[split_indices]
        # unpack residues and reshape annotations to accomodate the extra entries corresponding to each mismatch residue; an A/A needs only one entry, which matches the costs and annotation_symbols shape, but an A/T for ex. needs two entries, which requires that row of the costs and annotation_symbols.

        flat_residues, reindexer = np.unique_inverse(split_residues)
        bisect_results = np.array(
            self._suf_arr.bisect(
                flat_residues,
                prefix=curr_prefixes,
            ),
            dtype = BisectResult,
        )
        counts = np.array([x.count for x in bisect_results])
        occurring = counts > 0
        next_prefixes = bisect_results[occurring]
        # query the suffix array and mask non-occurring sequences.

        annotation_mask = occurring[reindexer]
        next_costs = split_costs[annotation_mask]
        next_annotation_symbols = split_annotation_symbols[annotation_mask]
        next_annotation_masses = split_annotation_masses[annotation_mask]
        # apply occurrence mask to symbolic annotation

        return (
            next_costs.min() if len(next_costs) > 0 else np.inf,
            (
                next_prefixes,
                curr_state[1] + [(next_costs,next_annotation_symbols,next_annotation_masses),],
            ),
        )

    @classmethod
    def state_type(self) -> type:
        return tuple
