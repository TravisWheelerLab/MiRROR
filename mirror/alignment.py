import dataclasses, json
from time import time
from typing import Self, Any
# standard

from .util import ravel
from .io import serialize_dataclass, deserialize_dataclass, SerializableDataclass
from .sequences.suffix_array import SuffixArray
from .fragments.types import ResidueStateSpace
from .graphs.types import AlignedPaths, AugmentedLetter
from .graphs.align import align_spectrum_graphs
from .evaluation.costmodels import SymmetricNodeCostModel, AnnotatedEdgeCostModel, MassConstrainedPathCostModel, SuffixArrayPathCostModel

from .annotation import AnnotationResult, AnnotationParams
# local

import numpy as np

@dataclasses.dataclass(slots=True)
class AlignmentResult(SerializableDataclass):
    aligned_prefix_paths: list[AlignedPaths]
    aligned_suffix_paths: list[AlignedPaths]
    _profile: dict[str, float]

    def __len__(self) -> int:
        return len(self.prod_topology)

@dataclasses.dataclass(slots=True)
class AlignmentParams(SerializableDataclass):
    weight_key: str
    cost_threshold: float
    node_match_cost: float
    node_mismatch_cost: float
    edge_match_cost: float
    edge_mismatch_cost: float
    edge_gap_cost: float

    @classmethod
    def from_config(cls, cfg):
        cost_cfg = cfg['cost_model']
        return cls(
            cost_threshold = cfg['cost_threshold'],
            weight_key = cfg['weight_key'],
            node_match_cost = cost_cfg['node_match'],
            node_mismatch_cost = cost_cfg['node_mismatch'],
            edge_match_cost = cost_cfg['edge_match'],
            edge_mismatch_cost = cost_cfg['edge_mismatch'],
            edge_gap_cost = cost_cfg['edge_gap'],
        )

def align(
    anno: AnnotationResult,
    params: AlignmentParams,
    augmented_alphabet: list[AugmentedLetter],
    forward_suffix_array: SuffixArray,
    reverse_suffix_array: SuffixArray,
    residue_space: ResidueStateSpace,
    verbose: bool = False,
) -> AlignmentResult:
    profile = {}
    edge_cost = AnnotatedEdgeCostModel.from_annotation(
        anno.annotation_index,
        residue_space,
        params.edge_mismatch_cost,
        params.edge_gap_cost,
    )
    t = time()
    n = len(anno)
    aligned_prefix_paths = [None for _ in range(n)]
    aligned_suffix_paths = [None for _ in range(n)]
    for i in range(n):
        lower_graph, upper_graph, pivot_graph, symmetric_graph, node_cost, path_cost = anno.spectrum_topology[i]
        forward_path_cost = SuffixArrayPathCostModel.from_mass_constraint(
            path_cost,
            residue_space,
            forward_suffix_array,
        )
        aligned_prefix_paths[i] = align_spectrum_graphs(
            lower_graph,
            upper_graph,
            lower_graph.boundary_node,
            upper_graph.boundary_node,
            node_cost,
            edge_cost,
            forward_path_cost,
            anno.node_lookup,
            anno.node_lookup,
            augmented_alphabet,
            threshold = 5, # placeholder, pending dynamic threshold.
        )
        reverse_path_cost = SuffixArrayPathCostModel.from_mass_constraint(
            path_cost,
            residue_space,
            reverse_suffix_array,
        )
        aligned_suffix_paths[i] = align_spectrum_graphs(
            lower_graph,
            upper_graph,
            lower_graph.boundary_node,
            upper_graph.boundary_node,
            node_cost,
            edge_cost,
            reverse_path_cost,
            anno.node_lookup,
            anno.node_lookup,
            augmented_alphabet,
            threshold = 5, # placeholder, pending dynamic threshold.
        )
    profile["align"] = time() - t

    if verbose:
       print(json.dumps(profile, indent=4))
    return AlignmentResult(
        aligned_prefix_paths,
        aligned_suffix_paths,
        profile,
    )
