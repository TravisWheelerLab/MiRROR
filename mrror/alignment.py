import dataclasses, json
from time import time
from typing import Self, Any
# standard

from .util import ravel
from .io import serialize_dataclass, deserialize_dataclass, SerializableDataclass
from .fragments import TargetMassStateSpace
from .graphs.types import SpectrumGraph, PivotGraph, WeightedProductGraph
from .graphs.propagate import propagate_cost

from .evaluation.spectrum_topology import construct_spectrum_topology
from .evaluation.costmodels import MatchedNodeCostModel, AnnotatedProductEdgeCostModel

from .annotation import AnnotationResult, AnnotationParams
# local

import numpy as np

@dataclasses.dataclass(slots=True)
class AlignmentResult(SerializableDataclass):
    fragment_masses: np.ndarray
    symmetries: list[np.ndarray]
    prod_topology: list[WeightedProductGraph]
    left_topology: list[SpectrumGraph]
    right_topology: list[SpectrumGraph]
    pivot_topology: list[PivotGraph]
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
    targets: TargetMassStateSpace,
    params: AlignmentParams,
    verbose: bool = False,
) -> AlignmentResult:
    profile = {}

    t = time()
    fragment_masses, symmetries, left_topology, right_topology, pivot_topology = construct_spectrum_topology(
        anno.peaks,
        anno.pairs,
        anno.left_boundaries,
        anno.pivots,
        anno.right_boundaries,
        anno.tolerance,
        params.weight_key,
    )
    profile["construct"] = time() - t
    if verbose:
        pass

    t = time()
    prod_topology = [propagate_cost(
            l,
            l.sources(),
            r,
            r.sources(),
            params.cost_threshold,
            MatchedNodeCostModel(
                (ravel(i, j, r.order()) for (i,j) in sym),
                params.node_match_cost,
                params.node_mismatch_cost,
            ),
            AnnotatedProductEdgeCostModel(
                l,
                r,
                params.weight_key,
                anno.pairs,
                anno.left_boundaries,
                rb,
                params.edge_match_cost,
                params.edge_mismatch_cost,
                params.edge_gap_cost,
            ),
        ) for (l,r,sym,rb) in zip(left_topology,right_topology,symmetries,anno.right_boundaries)]
    profile["propagate"] = time() - t
    if verbose:
        pass

    if verbose:
       print(json.dumps(profile, indent=4))
    return AlignmentResult(
        fragment_masses,
        symmetries,
        prod_topology,
        left_topology,
        right_topology,
        pivot_topology,
        profile,
    )
