import dataclasses, json
from time import time
from typing import Self, Any
# standard

from .util import ravel
from .io import serialize_dataclass, deserialize_dataclass, SerializableDataclass
from .graphs.types import SpectrumGraph, PivotGraph, WeightedProductGraph
from .graphs.propagate import propagate_cost

from .evaluation.costmodels import MatchedNodeCostModel, AnnotatedProductEdgeCostModel

from .annotation import AnnotationResult, AnnotationParams
# local

import numpy as np

@dataclasses.dataclass(slots=True)
class AlignmentResult(SerializableDataclass):
    prod_topology: list[WeightedProductGraph]
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
    verbose: bool = False,
) -> AlignmentResult:
    profile = {}

    t = time()
    n = len(anno)
    prod_topology = [None for _ in range(n)]
    for i in range(n):
        node_costmodel = MatchedNodeCostModel(
            (ravel(i, j, u.order()) for (i,j) in anno.symmetric_nodes[i]),
            params.node_match_cost,
            params.node_mismatch_cost,
        )
        edge_costmodel = AnnotatedProductEdgeCostModel(
            anno.lower_topology[i],
            anno.upper_topology[i],
            params.weight_key,
            anno.pairs,
            anno.lower_boundaries,
            anno.right_boundaries[i],
            params.edge_match_cost,
            params.edge_mismatch_cost,
            params.edge_gap_cost,
        )
        prod_topology[i] = propagate_cost(
            anno.lower_topology[i],
            anno.lower_topology[i].sources(),
            anno.upper_topology[i],
            anno.upper_topology[i].sources(),
            params.cost_threshold,
            node_costmodel,
            edge_costmodel,
        )
    profile["propagate"] = time() - t
    if verbose:
        pass

    if verbose:
       print(json.dumps(profile, indent=4))
    return AlignmentResult(
        prod_topology,
        profile,
    )
