import dataclasses, json
from time import time
from typing import Self, Any
# standard

from .fragments.types import TargetMassStateSpace
from .graphs.types import SpectrumGraph, PivotGraph, WeightedProductGraph
from .graphs.spectrum_topology import construct_spectrum_topology
from .graphs.propagate import propagate_cost
from .annotation import AnnotationResult, AnnotationParams
# local

import numpy as np

@dataclasses.dataclass(slots=True)
class AlignmentResult:
    fragment_masses: np.ndarray
    symmetries: list[np.ndarray]
    sparse_prod: list[WeightedProductGraph]
    lo_adj: list[SpectrumGraph]
    hi_adj: list[SpectrumGraph]
    pivot_adj: list[PivotGraph]
    _profile: dict[str, float]

@dataclasses.dataclass(slots=True)
class AlignmentParams:
    cost_threshold: float
    cost_model: tuple[float,float,float,float]

    @classmethod
    def from_config(cls, cfg):
        return cls(
            cost_threshold = cfg['cost_threshold'],
            cost_model = list(cfg['cost_model'].values()),
        )

def align(
    anno: AnnotationResult,
    targets: TargetMassStateSpace,
    params: AlignmentParams,
    verbose: bool = False,
) -> AlignmentResult:
    profile = {}

    t = time()
    fragment_masses, symmetries, lo_adj, hi_adj, pivot_adj = construct_spectrum_topology(
        anno.peaks,
        anno.pairs,
        anno.left_boundaries,
        anno.pivots,
        anno.right_boundaries,
    )
    profile["construct"] = time() - t
    if verbose:
        print("spectrum topology")
        print(f"{fragment_masses}\n")
        for (i,(sym, lo, hi, pivot)) in enumerate(zip(symmetries,lo_adj,hi_adj,pivot_adj)):
            print(f"sym[{i}]:\n\t{sym}")
            print(f"lo[{i}]:\n\t{lo.graph.edges(data=True)}")
            print(f"hi[{i}]:\n\t{hi.graph.edges(data=True)}")
            print(f"pivot[{i}]:\n\t{pivot.graph.edges(data=True)}")

    t = time()
    print([[x for x in lo.graph if lo.graph.in_degree(x) == 0] for lo in lo_adj])
    print([[x for x in hi.graph if hi.graph.in_degree(x) == 0] for hi in hi_adj])
    product_graphs = [propagate_cost(
        left = lo,
        left_sources = [x for x in lo.graph if lo.graph.in_degree(x) == 0], # [lo.boundary_source,]
        right = hi,
        right_sources = [x for x in hi.graph if hi.graph.in_degree(x) == 0], # [hi.boundary_source,]
        matched_nodes = sym,
        threshold = params.cost_threshold,
        cost_model = params.cost_model,
    ) for (lo,hi,sym) in zip(lo_adj, hi_adj, symmetries)]
    profile["propagate"] = time() - t
    if verbose:
        print("propagate")
        for (i,prod) in enumerate(product_graphs):
            print(f"prod[{i}]:\n\t{prod.graph.edges}\n\t{prod.weights}")

    if verbose:
       print(json.dumps(profile, indent=4))
    return AlignmentResult(
        fragment_masses,
        symmetries,
        product_graphs,
        lo_adj,
        hi_adj,
        pivot_adj,
        profile,
    )
