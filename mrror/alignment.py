import dataclasses, json
from time import time
from typing import Self, Any
# standard

from .fragments.types import TargetMassStateSpace
from .graphs.types import Adj, SparseWeightedProductAdj
from .graphs.spectrum_graphs import construct_spectrum_topology
from .graphs.propagate import propagate_cost
from .annotation import AnnotationResult, AnnotationParams
# local

import numpy as np

@dataclasses.dataclass(slots=True)
class AlignmentResult:
    fragment_masses: np.ndarray
    symmetries: list[np.ndarray]
    sparse_prod: list[SparseWeightedProductAdj]
    lo_adj: list[Adj]
    hi_adj: list[Adj]
    cut_adj: list[Adj]

@dataclasses.dataclass(slots=True)
class AlignmentParams:
    cost_threshold: float
    cost_model: tuple[float,float,float,float]
    # suffix_array: SuffixArray

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
    fragment_masses, symmetries, (lo_adj, hi_adj, cut_adj) = construct_spectrum_topology(
        anno.peaks,
        anno.pairs,
        anno.left_boundaries,
        anno.pivots,
        anno.right_boundaries,
    )
    profile["construct"] = time() - t
    if verbose:
        print(lo_adj)
        print(hi_adj)
        print(cut_adj)

    t = time()
    sparse_prod = [propagate_cost(
        left = lo,
        right = hi,
        matched_nodes = anno.pivots.symmetries[i],
        threshold = params.cost_threshold,
        cost_model = params.cost_model,
    ) for (i,(lo,hi)) in enumerate(zip(lo_adj, hi_adj))]
    profile["propagate"] = time() - t
    if verbose:
        print(sparse_prod)

    if verbose:
        print(json.dumps(profile, indent=4))
    return AlignmentResult(
        fragment_masses,
        symmetries,
        sparse_prod,
        lo_adj,
        hi_adj,
        cut_adj,
    )
