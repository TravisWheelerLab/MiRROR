import dataclasses
from time import time
from typing import Self, Any
# standard

from .fragments.types import TargetMassStateSpace
from .graphs.types import Adj, SparseWeightedProductAdj
from .annotation import AnnotationResult, AnnotationParams
# local

import numpy as np

@dataclasses.dataclass(slots=True)
class AlignmentResult:
    sparse_prod: list[SparseWeightedProductAdj]
    lo_adj: list[Adj]
    hi_adj: list[Adj]

@dataclasses.dataclass(slots=True)
class AlignmentParams:
    cost_model: tuple[float,float,float,float]
    # (substitution, match, vertical gap, horizontal gap)
    cost_threshold: float
    # suffix_array: SuffixArray

    @classmethod
    def from_config(cls, *args, **kwargs):
        pass

def align(
    anno: AnnotationResult,
    targets: TargetMassStateSpace,
    params: AlignmentParams,
    pair_target_space: tuple[np.ndarray,np.ndarray],
    boundary_target_space: tuple[np.ndarray,np.ndarray],
    verbose: bool = False,
) -> AlignmentResult:
    profile = {}
    pair_target_masses, pair_target_indices = pair_target_space
    boundary_target_masses, boundary_target_spaces = boundary_target_space

    lo_adj, hi_adj = construct_spectrum_graphs(
        anno.decharged_peaks,
        anno.pairs,
        anno.left_boundaries,
        anno.pivots.cluster_points,
        anno.right_boundaries,
    )

    sparse_prod = [propagate_cost(
        left = lo,
        right = hi,
        paired_nodes = anno.pivots.symmetries[i],
        pair_costs = np.fill(len(anno.pivots.symmetries[i]), params.paired_cost),
        unpaired_cost = params.unpaired_cost,
        threshold = params.cost_threshold,
        cost_model = params.cost_model,
    )]
