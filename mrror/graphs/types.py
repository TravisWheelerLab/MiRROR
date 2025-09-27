import dataclasses

import numpy as np
import numba

_cost_ty = numba.types.float64 # weights
_index_ty = numba.types.uint32 # internal node indices
_label_ty = numba.types.uint64 # external node labels

@dataclasses.dataclass(slots=True)
class SparseWeightedProductAdj:
    n: int
    # number of nodes in the sparse product graph.
    
    node_index: dict[int,int]
    # node label -> index.
    
    sparse_adj: list[np.ndarray]
    # sparse adj.
    
    sparse_cost: list[float]
    # sparse cost list.
    
    sparse_labels: list[int]
    # sparse index -> label.

    def ravel(self, u: int, w: int) -> int:
        (u * self.n) + w

    def unravel(self, v) -> tuple[int,int]:
        (
            u // self.n,
            u % self.n,
        )
