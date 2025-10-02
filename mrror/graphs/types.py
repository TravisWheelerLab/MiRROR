import dataclasses

import numpy as np
import numba

_cost_ty = numba.types.float64 # weights
_index_ty = numba.types.uint32 # internal node indices
_label_ty = numba.types.uint64 # external node labels

@dataclasses.dataclass(slots=True)
class Adj:
    adj: list[np.ndarray]
    order: int
    sources: list[int]

    @classmethod
    def from_edges(cls, edges: list[tuple[int,int]]):
        order = len(adj)
        in_degrees = np.zeros(order)
        for v in range(order):
            for w in adj[v]:
                in_degrees[w] += 1
        cls(
            adj,
            order,
            [x for x in range(order) if in_degrees[x] == 0],
        )

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
