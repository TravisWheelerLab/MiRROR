import dataclasses

import numpy as np
import numba

_cost_ty = numba.types.float32 # weights
_index_ty = numba.types.uint32 # internal node indices
_label_ty = numba.types.uint32 # external node labels

@dataclasses.dataclass(slots=True)
class Adj:
    adj: list[np.ndarray]
    order: int
    sources: list[int]

    @classmethod
    def from_adj(cls,
        adj: list[list[int]],
    ):
        order = len(adj)
        in_degree = np.zeros(order)
        for v in range(order):
            for w in adj[v]:
                in_degree[w] += 1
        return cls(
            adj = [np.array(x) for x in adj],
            order = order,
            sources = [x for x in range(order) if in_degree[x] == 0],
        )

    @classmethod
    def from_edges(cls,
        edges: np.ndarray,
        reverse: bool = False,
        directed: bool = True,
    ):
        if reverse:
            edges = edges[::-1,:]
        elif not(directed):
            edges = np.concat([edges,edges[::-1,:]],axis=1)
        lo = edges.min()
        hi = edges.max()
        edges = edges - lo
        order = hi - lo + 1
        adj = [[] for _ in range(order)]
        edge_src, edge_tgt = edges
        for (i, j) in zip(edge_src, edge_tgt):
            adj[i].append(j)
        return cls.from_adj(adj)

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

    @classmethod
    def from_edges(cls,
        n,
        node_idx,
        edge_src,
        edge_tgt,
        cost,
        labels
    ):
        adj = [[] for _ in range(n)]
        for (i, j) in zip(edge_src, edge_tgt):
            adj[i].append(j)
        return cls(
            n,
            node_idx,
            [np.array(x) for x in adj],
            cost,
            labels,
        )
