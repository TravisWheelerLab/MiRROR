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
class SparseAdj:
    order: int # number of nodes = number of labels. indices run [0 .. n-1]
    node_index: dict[int,int] # label -> index.
    node_labels: list[int] # index -> label.
    adj: list[np.ndarray] # index -> neighbor indices
    sources: list[int]

    @classmethod
    def from_edges(cls,
        edges: np.ndarray,
        reverse: bool = False,
        directed: bool = True,
    ):
        node_labels, edge_reindexer = np.unique_inverse(edges)
        order = node_labels.size
        node_indices = np.arange(order)
        edges = node_indices[edge_reindexer]
        if reverse:
            edges = edges[::-1,:]
        elif not(directed):
            edges = np.concat([edges,edges[::-1,:]],axis=1)
        adj = [[] for _ in range(order)]
        in_degree = np.zeros(order)
        for (i, j) in zip(edges[0,:], edges[1,:]):
            adj[i].append(j)
            in_degree[j] += 1
        return cls(
            order = order,
            node_index = { label: index for (label, index) in zip(node_labels, node_indices)},
            node_labels = node_labels,
            adj = [np.array(x) for x in adj],
            sources = [i for i in range(order) if in_degree[i] == 0],
        )

@dataclasses.dataclass(slots=True)
class SparseWeightedProductAdj:
    order: int
    node_index: dict[int,int]    
    node_labels: list[int]
    adj: list[np.ndarray]
    node_cost: list[float] # index -> cost
    right_order: int

    def ravel(self, u: int, w: int) -> int:
        return (u * self.right_order) + w

    def unravel(self, v) -> tuple[int,int]:
        return (
            u // self.right_order,
            u % self.right_order,
        )

    @classmethod
    def from_edges(cls,
        order,
        node_idx,
        labels,
        edge_src,
        edge_tgt,
        cost,
        right_order,
    ):
        adj = [[] for _ in range(order)]
        for (i, j) in zip(edge_src, edge_tgt):
            adj[i].append(j)
        return cls(
            order,
            node_idx,
            labels,
            [np.array(x) for x in adj],
            cost,
            right_order,
        )
