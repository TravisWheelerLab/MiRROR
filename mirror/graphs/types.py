import dataclasses, abc
from typing import Self, Any, Union, Iterator

import numpy as np

from ..util import ravel, unravel

@dataclasses.dataclass(slots=True)
class Graph:
    adj: np.ndarray
    idx: np.ndarray
    off: np.ndarray

    def size(self):
        return len(self.adj)

    def order(self):
        return len(self.off) - 1

    def adjacent(self, i: int) -> np.ndarray:
        return self.adj[self.off[i]:self.off[i+1]]

    def edge_index(self, i: int) -> np.ndarray:
        return self.idx[self.off[i]:self.off[i+1]]

def _construct_graph(
    order: int,
    edges: np.ndarray,
    index: np.ndarray,
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    source_order = edges[:,0].argsort()
    edges = edges[source_order]
    src = edges[:,0]
    lpad_deg = np.zeros(order + 1, dtype=int)
    for i in src:
        lpad_deg[i + 1] += 1
    return (
        edges[:,1], # adj
        index[source_order], # idx
        np.cumsum(lpad_deg), # off
    )

@dataclasses.dataclass(slots=True)
class SpectrumGraph:
    graph: Graph
    boundary_node: int
    axial_node: int
    gap_state_node: int

    @classmethod
    def from_index(
        cls,
        pairs: np.ndarray,
        pairs_idx: np.ndarray,
        boundary_sources: np.ndarray,
        boundary_idx: np.ndarray,
        axial_sinks: np.ndarray,
        axial_idx: np.ndarray,
        num_peaks: int = None,
    ) -> Graph:
        """Builds a SpectrumGraph from the edges described by `pairs`.
        the graph will contain three special nodes:
        1) the boundary node := pairs.max() + 1 is a source whose outgoing edges point to each node in `boundary_sources`.
        2) the axial node := pairs.max() + 2 is a sink node whose incoming edges point from each node in `axial_sinks`.
        3) the gap state node := pairs.max() + 3 is a singleton node, with no outgoing or incoming edges, which is used to represent the branch-and-bound state of the propagate algorithm."""
        max_node = num_peaks - 1 if num_peaks else max(max(pairs.flatten(),default=-1),max(boundary_sources,default=-1),max(axial_sinks,default=-1))
        boundary_node = max_node + 1
        axial_node = max_node + 2
        gap_state_node = max_node + 3
        boundary_edges = np.array([[boundary_node, source] for source in boundary_sources]) if len(boundary_sources) > 0 else np.empty(shape=(0,2),dtype=int)
        axial_edges = np.array([[sink, axial_node] for sink in axial_sinks]) if len(axial_sinks) else np.empty(shape=(0,2),dtype=int)
        edges = np.vstack([pairs,boundary_edges,axial_edges])
        adj, idx, off = _construct_graph(
            order = gap_state_node + 1,
            edges = edges,
            index = np.concat([pairs_idx, boundary_idx, axial_idx]),
        )
        return cls(
            Graph(adj, idx, off),
            boundary_node,
            axial_node,
            gap_state_node,
        )

@dataclasses.dataclass(slots=True)
class SparseGraph:
    adj: np.ndarray
    idx: np.ndarray
    off: np.ndarray
    vtx: np.ndarray

    def size(self):
        return len(self.adj)

    def order(self):
        return len(self.vtx) - 1

    def occupancy(self):
        return len(self.off) - 1

    def adjacent(self, i: int) -> np.ndarray:
        i = self.vtx[i + 1]
        return self.adj[self.off[i]:self.off[i+1]]

    def edge_index(self, i: int) -> np.ndarray:
        i = self.vtx[i + 1]
        return self.idx[self.off[i]:self.off[i+1]]

    @classmethod
    def from_edges(
        cls,
        capacity: int,
        edges: list[tuple[int,int]],
        edge_indices: np.ndarray,
        reverse: bool = False,
    ):
        edges = np.array(edges)
        if reverse:
            edges = edges[:,::-1]
        shifted_sources = edges[:,0] + 1
        # why shift by 1? in case there is a node at 0. the true node at 0 is a singleton used to represent all other nodes in capacity that aren't included in edges.
        occupants = np.unique(shifted_sources)
        occupancy = len(occupants)
        vtx = np.zeros(capacity + 1, dtype=int)
        vtx[occupants] = np.arange(occupancy) + 1
        edges = np.stack([
            vtx[shifted_sources],
            edges[:,1],
        ], axis=1)
        # why only make the edge sources dense?
        # the adjacent function is passed a sparse node index and returns a list of sparse node indices. the denseness of edge sources determines the efficiency of the underlying graph structure, so sources need to be dense. target values don't determine anything* about the structure, so by storing them as the original sparse values we avoid a spurious conversion from dense to sparse, which would require another array.
        # * except its order.
        adj, idx, off = _construct_graph(
            order = occupancy + 1,
            edges = edges,
            index = edge_indices,
        )
        return cls(
            adj,
            idx,
            off,
            vtx,
        )

@dataclasses.dataclass(slots=True)
class PivotGraph:
    graph: SparseGraph

    @classmethod
    def from_index(
        cls,
        pivot_pairs: np.ndarray,
        pivot_pairs_idx: np.ndarray,
    ) -> Self:
        return cls(SparseGraph.from_edges(
            capacity = pivot_pairs.max() + 1,
            edges = np.concat([pivot_pairs,pivot_pairs[:,::-1]]),
            edge_indices = np.concat([pivot_pairs_idx,pivot_pairs_idx]),
        ))

@dataclasses.dataclass(slots=True)
class SymmetricGraph:
    graph: SparseGraph

    @classmethod
    def from_index(
        cls,
        symmetric_pairs: np.ndarray,
    ) -> Self:
        return cls(SparseGraph.from_edges(
            capacity = symmetric_pairs.max() + 1,
            edges = np.concat([symmetric_pairs,symmetric_pairs[:,::-1]]),
            edge_indices = np.full(len(symmetric_pairs) * 2, 0),
        ))

@dataclasses.dataclass(slots=True)
class ProductGraph:
    topology: SparseGraph
    first_order: int
    second_order: int

    def size(self):
        return self.topology.size()

    def order(self):
        return self.topology.order()

    def occupancy(self):
        return self.topology.occupancy()

    def adjacent(self, i: int) -> np.ndarray:
        return self.topology.adjacent(i)

    def edge_index(self, i: int) -> np.ndarray:
        return self.topology.edge_index(i)

    def ravel(self, first_position: int, second_position: int) -> int:
        return ravel(first_position, second_position, self.second_order)

    def unravel(self, raveled_position: int) -> tuple[int,int]:
        return unravel(raveled_position, self.second_order)

@dataclasses.dataclass(slots=True)
class Pathspace:
    paths: np.ndarray
    off: np.ndarray

    def __len__(self) -> int:
        return len(self.off) - 1

    def __getitem__(self, i: int) -> np.ndarray:
        return self.paths[self.off[i]:self.off[i+1]]

    @classmethod
    def from_paths(
        cls,
        paths: list[list[int]],
    ) -> Self:
        return cls(
            paths = np.concat(paths),
            off = np.cumsum([0,] + [len(x) for x in paths])
        )
