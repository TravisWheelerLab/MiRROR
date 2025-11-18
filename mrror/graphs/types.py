import dataclasses, abc
from typing import Self, Any, Union, Iterator

import numpy as np
import networkx as nx

@dataclasses.dataclass(slots=True)
class SpectrumGraph:
    graph: nx.DiGraph
    boundaries: np.ndarray
    boundary_source: int

    def order(self) -> int:
        """Returns the largest node label plus one.
        
        This graph is sparse, so calling self.graph.order() will return the number of nodes, but that value will not `ravel` or `unravel` correctly, so we need the true order to be one greater than the largest node label."""
        return max(self.graph.nodes) + 1

    def sources(self) -> list[int]:
        return [x for x in self.graph if self.graph.in_degree(x) == 0]

    @classmethod
    def from_edges_and_boundaries(
        cls,
        edges: np.ndarray,
        boundaries: np.ndarray,
        multiresidue_boundaries: list[np.ndarray] = [],
        weight_key: str = 'weight',
    ) -> Self:
        g = nx.DiGraph()
        g.add_weighted_edges_from(edges.T, weight=weight_key)
        # construct digraph with edges
        boundary_source = max(edges.max(),boundaries.max()) + 1
        # create an artificial source to point to boundaries
        if not(boundaries is None):
            boundary_edges = np.vstack([
                np.full(boundaries.shape[1],boundary_source),
                boundaries,
            ])
            g.add_weighted_edges_from(((i,j,(x,1)) for (i,j,x) in boundary_edges.T), weight=weight_key)
            # construct boundary edges weighted with k = 1 to denote single-residue ids
        for (k, multi_boundaries) in enumerate(multiresidue_boundaries):
            boundary_edges = np.vstack([
                np.full(multi_boundaries.shape[1],boundary_source),
                multi_boundaries,
            ])
            g.add_weighted_edges_from(((i,j,(x,k + 1)) for (i,j,x) in boundary_edges.T), weight=weight_key)
            # construct multi-residue boundary edges weighted with k > 1 to denote multi-residue ids.

        return cls(
            g,
            boundaries,
            boundary_source,
        )

@dataclasses.dataclass(slots=True)
class PivotGraph:
    graph: nx.DiGraph

    @classmethod
    def from_edges(
        cls,
        edges: np.ndarray,
    ) -> Self:
        g = nx.DiGraph()
        g.add_weighted_edges_from(edges.T)
        return cls(g)

@dataclasses.dataclass(slots=True)
class ProductEdgeWeight:
    """Annotation data associated to an edge in a WeightedProductGraph. Note that comparisons between ProductEdgeWeight are computed from the order of their minimum cost. Two completely different annotations may be equal if they have the same minimum cost."""
    costs: np.ndarray
    left_annotation: np.ndarray
    right_annotation: np.ndarray

    def __eq__(self, other):
        return self.costs.min() == other.costs.min()

    def __lt__(self, other):
        return self.costs.min() < other.costs.min()

    @classmethod
    def from_comparison(
        cls,
        costs: np.ndarray,
        left_anno: np.ndarray,
        right_anno: np.ndarray,
    ) -> Self:
        return cls(
            costs,
            left_anno,
            right_anno,
        )

    @classmethod
    def from_left_gap(
        cls,
        costs: list[np.ndarray],
        anno: list[np.ndarray],
    ) -> Self:
        return cls(
            costs,
            np.full_like(anno, -1),
            anno,
        )
    
    @classmethod
    def from_right_gap(
        cls,
        costs: list[np.ndarray],
        anno: list[np.ndarray],
    ) -> Self:
        return cls(
            costs,
            anno,
            np.full_like(anno, -1),
        )
    

@dataclasses.dataclass(slots=True)
class WeightedProductGraph:
    """A sparse product of two graphs as returned by propagate_cost. Stores node and edge weights in dictionaries. Implements ravel and unravel, which map between nodes in the product and pairs of nodes in the operand graphs."""
    graph: nx.DiGraph
    right_operand_order: int
    node_weights: dict[int,float]
    edge_weights: dict[int,dict[int,tuple[float,ProductEdgeWeight]]] # this is an inefficient way to store and retrieve edge weights, but the rest of the graph data are stored in dictionaries anyways, so until we move away from networkx, it's gonna be like this.

    def ravel(
        self,
        left_node: int,
        right_node: int,
    ) -> int:
        return (left_node * self.right_operand_order) + right_node

    def unravel(
        self,
        product_node: int,
    ) -> tuple[int,int]:
        return (
            product_node // self.right_operand_order,
            product_node % self.right_operand_order,
        )

@dataclasses.dataclass(slots=True)
class PathSpace:
    """A collection of paths through a WeightedProductGraph, associated to cost and state. Implements __len__, __getitem__, __iter__. Paths are stored in a single array of integers, segmented by offsets. The i^th path is given by path[offset[i]:offset[i+1]]."""
    path: np.ndarray
    offset: np.ndarray
    cost: np.ndarray
    state: np.ndarray

    def __len__(self) -> int:
        return len(self.offset) - 1

    def __getitem__(self, i: int) -> tuple[float,Any,list]:
        l = self.offset[i]
        r = self.offset[i + 1]
        return (
            self.cost[i],
            self.state[i],
            self.path[l:r],
        )

    def __iter__(self) -> Iterator:
        return (self.__getitem__(i) for i in range(len(self)))

    @classmethod
    def empty(cls):
        return cls(
            path = np.array([]),
            offset = np.array([0,]),
            cost = np.array([]),
            state = np.array([]),
        )
