import dataclasses, abc
from typing import Self, Any, Union

import numpy as np
import networkx as nx

@dataclasses.dataclass(slots=True)
class SpectrumGraph:
    graph: nx.DiGraph
    boundaries: np.ndarray
    boundary_source: int

    def order(self) -> int:
        """Returns the successor of the largest node.
        
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
            anno,
            np.full_like(anno, -1),
        )
    
    @classmethod
    def from_right_gap(
        cls,
        costs: list[np.ndarray],
        anno: list[np.ndarray],
    ) -> Self:
        return cls(
            costs,
            np.full_like(anno, -1),
            anno,
        )
    

@dataclasses.dataclass(slots=True)
class WeightedProductGraph:
    graph: nx.DiGraph
    right_operand_order: int
    node_weights: dict[int,float]
    edge_weights: dict[tuple[int,int],ProductEdgeWeight] # this is an inefficient way to store and retrieve edge weights, but the rest of the graph data are stored in dictionaries anyways, so until we move away from networkx, it's gonna be like this.

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
