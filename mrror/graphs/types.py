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

    @classmethod
    def from_edges_and_boundaries(
        cls,
        edges: np.ndarray,
        boundaries: np.ndarray,
        multiresidue_boundaries: list[np.ndarray] = [],
    ) -> Self:
        g = nx.DiGraph()
        g.add_weighted_edges_from(edges.T)
        # construct digraph with edges
        boundary_source = max(edges.max(),boundaries.max()) + 1
        # create an artificial source to point to boundaries
        if not(boundaries is None):
            boundary_edges = np.vstack([
                np.full(boundaries.shape[1],boundary_source),
                boundaries,
            ])
            g.add_weighted_edges_from((i,j,(x,1)) for (i,j,x) in boundary_edges.T)
            # construct boundary edges weighted with k = 1 to denote single-residue ids
        for (k, multi_boundaries) in enumerate(multiresidue_boundaries):
            boundary_edges = np.vstack([
                np.full(multi_boundaries.shape[1],boundary_source),
                multi_boundaries,
            ])
            g.add_weighted_edges_from((i,j,(x,k + 1)) for (i,j,x) in boundary_edges.T)
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
class WeightedProductGraph:
    graph: nx.DiGraph
    right_operand_order: int
    node_weights: dict[int,float]

    @classmethod
    def from_edges(
        cls,
        edges: np.ndarray,
        right_operand_order: int,
    ) -> Self:
        g = nx.DiGraph()
        g.add_weighted_edges_from(edges.T)
        return cls(
            g,
            right_operand_order,
        )

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
