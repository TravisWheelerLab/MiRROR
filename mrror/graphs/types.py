import dataclasses, abc
from typing import Self, Any, Union, Iterator

import numpy as np
import networkx as nx

@dataclasses.dataclass(slots=True)
class SpectrumGraph:
    graph: nx.DiGraph
    boundaries: np.ndarray
    boundary_source: int
    pivots: np.ndarray
    pivot_sink: int
    weight_key: str

    def display(self) -> str:
        return f"SpectrumGraph\nboundaries {self.boundaries}\nboundary_source {self.boundary_source}\npivots {self.pivots}\npivot_sink {self.pivot_sink}\nedges " + ' '.join([f"({str(i)} -> {str(j)})" for (i,j) in self.graph.edges()])

    def order(self) -> int:
        """Returns the largest node label plus one.
        
        This graph is sparse, so calling self.graph.order() will return the number of nodes, but that value will not `ravel` or `unravel` correctly, so we need the true order to be one greater than the largest node label."""
        return max(self.graph.nodes) + 1

    def sources(self) -> list[int]:
        return [x for x in self.graph if self.graph.in_degree(x) == 0]

    def sinks(self) -> list[int]:
        return [x for x in self.graph if self.graph.out_degree(x) == 0]

    def get_weight(self, i: int, j: int) -> int:
        return self.graph[i][j][self.weight_key]

    @classmethod
    def empty(cls, weight_key: str = "weight") -> Self:
        graph = nx.DiGraph()
        graph.add_nodes_from([0,])
        return cls(
            graph = graph,
            boundaries = np.empty((0,),dtype=int),
            boundary_source = 1,
            pivots = np.empty((0,),dtype=int),
            pivot_sink = 2,
            weight_key = weight_key,
        )

    @classmethod
    def from_edges_and_boundaries(
        cls,
        edges: np.ndarray,
        boundaries: np.ndarray,
        pivots: np.ndarray,
        multiresidue_boundaries: list[np.ndarray] = [],
        weight_key: str = "weight",
    ) -> Self:
        g = nx.DiGraph()
        g.add_weighted_edges_from(edges, weight=weight_key)
        # construct digraph with edges
        if edges.size > 0:
            if boundaries.size > 0:
                boundary_source = max(edges.max(),boundaries.max()) + 1
            else:
                boundary_source = edges.max() + 1
                # handle empty boundaries.
        else:
            return cls.empty(weight_key = weight_key)
            # handle empty graphs
        pivot_sink = boundary_source + 1
        # create an artificial source to point to boundaries and likewise a sink to pivots.
        g.add_weighted_edges_from(((boundary_source,x,(w,1)) for (x,w) in boundaries), weight=weight_key)
        # construct boundary edges weighted with k = 1 to denote single-residue ids
        g.add_weighted_edges_from(((x,pivot_sink,-1) for x in pivots), weight=weight_key)
        # construct pivot edges weighted with -1 to denote null annotation.
        for (k, multi_boundaries) in enumerate(multiresidue_boundaries):
            boundary_edges = np.vstack([
                np.full(multi_boundaries.shape[1],boundary_source),
                multi_boundaries,
            ])
            g.add_weighted_edges_from(((i,j,(x,k + 1)) for (i,j,x) in boundary_edges), weight=weight_key)
            # construct multi-residue boundary edges weighted with k > 1 to denote multi-residue ids.

        return cls(
            g,
            boundaries,
            boundary_source,
            pivots,
            pivot_sink,
            weight_key,
        )

@dataclasses.dataclass(slots=True)
class PivotGraph:
    graph: nx.Graph
    weight_key: str

    # def __post_init__(self):
    #     for (k,v) in sorted(self.graph.adj.items()):
    #         print(k,'\t',[int(x) for (x,_) in v.items()])

    def get_weight(self, i: int, j: int) -> int:
        return self.graph[i][j][self.weight_key]

    @classmethod
    def from_edges(
        cls,
        edges: np.ndarray,
        weight_key: str = 'weight',
    ) -> Self:
        g = nx.Graph()
        g.add_weighted_edges_from(edges, weight=weight_key)
        return cls(g, weight_key)

@dataclasses.dataclass(slots=True)
class ProductEdgeWeight:
    """Annotation data associated to an edge in a WeightedProductGraph.
    
    Note that comparisons between ProductEdgeWeight are computed from the order of their minimum cost. Two completely different annotations may be equal if they have the same minimum cost."""
    costs: np.ndarray
    lower_annotation: np.ndarray
    upper_annotation: np.ndarray

    def __eq__(self, other):
        return self.costs.min() == other.costs.min()

    def __lt__(self, other):
        return self.costs.min() < other.costs.min()

    @classmethod
    def from_comparison(
        cls,
        costs: np.ndarray,
        lower_anno: np.ndarray,
        upper_anno: np.ndarray,
    ) -> Self:
        return cls(
            costs,
            lower_anno,
            upper_anno,
        )

    @classmethod
    def from_lower_gap(
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
    def from_upper_gap(
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
    upper_operand_order: int
    node_weights: dict[int,float]
    edge_weights: dict[int,dict[int,tuple[float,ProductEdgeWeight]]] 

    def sources(self) -> list[int]:
        return [x for x in self.graph if self.graph.in_degree(x) == 0]

    def ravel(
        self,
        lower_node: int,
        upper_node: int,
    ) -> int:
        return (lower_node * self.upper_operand_order) + upper_node

    def unravel(
        self,
        product_node: int,
    ) -> tuple[int,int]:
        return (
            product_node // self.upper_operand_order,
            product_node % self.upper_operand_order,
        )
