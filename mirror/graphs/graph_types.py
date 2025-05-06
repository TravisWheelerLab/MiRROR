from dataclasses import dataclass
from typing import Iterator, Any
from abc import ABC, abstractmethod
from itertools import chain, product

from networkx import DiGraph, is_directed_acyclic_graph, is_bipartite

class BipartiteGraph(DiGraph):
    def __init__(self,
        edges = Iterator[tuple[Any,Any]],
    ):
        graph = DiGraph(edges)
        if is_bipartite(graph):
            super(BipartiteGraph, self).__init__(incoming_graph_data = graph)
        else:
            raise ValueError("not a bipartite graph!")

class DAG:
    def __init__(self,
        graph: DiGraph,
        weight_key: str,
    ):
        if is_directed_acyclic_graph(graph):
            self.graph = graph
            self.weight_key = weight_key
            self._sinks = [i for i in range(self.order()) if list(self.adj_out(i)) == []]
            self._sources = [i for i in range(self.order()) if list(self.adj_in(i)) == []]
        else:
            raise ValueError("not a directed acyclic graph!")
            
    def order(self) -> int:
        return self.graph.order()
    
    def size(self) -> int:
        return self.graph.size()
    
    def adj_out(self, i: int) -> Iterator[int]:
        return self.graph.successors(i)

    def weight_out(self, i: int, j: int) -> Any:
        if i == j:
            return None
        else:
            return self.graph[i][j][self.weight_key]
    
    def sinks(self) -> list[int]:
        return self._sinks
    
    def adj_in(self, i: int) -> Iterator[int]:
        return self.graph.predecessors(i)
    
    def weight_in(self, i: int, j: int) -> Any:
        if i == j:
            return None
        else:
            return self.weight_out(j, i)
    
    def sources(self) -> list[int]:
        return self._sources

class MultiDAG(DAG):
    """A directed acyclic graph that allows multiple edges between a pair
    of nodes. A multi-edge comprising `n` edges is represented as a single 
    edge with integer weight `n`. Constructed from an iterable of edges.
    To change the weight key, set the `count_weight_key` kwarg."""
    def __init__(self,
        edges: Iterator[tuple[int, int]],
        count_weight_key = "count"
    ):
        multigraph = DiGraph()
        for (i, j) in edges:
            multigraph.add_edge(i,j)
            if count_weight_key not in multigraph[i][j]:
                multigraph[i][j][count_weight_key] = 0
            multigraph[i][j][count_weight_key] += 1
        super(MultiDAG, self).__init__(
            graph = multigraph,
            weight_key = count_weight_key,
        )

@dataclass
class ProductDAG(ABC, DAG):
    first_graph: DAG
    second_graph: DAG

    def order(self) -> int:
        return self.first_graph.order() * self.second_graph.order()

    def ravel(self, u: int, v: int) -> int:
        """given a node `u in self.first_graph` and `v in self.second_graph`, associate `(u, v)` to a node `i` in the product graph."""
        return (u * self.second_graph.order()) + v

    def unravel_first(self, i: int) -> int:
        """project node `i` of the product graph onto its node in `self.first_graph`"""
        return i // self.second_graph.order()

    def unravel_second(self, i: int) -> int:
        """project node `i` of the product graph onto its node in `self.second_graph`"""
        return i % self.second_graph.order()

    def unravel(self, i: int) -> tuple[int, int]:
        """project node `i` of the product graph onto the node pair `(u, v)` such that `u in self.first_graph` and `v in self.second_graph`."""
        return (
            self.unravel_first(i), 
            self.unravel_second(i))

    def weight_out(self, i, j):
        u, v = self.unravel(i)
        x, w = self.unravel(j)
        return (
            self.first_graph.weight_out(u, x),
            self.second_graph.weight_out(v, w))

    def weight_in(self, i, j):
        u, v = self.unravel(i)
        x, w = self.unravel(j)
        return (
            self.first_graph.weight_in(u, x),
            self.second_graph.weight_in(v, w))

    @abstractmethod
    def size(self) -> int:
        """count the edges in the product"""
    
    @abstractmethod
    def adj_out(self, i: int) -> Iterator[int]:
        """given `i`, return an iterator of nodes `j` such that `(i, j)` is an edge"""
    
    @abstractmethod
    def adj_in(self, i: int) -> Iterator[int]:
        """given `i`, return an iterator of nodes `j` such that `(j, i)` is an edge"""

@dataclass
class DirectProductDAG(ProductDAG):
    first_graph: DAG
    second_graph: DAG

    def size(self) -> int:
        return self.first_graph.size() * self.second_graph.size()

    def adj_out(self, i: int) -> Iterator[int]:
        return map(
            lambda t: self.ravel(*t), 
            product(
                self.first_graph.adj_out(self.unravel_first(i)), 
                self.second_graph.adj_out(self.unravel_second(i))))
    
    def sinks(self) -> Iterator[int]:
        return map(
            lambda t: self.ravel(*t),
            chain(
                product(
                    range(self.first_graph.order()),
                    self.second_graph.sinks()),
                product(
                    self.first_graph.sinks(),
                    range(self.second_graph.order()))))

    def adj_in(self, i: int) -> Iterator[int]:
        return map(
            lambda t: self.ravel(*t), 
            product(
                self.first_graph.adj_in(self.unravel_first(i)), 
                self.second_graph.adj_in(self.unravel_second(i))))
    
    def sources(self) -> Iterator[int]:
        return map(
            lambda t: self.ravel(*t), 
            chain(
                product(
                    range(self.first_graph.order()),
                    self.second_graph.sources()),
                product(
                    self.first_graph.sources(),
                    range(self.second_graph.order()))))

@dataclass
class BoxProductDAG(ProductDAG):
    first_graph: DAG
    second_graph: DAG

    def size(self) -> int:
        return (self.first_graph.size() * self.second_graph.order()) + (self.second_graph.size() * self.first_graph.order())

    def adj_out(self, i: int) -> Iterator[int]:
        u, v = self.unravel(i)
        return chain(
                map(
                    lambda x: self.ravel(x, v),
                    self.first_graph.adj_out(u)), 
                map(
                    lambda w: self.ravel(u, w),
                    self.second_graph.adj_out(v)))
    
    def sources(self) -> Iterator[int]:
        return map(
            lambda t: self.ravel(*t),
            product(
                self.first_graph.sources(),
                self.second_graph.sources()))

    def adj_in(self, i: int) -> Iterator[int]:
        u, v = self.unravel(i)
        return chain(
                map(
                    lambda x: self.ravel(x, v),
                    self.first_graph.adj_in(u)), 
                map(
                    lambda w: self.ravel(u, w),
                    self.second_graph.adj_in(v)))
    
    def sinks(self) -> Iterator[int]:
        return map(
            lambda t: self.ravel(*t), 
            product(
                self.first_graph.sinks(),
                self.second_graph.sinks()))

class StrongProductDAG(ProductDAG):
    def __init__(self,
        first_graph: DAG,
        second_graph: DAG,
    ):
        self.first_graph = first_graph
        self.second_graph = second_graph
        self.subgraph_box = BoxProductDAG(first_graph, second_graph)
        self.subgraph_direct = DirectProductDAG(first_graph, second_graph)
    
    def size(self) -> int:
        return self.subgraph_box.size() + self.subgraph_direct.size()
    
    def adj_out(self, i: int) -> Iterator[int]:
        return chain(
            self.subgraph_direct.adj_out(i),
            self.subgraph_box.adj_out(i))
    
    def sinks(self) -> Iterator[int]:
        return self.subgraph_box.sinks()

    def adj_in(self, i: int) -> Iterator[int]:
        return chain(
            self.subgraph_direct.adj_in(i),
            self.subgraph_box.adj_in(i))
    
    def sources(self) -> Iterator[int]:
        return self.subgraph_box.sources()