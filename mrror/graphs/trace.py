import abc
from typing import Iterator, Any, Self
from collections import deque

from .types import WeightedProductGraph

import numpy as np

class AbstractPathCostModel(abc.ABC):

    @abc.abstractmethod
    def initial_state(self, node: int) -> tuple[float,Any]:
        """Constructs the initial cost and cost state for a given node.
        
        arguments:
        - node: int, the index of a node in the graph.
        
        return:
        - initial_cost: float, typically zero or the cost of the given node.
        - initial_state: T, representing the path with one node."""

    @abc.abstractmethod
    def next_state(self, cost_state, curr_node: int, next_node: int) -> tuple[float,Any]:
        """Constructs the successor cost and cost state for a given cost state and edge.
        
        arguments:
        - cost_state: T, the state of the path.
        - curr_node: int, the last node in the path.
        - next_node: int, a node adjacent to curr_node.
        
        return:
        - delta_cost: float, the cost incurred by traversing the edge (curr_node, next_node) with cost_state.
        - next_state: T, the state of the new path including next_node."""

    @classmethod
    @abc.abstractmethod
    def path_space_type(self) -> type:
        """Returns the path space class which implements the classmethod constructor from_traced_paths."""

class AbstractPathSpace(abc.ABC):

    @abc.abstractmethod
    def __len__(self) -> int:
        """The number of paths in the path space."""

    @abc.abstractmethod
    def __getitem__(self, i: int) -> tuple[float,Any,list]:
        """Get the i^th path."""

    @abc.abstractmethod
    def __iter__(self) -> Iterator:
        """Iterate the path space. Typically equivalent to
        
            (pathspace[i] for i in range(len(pathspace)))."""

    @abc.abstractmethod
    def __add__(self, other: Self) -> Self:
        """Concatenate one path space onto another in order of the sum, i.e., if `a` and `b` are path spaces,
        
            list(a + b) == list(a) + list(b)."""

    @classmethod
    @abc.abstractmethod
    def empty(cls) -> Self:
        """Return an empty path space."""

    @classmethod
    @abc.abstractmethod
    def from_traced_paths(cls, trace_result: Iterator[tuple[float,Any,list[int]]]) -> Self:
        """Constructs a path space object from the output of _trace."""

def _trace(
    adj,
    initial_states: list[tuple[float,list[int],Any,int]],
    threshold: float,
    cost_model: AbstractPathCostModel,
) -> Iterator[tuple[float,Any,list[int]]]:
    q = deque(initial_states)
    while len(q) > 0:
        curr_cost, curr_cost_state, curr_path, curr_node = q.pop()
        if curr_cost < threshold:
            next_path = curr_path + [curr_node,]
            neighbors = list(adj[curr_node])
            if len(neighbors) == 0:
                yield (curr_cost, curr_cost_state, next_path)
            else:
                for next_node in neighbors:
                    delta_cost, next_cost_state = cost_model.next_state(curr_cost_state, curr_node, next_node)
                    next_cost = curr_cost + delta_cost
                    q.append((next_cost, next_cost_state, next_path, next_node))

def trace(
    prop_graph: WeightedProductGraph,
    sources: Iterator[int],
    threshold: float,
    cost_model: AbstractPathCostModel,
) -> AbstractPathSpace:
    path_space_cls = cost_model.path_space_type()
    return path_space_cls.from_traced_paths(_trace(
            prop_graph.graph.adj,
            [(*cost_model.initial_state(x), [], x) for x in sources],
            threshold,
            cost_model,
    ))
