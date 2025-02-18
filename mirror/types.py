from numpy import ndarray
from pyopenms import MSExperiment
from networkx import DiGraph

from collections.abc import Iterator
from typing import Any

# the types module is an initial object; it cannot have any local dependencies.

#=============================================================================#

TargetGroup = list[float]

Edge = tuple[int,int]
SingularPath = list[int]
DualPath = list[tuple[int,int]]
GraphPair = tuple[DiGraph, DiGraph]