from numpy import ndarray
from pyopenms import MSExperiment
from networkx import DiGraph

from collections.abc import Iterator
from typing import Any

#=============================================================================#

TargetGroup = list[float]
Gap = tuple[float, tuple[int,int], int]

Edge = tuple[int,int]
SingularPath = list[int]
DualPath = list[tuple[int,int]]
GraphPair = tuple[DiGraph, DiGraph]