from numpy import ndarray
from pyopenms import MSExperiment

TargetGroup = list[float]
Gap = tuple[float, tuple[int,int], int]


Edge = tuple[int,int]
SingularPath = list[int]
DualPath = list[tuple[int,int]]
GraphPair = tuple[nx.DiGraph, nx.DiGraph]