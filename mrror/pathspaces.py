import dataclasses
from typing import Self, Iterator

from .graphs.trace import AbstractPathSpace
from .sequences.suffix_array import BisectResult

import numpy as np

AnnotatedResiduePathState = list[tuple[np.ndarray,np.ndarray]] # each tuple weights an edge in the path with cost and residue annotation data.

@dataclasses.dataclass(slots=True)
class AnnotatedResiduePathSpace(AbstractPathSpace):
    """A collection of paths traced through a WeightedProductGraph using the AnnotatedResiduePathCostModel. Each path is associated to a cost and an AnnotatedResiduePathState in the form of an annotation matrix and a cost vector."""
    path: np.ndarray
    offset: np.ndarray
    cost: np.ndarray
    annotation: np.ndarray
    annotation_cost: np.ndarray

    def __len__(self) -> int:
        return len(self.offset) - 1

    def __getitem__(self, i: int) -> tuple:
        l = self.offset[i]
        r = self.offset[i + 1]
        return (
            self.cost[i],
            self.path[l:r],
            self.annotation[i],
            self.annotation_cost[i],
        )

    def __iter__(self) -> Iterator:
        return (self.__getitem__(i) for i in range(len(self)))

    def __add__(self, other: Self) -> Self:
        concat_offset = np.concat([self.offset[:-1], other.offset + self.offset[-1]])
        # [0, 3, 5, 10] + [0, 2, 4, 7] = [0, 3, 5, 10, 12, 14, 17]
        return type(self)(
            path = np.concat([self.path, other.path]),
            offset = concat_offset,
            cost = np.concat([self.cost, other.cost]),
            annotation = np.concat([self.annotation, other.annotation]),
            annotation_cost = np.concat([self.annotation_cost, other.annotation_cost]),
        )

    def get_path(self, i: int) -> float:
        l = self.offset[i]
        r = self.offset[i + 1]
        print(i,l,r)
        return self.path[l:r]

    def get_cost(self, i: int) -> float:
        return self.cost[i]
    
    @classmethod
    def empty(cls):
        return cls(
            path = np.array([]),
            offset = np.array([0,]),
            cost = np.array([]),
            state = np.array([]),
        )

    @classmethod
    def from_traced_paths(
        cls,
        trace_results: Iterator[tuple[
            float,
            AnnotatedResiduePathState,
            list[int],
        ]],
    ) -> Self:
        trace_results = list(trace_results)
        n_paths = len(trace_results)
        costs = np.empty((n_paths,), dtype=float)
        paths = np.empty((n_paths,), dtype=list)
        path_annotations = np.empty((n_paths,), dtype=list)
        path_anno_costs = np.empty((n_paths,), dtype=list)
        for (i,res) in enumerate(trace_results):
            states = res[1]
            anno_costs, annotations = zip(*states)
            costs[i] = res[0]
            paths[i] = res[2]
            path_annotations[i] = list(annotations)
            path_anno_costs[i] = list(anno_costs)
        order = np.argsort(costs)
        costs = costs[order]
        paths = paths[order]
        path_annotations = path_annotations[order]
        path_anno_costs = path_anno_costs[order]
        return cls(
            path = np.concat(paths),
            offset = np.cumsum([0,] + [len(x) for x in paths]),
            cost = costs,
            annotation = path_annotations,
            annotation_cost = path_anno_costs,
        )

SuffixArrayPathState = tuple[
    list[BisectResult],         # one bisect result: one sequence.
    AnnotatedResiduePathState,    # per-position annotations. see above.
]

@dataclasses.dataclass(slots=True)
class SuffixArrayPathSpace(AnnotatedResiduePathSpace):
    """A collection of paths traced through a WeightedProductGraph using the SuffixArrayPathCostModel. Each path is associated to a cost and an SuffixArrayPathState in the form of a list of bisect results, an annotation matrix, and an annotation cost vector. Annotation data is identical to that of the parent class AnnotatedResiduePathSpace."""
    path: np.ndarray
    offset: np.ndarray
    cost: np.ndarray
    annotation: np.ndarray
    annotation_cost: np.ndarray
    path_bisect_result: np.ndarray
    bisect_offset: np.ndarray

    def __getitem__(self, i: int) -> tuple:
        l = self.offset[i]
        r = self.offset[i + 1]
        l2 = self.bisect_offset[i]
        r2 = self.bisect_offset[i + 1]
        return (
            self.cost[i],
            self.path[l:r],
            self.annotation[i],
            self.annotation_cost[i],
            self.path_bisect_result[l2:r2],
        )

    def __add__(self, other: Self) -> Self:
        concat_offset = np.concat([self.offset[:-1], other.offset + self.offset[-1]])
        concat_bisect_offset = np.concat([self.bisect_offset[:-1], other.bisect_offset + self.bisect_offset[-1]])
        # [0, 3, 5, 10] + [0, 2, 4, 7] = [0, 3, 5, 10, 12, 14, 17]
        return type(self)(
            path = np.concat([self.path, other.path]),
            offset = concat_offset,
            cost = np.concat([self.cost, other.cost]),
            annotation = np.concat([self.annotation, other.annotation]),
            annotation_cost = np.concat([self.annotation_cost, other.annotation_cost]),
            path_bisect_result = np.concat([self.path_bisect_result, other.path_bisect_result]),
            bisect_offset = concat_bisect_offset,
        )

    def get_bisect_result(self, i: int) -> np.ndarray:
        l = self.bisect_offset[i]
        r = self.bisect_offset[i + 1]
        return self.path_bisect_result[l:r]

    @classmethod
    def empty(cls):
        return cls(
            path = np.array([]),
            offset = np.array([0,]),
            cost = np.array([]),
            state = np.array([]),
        )

    @classmethod
    def from_traced_paths(
        cls,
        trace_results: Iterator[tuple[
            float,
            AnnotatedResiduePathState,
            list[int],
        ]],
    ) -> Self:
        trace_results = list(trace_results)
        n_paths = len(trace_results)
        costs = np.empty((n_paths,), dtype=float)
        paths = np.empty((n_paths,), dtype=list)
        path_bisect_results = np.empty((n_paths,), dtype=list)
        path_annotations = np.empty((n_paths,), dtype=list)
        path_anno_costs = np.empty((n_paths,), dtype=list)
        for (i,res) in enumerate(trace_results):
            bisect_results, states = res[1]
            anno_costs, annotations = zip(*states)
            costs[i] = res[0]
            paths[i] = res[2]
            path_bisect_results[i] = bisect_results
            path_annotations[i] = annotations
            path_anno_costs[i] = anno_costs
        order = np.argsort(costs)
        costs = costs[order]
        paths = paths[order]
        path_bisect_results = path_bisect_results[order]
        path_annotations = path_annotations[order]
        path_anno_costs = path_anno_costs[order]
        return cls(
            path = np.concat(paths),
            offset = np.cumsum([0,] + [len(x) for x in paths]),
            cost = costs,
            annotation = path_annotations,
            annotation_cost = path_anno_costs,
            path_bisect_result = np.concat(path_bisect_results),
            bisect_offset = np.cumsum([0,] + [len(x) for x in path_bisect_results]),
        )
