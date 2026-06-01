import abc
import dataclasses
from typing import Union, Any, Iterator, Self

import numpy as np

from ..util import merge_compare_exact_unique, ravel, unravel, combine_symbols, combine_masses
from ..fragments.types import ResidueStateSpace
from ..sequences.suffix_array import SuffixArray, BisectResult
from ..fragments.types import TargetMasses, BoundaryResult, PairResult
from ..graphs.types import SpectrumGraph, PivotGraph, SymmetricGraph
from ..graphs.align import AbstractNodeCostModel, AbstractEdgeCostModel, AbstractPathCostModel, AugmentedLetter

@dataclasses.dataclass(slots=True)
class SymmetricNodeCostModel:
    mass: np.ndarray
    reflector: float
    tolerance: float

    def __call__(self, i: int, j: int) -> float:
        return 0. if abs(self.mass[i] + self.mass[j] - self.reflector) < self.tolerance else 1.

    @classmethod
    def from_axis(
        cls,
        fragment_mass: np.ndarray,
        axis: float,
        tolerance: float,
    ) -> Self:
        return cls(
            mass = fragment_mass,
            reflector = 2 * axis,
            tolerance = tolerance,
        )

@dataclasses.dataclass(slots=True)
class AnnotatedEdgeCostModel:
    cost: np.ndarray
    mass: np.ndarray
    residue_id: np.ndarray
    res_id_dims: tuple[int,int]
    segment: np.ndarray

    def __call__(self, i: int, j: int) -> tuple[int,AugmentedLetter]:
        i_lo, i_hi = self.segment[i:i+2]
        j_lo, j_hi = self.segment[j:j+2]
        edge_costs = self.cost[i_lo:i_hi] + self.cost[j_lo:j_hi].reshape((-1,1))
        comp_costs = 1 - (self.residue_id[i_lo:i_hi] == self.residue_id[j_lo:j_hi].reshape((-1,1)))
        costs = edge_costs + comp_costs
        min_i, min_j = np.unravel_index(costs.argmin,costs.shape)
        min_cost = costs[min_i,min_j]
        min_anno_amino, min_anno_mod = np.unravel_index(self.residue_id[i_lo + i_min],self.res_id_dims)
        # OK to use just i_min because edge_costs are an epsilon relative to comp_costs; edge costs are used to determine which pair of matching residue ids are used for the final match, but will never dominate the mismatch cost and therefore will never lead to min_i encoding a different id than min_j.
        
        min_anno_mass = self.mass[i_lo + i_min]
        min_anno = (min_anno_amino, min_anno_mod, min_anno_mass) 
        # (int,int,float) aliased as AugmentedLetter

        return (min_cost, min_anno)


@dataclasses.dataclass(slots=True)
class MassConstrainedPathCostModel:
    target_mass: float

    @classmethod
    def initial_state(cls) -> None:
        """The mass constrained path cost model is stateless. Its only function is to check whether the peptide mass of a given AugmentedLetter exceeds the target mass."""
        return None
    
    def __call__(self, path_state: None, edge_anno: AugmentedLetter):
        new_mass = edge_anno
        amino = self.residue_space.amino_symbols[amino_idx]
        new_pfx = self.suffix_array.bisect([amino,],pfx)[0],
        return (
            np.inf if (new_pfx.count == 0 or new_mass > self.target) else 0.,
            new_pfx,
        )

    @classmethod
    def from_axis(
        cls,
        axis: float,
        tolerance: float,
    ) -> Self:
        return cls(
            target_mass = 2 * (axis + tolerance),
        )

@dataclasses.dataclass(slots=True)
class SuffixArrayPathCostModel(MassConstrainedPathCostModel):
    residue_space: ResidueStateSpace
    suffix_array: SuffixArray

    def __call__(self, path_state: tuple[float,BisectResult], edge_anno: AugmentedLetter):
        mass, pfx = path_state
        amino_idx, _, = edge_anno
        amino = self.residue_space.amino_symbols[amino_idx]
        new_pfx = self.suffix_array.bisect([amino,],pfx)[0],
        return (
            np.inf if (new_pfx.count == 0 or new_mass > self.target) else 0.,
            new_pfx,
        )

    @classmethod
    def from_mass_constraint(
        cls,
        constraint: MassConstrainedPathCostModel,
        residue_space: ResidueStateSpace,
        suffix_array: SuffixArray,
    ) -> Self:
        return cls(
            constraint.target_mass,
            constraint.tolerance,
            residue_space,
            suffix_array,
        )
