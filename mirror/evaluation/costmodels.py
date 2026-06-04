import abc
import dataclasses
from typing import Union, Any, Iterator, Self

import numpy as np

from ..util import merge_compare_exact_unique, ravel, unravel, combine_symbols, combine_masses
from ..fragments.types import ResidueStateSpace
from ..sequences.suffix_array import SuffixArray, BisectResult
from ..fragments.types import TargetMasses, BoundaryResult, PairResult, UniqueFragmentIndex, AnnotationIndex
from ..graphs.types import SpectrumGraph, PivotGraph, SymmetricGraph, AbstractNodeCostModel, AbstractEdgeCostModel, AbstractPathCostModel, AugmentedLetter

@dataclasses.dataclass(slots=True)
class SymmetricNodeCostModel:
    """Assigns pairs of nodes in the product grapha  cost based on their reflection symmetry udner a given axis. There is one node cost model per axis."""
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
            mass = np.concat([fragment_mass,[0.,0.,0.,0.,0.,]]),
            reflector = 2 * axis,
            tolerance = tolerance,
        )

@dataclasses.dataclass(slots=True)
class AnnotatedEdgeCostModel:
    """Assigns pairs of spectrum graph edges a cost in the product graph based on the similarity of their annotation and the cost of those annotations. There is one edge cost model per spectrum."""
    cost: np.ndarray
    mass: np.ndarray
    residue_id: np.ndarray
    res_id_dims: tuple[int,int]
    segment: np.ndarray
    mismatch_penalty: int
    gap_penalty: int

    def __call__(self, i: int, j: int) -> tuple[int,AugmentedLetter]:
        if i == j == -1:
            return (
                self.gap_penalty,
                (-1,-1,0.)
            )
        if i == -1 or j == -1:
            idx = max(i,j)
            lo, hi = self.segment[idx:idx+2]
            edge_cost = self.cost[lo:hi]
            idx_min = edge_cost.argmin()
            min_cost = edge_cost[idx_min]
            min_anno_amino, min_anno_mod = np.unravel_index(self.residue_id[lo + idx_min],self.res_id_dims)
            min_anno_mass = self.mass[lo + idx_min]
            return (
                self.mismatch_penalty,
                (min_anno_amino,min_anno_mod,min_anno_mass),
            )
           
        i_lo, i_hi = self.segment[i:i+2]
        j_lo, j_hi = self.segment[j:j+2]
        edge_costs = self.cost[i_lo:i_hi] + self.cost[j_lo:j_hi].reshape((-1,1))
        comp_costs = 1 - (self.residue_id[i_lo:i_hi] == self.residue_id[j_lo:j_hi].reshape((-1,1)))
        costs = edge_costs + comp_costs
        i_min, j_min = np.unravel_index(costs.argmin(),costs.shape)
        min_cost = costs[i_min,j_min]
        min_anno_amino, min_anno_mod = np.unravel_index(self.residue_id[i_lo + i_min],self.res_id_dims)
        # OK to use just i_min because edge_costs are an epsilon relative to comp_costs; edge costs are used to determine which pair of matching residue ids are used for the final match, but will never dominate the mismatch cost and therefore will never lead to min_i encoding a different id than min_j.
        
        min_anno_mass = self.mass[i_lo + i_min]
        min_anno = (min_anno_amino, min_anno_mod, min_anno_mass) 
        # (int,int,float) aliased as AugmentedLetter

        return (min_cost, min_anno)

    @classmethod
    def from_annotation(
        cls,
        index: AnnotationIndex,
        residue_space: ResidueStateSpace,
        mismatch: int,
        gap: int,
    ) -> Self:
        amino_id = index.state[:,0]
        mod_id = index.state[:,1]
        amino_mass = residue_space.get_amino_mass(amino_id)
        mod_mass = residue_space.get_modification_mass(mod_id)
        n_amino = residue_space.n_aminos()
        n_mod = residue_space.n_total_modifications()
        dims = (n_amino, n_mod)
        return cls(
            cost = index.cost,
            mass = amino_mass + mod_mass,
            residue_id = np.ravel_multi_index(
                np.array([amino_id,mod_id]),
                dims,
            ),
            res_id_dims = dims,
            segment = index.inner_offset,
            mismatch_penalty = mismatch,
            gap_penalty = gap,
        )


@dataclasses.dataclass(slots=True)
class MassConstrainedPathCostModel:
    """Prunes paths whose peptide mass exceeds a given target, which is inferred from an axis of reflection. There is one path cost model per axis."""
    target_mass: float
    
    def __call__(self, peptide_mass: float, edge_anno: AugmentedLetter):
        _, _, delta_mass = edge_anno
        new_mass = peptide_mass + delta_mass
        return (
            np.inf if new_mass > self.target else 0.,
            new_mass,
        )

    @classmethod
    def initial_state(cls) -> None:
        """The mass constrained path cost model state consists only of the peptide mass of the path. The initial peptide mass is zero."""
        return 0.

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
    """Prunes paths whose peptide mass exceeds a given target, which is inferrewd from an axis of reflection, or whose amino acid sequence is not present in the given suffix array. There is one path cost model per axis, but they all share the same SuffixArray object. The SuffixArrayPathCostModel may be constructed from a MassConstrainedPathCostModel by providing a residue space and a suffix array."""
    residue_space: ResidueStateSpace
    suffix_array: SuffixArray

    def __call__(self, path_state: tuple[float,BisectResult], edge_anno: AugmentedLetter):
        pfx, mass = path_state
        amino_idx, mod_idx, step_mass = edge_anno
        amino = self.residue_space.amino_symbols[amino_idx]
        new_pfx = self.suffix_array.bisect([amino,],pfx)[0]
        # print("\t\t",amino,new_pfx.count)
        new_mass = mass + step_mass
        return (
            np.inf if (new_pfx.count == 0 or new_mass > self.target_mass) else 0.,
            (
                new_pfx,
                new_mass,
            ),
        )

    @classmethod
    def initial_state(cls) -> None:
        """The suffix array path cost model maintains a state with two components: the BisectResult describing a prefix in the suffix array, and the peptide mass. The initial BisectResult is a NoneType, representing a prefix equal to the empty string. The initial peptide mass is zero."""
        return (
            None,
            0.,
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
            residue_space,
            suffix_array,
        )
