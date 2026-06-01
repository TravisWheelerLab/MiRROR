import abc
import dataclasses
from typing import Union, Any, Iterator, Self

import numpy as np

from ..util import merge_compare_exact_unique, ravel, unravel, combine_symbols, combine_masses
from ..fragments.types import ResidueStateSpace
from ..sequences.suffix_array import SuffixArray, BisectResult
from ..fragments.types import TargetMasses, BoundaryResult, PairResult, UniqueFragmentIndex, AnnotationIndex
from ..graphs.types import SpectrumGraph, PivotGraph, SymmetricGraph
from ..graphs.align import AbstractNodeCostModel, AbstractEdgeCostModel, AbstractPathCostModel, AugmentedLetter

@dataclasses.dataclass(slots=True)
class SymmetricNodeCostModel:
    """Assigns pairs of nodes in the product graph a cost based on their reflection symmetry under a given axis. There is one node cost model per axis."""
    lower_mass: np.ndarray
    upper_mass: np.ndarray
    reflector: float
    tolerance: float

    def __call__(self, i: int, j: int) -> float:
        return 0. if abs(self.lower_mass[i] + self.upper_mass[j] - self.reflector) < self.tolerance else 1.

    # TODO - constructor

@dataclasses.dataclass(slots=True)
class AnnotatedEdgeCostModel:
    """Assigns pairs of edges in the product graph a cost based on the similarity of their annotation and the cost of those annotations. There is one edge cost model per spectrum."""
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

    @classmethod
    def from_annotation(
        cls,
        index: AnnotationIndex,
        residue_space: ResidueStateSpace,
    ) -> Self:
        amino_id = index.state[:,0]
        mod_id = index.state[:,1]
        amino_mass = residue_space.get_amino_mass(amino_id)
        mod_mass = residue_space.get_mod_mass(mod_id)
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
        )

@dataclasses.dataclass(slots=True)
class MassConstrainedPathCostModel:
    """Prunes paths whose peptide mass exceeds a given target, which is inferred from an axis of reflection. There is one path cost model per axis."""
    target_mass: float
    tolerance: float

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

@dataclasses.dataclass(slots=True)
class SuffixArrayPathCostModel(MassConstrainedPathCostModel):
    """Prunes paths whose peptide mass exceeds a given target, which is inferred from an axis of reflection, or whose amino acid sequence is not present in the given suffix array. There is one path cost model per axis, but they all share the same SuffixArray object."""
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
