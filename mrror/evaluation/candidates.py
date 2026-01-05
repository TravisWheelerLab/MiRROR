import dataclasses
import itertools as it
from typing import Iterable, Iterator, Self

import numpy as np

from ..util import combine_symbols, combine_masses
from ..spectra.types import Peaks
from ..fragments.types import TargetMasses, PairResult, BoundaryResult
from ..graphs.types import SpectrumGraph, WeightedProductGraph
from ..graphs.trace import AbstractPathSpace

from .pathspaces import AnnotatedResiduePathSpace, SuffixArrayPathSpace
from .costmodels import AnnotatedResiduePathCostModel, SuffixArrayPathCostModel, MISMATCH_SEPARATOR

@dataclasses.dataclass(slots=True)
class CandidateResult:
    mass: np.ndarray
    sequence: str
    seq_segment: np.ndarray
    path: np.ndarray
    path_segment: np.ndarray
    path_pivot_idx: np.ndarray
    annotation: list[np.ndarray]
    offset: np.ndarray
    cost: np.ndarray

    def __len__(self) -> int:
        return len(self.mass)

    def __getitem__(self, i: int) -> tuple[float,str,np.ndarray,float]:
        l, r = self.seq_segment[i:i+2]
        l2, r2 = self.path_segment[i:i+2]
        return (
            self.cost[i],
            self.sequence[l:r],
            self.path[l2:r2],
            self.annotation[l:r],
            self.offset[i],
        )

    def __iter__(self) -> Iterator:
        return (self.__getitem__(i) for i in range(len(self)))

    def get_sequence(self, i: int) -> str:
        l, r = self.seq_segment[i:i+2]
        return self.sequence[l:r]

    def _peak_annotation(
        self,
        path: Iterable[int],
        peaks: Peaks,
        topology: SpectrumGraph,
        pairs: PairResult,
        boundaries: BoundaryResult,
        pair_component: int, # 0 or 1
    ) -> np.ndarray:
        weights = [topology.get_weight(j, i) for (i, j) in it.pairwise(path)]
        boundary_index = boundaries.index[weights[-1][0]]
        boundary_charge = boundaries.charge[weights[-1][0]]
        pair_indices = [pairs.indices[x,pair_component].tolist() for x in weights[:-1]]
        pair_charges = [pairs.charges[x,pair_component].tolist() for x in weights[:-1]]
        return (
            np.array(pair_indices + [boundary_index,]),
            np.array(pair_charges + [boundary_charge,]),
        )

    def _get_peaks(
        self,
        path: Iterable[int],
        peaks: Peaks,
        pivot_idx: int,
        topology: SpectrumGraph,
        pairs: PairResult,
        boundaries: BoundaryResult,
        pair_component: int # 0 or 1
    ) -> tuple[np.ndarray, np.ndarray]:
        prefix_path = path[:pivot_idx][::-1]
        prefix_peak_idx, prefix_charge = self._peak_annotation(
            prefix_path,
            peaks,
            topology,
            pairs,
            boundaries,
            pair_component,
        )
        prefix_mz = peaks.mz[prefix_peak_idx]
        prefix_intensity = peaks.intensity[prefix_peak_idx]
        # flip the first half of the path and recover its data.
        suffix_path = path[pivot_idx:]
        suffix_peak_idx, suffix_charge = self._peak_annotation(
            suffix_path,
            peaks,
            topology,
            pairs,
            boundaries,
            pair_component,
        )
        suffix_mz = peaks.mz[suffix_peak_idx]
        suffix_intensity = peaks.intensity[suffix_peak_idx]
        # recover the data for the second half of the path.
        return (
            np.concat([prefix_mz[::-1], suffix_mz]),
            np.concat([prefix_intensity[::-1], suffix_intensity]),
            np.concat([prefix_charge[::-1], suffix_charge])
        )

    def get_peaks(
        self,
        i: int,
        peaks: Peaks,
        pairs: PairResult,
        lower_boundaries: BoundaryResult,
        upper_boundaries: BoundaryResult,
        prod_topology: WeightedProductGraph,
        lower_topology: SpectrumGraph,
        upper_topology: SpectrumGraph,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reconstruct the m/z and intensity values of the peaks supporting a given candidate.
        NOTE: the arrays returned by this function are ordered according to the candidate path in the product topology, not by m/z. The output takes the form (
            np.concat([left_mz, right_mz]),
            np.concat([left_intensity, right_intensity]),
        )."""
        l, r = self.path_segment[i:i+2]
        prod_path = self.path[l:r]
        left_path, right_path = zip(*[prod_topology.unravel(x) for x in prod_path])
        # recover the path and unravel it
        path_pivot = self.path_pivot_idx[i]
        # describes where the direction of the path flips
        left_mz, left_intensity, left_charge = self._get_peaks(
            left_path,
            peaks,
            path_pivot,
            lower_topology,
            pairs,
            lower_boundaries,
            pair_component = 1,
        )
        right_mz, right_intensity, right_charge = self._get_peaks(
            right_path,
            peaks,
            path_pivot,
            upper_topology,
            pairs,
            upper_boundaries,
            pair_component = 0,
        )
        # retrieve data for left and right components of path.
        return (
            np.concat([left_mz, right_mz]),
            np.concat([left_intensity, right_intensity]),
            np.concat([left_charge, right_charge]),
        )

    def get_series(self, i: int) -> np.ndarray:
        """Annotate the series of each peak supporting a given candidate."""
        seq = self.get_sequence(i)
        return np.concat([
            ['b','y'] 
            for _ in range(self.seq_segment[i + 1] - self.seq_segment[i] - 1)
        ])

    def get_position(self, i: int) -> np.ndarray:
        """Annotate the position of each peak supporting a given candidate."""
        return 1 + np.concat([
            [x, x] 
            for x in range(self.seq_segment[i + 1] - self.seq_segment[i] - 1)
        ])

    @classmethod
    def from_list(
        cls,
        candidates: list[tuple[float,str,np.ndarray,float,float]],
    ):
        masses, paths, path_pivots, sequences, annotations, offsets, costs = [np.array(x,dtype=object) for x in zip(*candidates)]
        print("ANNOTATIONS", len(annotations))
        order = np.argsort(costs)
        return cls(
            mass = masses[order],
            sequence = ''.join(sequences[order]),
            seq_segment = np.cumsum([0,] + [len(x) for x in sequences[order]]),
            path = np.concat(paths[order]),
            path_segment = np.cumsum([0,] + [len(x) for x in paths[order]]),
            path_pivot_idx = path_pivots[order],
            annotation = annotations[order],
            offset = offsets[order],
            cost = costs[order],
        )

    @classmethod
    def empty(cls) -> Self:
        return cls(
            mass = np.array([]),
            sequence = "",
            seq_segment = np.array([0]),
            path = np.array([]),
            path_segment = np.array([0]),
            path_pivot_idx = 0,
            annotation = np.array([]),
            offset = np.array([],dtype=float),
            cost = np.array([np.inf]),
        )

def _retrieve_pivot_annotations(
    pivot_pair_ids: np.ndarray,
    pairs: PairResult,
    targets: TargetMasses,
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    unique_ids, reindexer = np.unique_inverse(pivot_pair_ids)
    # pivot_pair_ids is an (n,2) array. unique_ids is one dimensional. reindexer is an (n,2) array associating each cell of pivot_pair_ids to a cell in unique_ids.
    pair_anno = [pairs.get_annotation(i) for i in unique_ids]
    symbols = [targets.get_state_symbols(x) for (_,x) in pair_anno]
    masses = [targets.get_state_weights(x) for (_,x) in pair_anno]
    return (
        reindexer,
        pair_anno,
        symbols,
        masses,
    )

# this function reimplements many features implemented in the Annotated[...]CostModel classes. TODO: abstract both implementations elsewhere.
def generate_candidates(
    pivot_cluster: np.ndarray,
    aligned_affixes: AbstractPathSpace,
    prefixes: np.ndarray,
    suffixes: np.ndarray,
    affix_pairs: np.ndarray,
    pairs: PairResult,
    pair_targets: TargetMasses,
    sep: str = MISMATCH_SEPARATOR,
) -> CandidateResult:
    if len(prefixes) == 0 or len(suffixes) == 0 or len(aligned_affixes) == 0 or len(affix_pairs) == 0:
        return CandidateResult.empty()
    pivot_ids = affix_pairs[:,2:]
    reindexed_pivot_ids, pivot_anno, pivot_symbols, pivot_masses = _retrieve_pivot_annotations(pivot_ids, pairs, pair_targets)
    n = len(affix_pairs)
    candidates = []
    for i in range(n):
        left_anno_id, right_anno_id = reindexed_pivot_ids[i]
        # unpack pivot indices.
        
        left_cost, left_anno = pivot_anno[left_anno_id]
        right_cost, right_anno = pivot_anno[right_anno_id]
        pivot_costs = left_cost + right_cost
        left_sym = pivot_symbols[left_anno_id]
        right_sym = pivot_symbols[right_anno_id]
        pivot_sym = combine_symbols(left_sym, right_sym, sep)
        left_mass = pivot_masses[left_anno_id]
        right_mass = pivot_masses[right_anno_id]
        pivot_mass = combine_masses(left_mass, right_mass)
        pivot_order = np.argsort(pivot_costs)
        pivot_sym = pivot_sym[pivot_order]
        pivot_mass = pivot_mass[pivot_order]
        pivot_cost = pivot_costs.min()
        # construct pivot costs, symbols, and masses. reorder by cost.
        
        p_idx, s_idx = affix_pairs[i,:2]
        prefix_id, terminal_anno_id = prefixes[p_idx]
        suffix_id, terminal_anno_id = suffixes[s_idx]
        prefix_cost, prefix_path, prefix_sym, prefix_masses = aligned_affixes[prefix_id][:4]
        suffix_cost, suffix_path, suffix_sym, suffix_masses = aligned_affixes[suffix_id][:4]
        # retrieve affix costs, symbols, and masses.
       
        candidate_masses = prefix_masses[1:][::-1] + [pivot_mass,] + suffix_masses[1:]
        candidate_path = np.concat([prefix_path[::-1], suffix_path])
        candidate_annotation = prefix_sym[1:][::-1] + [pivot_sym,] + suffix_sym[1:]
        candidate_cost = prefix_cost + pivot_cost + suffix_cost
        # combine affix and pivot data. prefixes need to be reversed. all affixe annotations begin with a null character, due to the pivot sink edge, which must be trimmed.
        
        mass_seq = [x[0] for x in candidate_masses]
        observed_mass = np.sum(mass_seq) / 2
        expected_mass = 2 * pivot_cluster
        mass_offset_cluster = observed_mass - expected_mass
        optimal_pivot = np.abs(mass_offset_cluster).argmin()
        candidate_expected_mass = expected_mass[optimal_pivot]
        candidate_mass_offset = mass_offset_cluster[optimal_pivot]
        # form candidate.

        greedy_seq = ''.join([x[0][0].strip() for x in candidate_annotation])
        # assemble a greedy candidate sequence.

        candidates.append((
            candidate_expected_mass,            # mass
            candidate_path,                     # path
            len(prefix_path),                   # path pivot
            greedy_seq,                         # sequence
            candidate_annotation,               # annotation
            candidate_mass_offset,              # offset from pivot mass
            candidate_cost,                     # integrated cost
        ))

    return CandidateResult.from_list(candidates)
