import dataclasses
from typing import Iterator, Self

import numpy as np

from ..util import combine_symbols
from ..fragments.types import TargetMassStateSpace, PairResult
from ..graphs.trace import AbstractPathSpace

from .pathspaces import AnnotatedResiduePathSpace, SuffixArrayPathSpace
from .costmodels import AnnotatedResiduePathCostModel, SuffixArrayPathCostModel, MISMATCH_SEPARATOR

@dataclasses.dataclass(slots=True)
class CandidateResult:
    mass: float
    sequence: str
    annotation: list[np.ndarray]
    segment: np.ndarray
    offset: np.ndarray
    cost: np.ndarray

    def __len__(self) -> int:
        return len(self.segment) - 1

    def __getitem__(self, i: int) -> tuple[float,str]:
        l, r = self.segment[i:i+2]
        return (
            self.cost[i],
            self.sequence[l:r],
            self.annotation[l:r],
            self.offset[i],
        )

    def __iter__(self) -> Iterator:
        return (self.__getitem__(i) for i in range(len(self)))

    @classmethod
    def from_list(
        cls,
        candidates: list[tuple[float,str,np.ndarray,float,float]],
    ):
        masses, sequences, annotations, offsets, costs = [np.array(x,dtype=object) for x in zip(*candidates)]
        order = np.argsort(costs)
        return cls(
            mass = masses[order],
            sequence = ''.join(sequences[order]),
            annotation = annotations[order],
            segment = np.cumsum([0,] + [len(x) for x in sequences[order]]),
            offset = offsets[order],
            cost = costs[order],
        )

    @classmethod
    def empty(cls) -> Self:
        return cls(
            mass = 0.,
            sequence = "",
            annotation = np.array([]),
            segment = np.array([0]),
            offset = np.array([],dtype=float),
            cost = np.array([np.inf]),
        )

def _retrieve_pivot_annotations(
    pivot_pair_ids: np.ndarray,
    pairs: PairResult,
    targets: TargetMassStateSpace,
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    unique_ids, reindexer = np.unique_inverse(pivot_pair_ids)
    # pivot_pair_ids is an (n,2) array. unique_ids is one dimensional. reindexer is an (n,2) array associating each cell of pivot_pair_ids to a cell in unique_ids.
    pair_anno = [pairs.get_annotation(i) for i in unique_ids]
    symbols = [targets.symbolize_pairs(x) for (_,x) in pair_anno]
    return (
        reindexer,
        pair_anno,
        symbols,
    )

# this function reimplements many features implemented in the Annotated[...]CostModel classes. TODO: abstract both implementations elsewhere.
def generate_candidates(
    aligned_affixes: AbstractPathSpace,
    prefixes: np.ndarray,
    suffixes: np.ndarray,
    affix_pairs: np.ndarray,
    pairs: PairResult,
    targets: TargetMassStateSpace,
    sep: str = MISMATCH_SEPARATOR,
) -> CandidateResult:
    if len(prefixes) == 0 or len(suffixes) == 0 or len(aligned_affixes) == 0 or len(affix_pairs) == 0:
        return CandidateResult.empty()
    pivot_ids = affix_pairs[:,2:]
    reindexed_pivot_ids, pivot_anno, pivot_symbols = _retrieve_pivot_annotations(pivot_ids, pairs, targets)
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
        pivot_order = np.argsort(pivot_costs)
        pivot_sym = pivot_sym[pivot_order]
        pivot_cost = pivot_costs.min()
        # construct pivot costs and symbols, reorder by cost.
        p_idx, s_idx = affix_pairs[i,:2]
        prefix_id, terminal_anno_id = prefixes[p_idx]
        suffix_id, terminal_anno_id = suffixes[s_idx]
        prefix_cost, _, prefix_sym, __ = aligned_affixes[prefix_id]
        suffix_cost, _, suffix_sym, __ = aligned_affixes[suffix_id]
        # retrieve affix costs and symbols.
        candidate_cost = prefix_cost + pivot_cost + suffix_cost
        candidate_annotation = prefix_sym[::-1] + [pivot_sym,] + suffix_sym
        # form candidate. TODO: 1. calculate masses. 2. if using suffix array cost model, retrieve sequences.
        seq = [x[0][0] for x in candidate_annotation]
        candidates.append((
            0.,
            ' '.join(seq),
            candidate_annotation,
            0.,
            candidate_cost,
        ))
    return CandidateResult.from_list(candidates)
