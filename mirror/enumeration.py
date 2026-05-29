import dataclasses, json
import itertools as it
from time import time
from typing import Self, Any

from .io import serialize_dataclass, deserialize_dataclass, SerializableDataclass

from .fragments.types import TargetMasses
from .graphs.trace import AbstractPathSpace, trace
from .sequences.suffix_array import SuffixArray

from .evaluation.affix_pairing import orient_affixes, refine_affixes, pair_affixes
from .evaluation.candidates import CandidateResult, generate_candidates

from .annotation import AnnotationResult
from .alignment import AlignmentResult

import numpy as np

@dataclasses.dataclass(slots=True)
class EnumerationResult(SerializableDataclass):
    aligned_affixes: list[AbstractPathSpace]
    prefixes: list[np.ndarray]
    suffixes: list[np.ndarray]
    infixes: list[np.ndarray]
    affix_pairs: list[np.ndarray]
    candidates: list[CandidateResult]
    _profile: dict[str,float]

    def __len__(self) -> int:
        return len(self.candidates)

@dataclasses.dataclass(slots=True)
class EnumerationParams(SerializableDataclass):
    cost_threshold: float

    @classmethod
    def from_config(cls, cfg):
        return cls(
            cost_threshold = cfg['cost_threshold'],
        )

def enumerate_candidates(
    anno: AnnotationResult,
    algn: AlignmentResult,
    suffix_arrays: tuple[SuffixArray,SuffixArray],
    params: EnumerationParams,
    verbose: bool = False,
) -> EnumerationResult:
    n = len(algn)

    prefixes = None
    suffixes = None
    aligned_affixes = None
    # TODO - refactor with new AlignmentResult containing pathspaces.
    # step 2: combine affixes and pivot edges to construct peptide sequences.
    for i in range(n):
        affix_pairs[i] = pair_affixes(
            prefixes[i],
            suffixes[i],
            aligned_affixes[i],
            algn.prod_topology[i],
            algn.pivot_topology[i],
        )
        # connect prefix and suffix paths across edges in the pivot topology.

        candidates[i] = generate_candidates(
            anno.pivots.pivot_points[anno.pivots.clusters[i]],
            aligned_affixes[i],
            prefixes[i],
            suffixes[i],
            affix_pairs[i],
            anno.pairs,
            anno_params.pair_target_masses,
        )
        # concatenate affix pairs with pivot annotations to product full candidate sequences.

        # candidates[i] = rescore_candidates()
        # TODO: score, filter, and re-order the candidates.
    # end step 2, print results and timing data.
    if verbose:
        for i in range(n):
            for j in range(len(candidates[i])):
                print(i, j, candidates[i][j][:2])
        print(profile)
    return EnumerationResult(
        aligned_affixes = aligned_affixes,
        prefixes = prefixes,
        suffixes = suffixes,
        infixes = infixes,
        affix_pairs = affix_pairs,
        candidates = candidates,
        _profile = profile,
    )
