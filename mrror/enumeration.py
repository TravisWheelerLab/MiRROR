import dataclasses, json
import itertools as it
from time import time
from typing import Self, Any

from .io import serialize_dataclass, deserialize_dataclass, SerializableDataclass

from .fragments import TargetMassStateSpace
from .graphs.dfs import dfs
from .graphs.trace import AbstractPathSpace, trace
from .sequences.suffix_array import SuffixArray

from .evaluation.costmodels import AnnotatedResiduePathCostModel, SuffixArrayPathCostModel
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
    targets: TargetMassStateSpace,
    suffix_arrays: tuple[SuffixArray,SuffixArray],
    params: EnumerationParams,
    verbose: bool = False,
) -> EnumerationResult:
    n = len(algn)

    aligned_affixes = [None for _ in range(n)]
    prefixes = [None for _ in range(n)]
    suffixes = [None for _ in range(n)]
    infixes = [None for _ in range(n)]
    affix_pairs = [None for _ in range(n)]
    candidates = [None for _ in range(n)]
    profile = {}
    # initialize return types 

    # step 1: trace aligned paths through the product topology, then cluster into prefixes, affixes, and suffixes by boundary annotation.
    t = time()
    b_anno, y_anno = targets.get_series_loss_symbols()
    suffix_array, reversed_suffix_array = suffix_arrays
    if suffix_array is None:
        for i in range(n):
            aligned_affixes[i] = trace(
                algn.prod_topology[i],
                algn.prod_topology[i].sources(),
                params.cost_threshold,
                AnnotatedResiduePathCostModel(
                    algn.prod_topology[i],
                    algn.left_topology[i],
                    algn.right_topology[i],
                    targets,
                ),
            )
            # construct path space of each aligned product topology.

            pfx, sfx, ifx = orient_affixes(aligned_affixes[i], b_anno, y_anno)
            prefixes[i] = pfx
            suffixes[i] = sfx
            infixes[i] = ifx
            # categorize as prefixes, suffixes, or infixes according to the series annotation of path boundaries.
    else:
        for i in range(n):
            reverse_aligned_affixes = trace(
                algn.prod_topology[i],
                algn.prod_topology[i].sources(),
                params.cost_threshold,
                SuffixArrayPathCostModel(
                    reversed_suffix_array,
                    algn.prod_topology[i],
                    algn.left_topology[i],
                    algn.right_topology[i],
                    targets,
                ),
            )
            # trace suffixes through the forward suffix array.

            forward_aligned_affixes = trace(
                algn.prod_topology[i],
                algn.prod_topology[i].sources(),
                params.cost_threshold,
                SuffixArrayPathCostModel(
                    suffix_array,
                    algn.prod_topology[i],
                    algn.left_topology[i],
                    algn.right_topology[i],
                    targets,
                ),
            )
            # trace prefixes through the reversed suffix array.

            aln, pfx, sfx, inf = refine_affixes(
                reverse_aligned_affixes,
                forward_aligned_affixes,
                b_anno,
                y_anno,
            )
            aligned_affixes[i] = aln
            prefixes[i] = pfx
            suffixes[i] = sfx
            infixes[i] = ifx
            # refine forward and reverse affixes into prefix, suffix, and infix categories.
    profile["trace"] = time() - t
    # end step 1, print results.
    if verbose:
        for (a,p,s,i) in zip(aligned_affixes,prefixes,suffixes,infixes):
            for (tag,afx) in (("prefix",p),("suffix",s),("infix",i)):
                for (x,y) in afx:
                    cost, __, annotation = a[x][:3]
                    anno_res = [u[:,0] for u in annotation]
                    anno_loss = [u[:,2] for u in annotation]
                    term = anno_loss[-1][y]
                    print(f"{tag} {x} {y} {cost} {[v[0] for v in anno_res]} {term}")

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
            aligned_affixes[i],
            prefixes[i],
            suffixes[i],
            affix_pairs[i],
            anno.pairs,
            targets,
        )
        # 

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
