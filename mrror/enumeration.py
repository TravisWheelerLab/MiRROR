import dataclasses, json
from time import time
from typing import Self, Any
import itertools as it

from .io import serialize_dataclass, deserialize_dataclass, SerializableDataclass
from .fragments import TargetMassStateSpace
from .graphs.dfs import dfs
from .graphs.trace import AbstractPathSpace, trace
from .sequences.suffix_array import SuffixArray
from .costmodels import AnnotatedResiduePathCostModel, SuffixArrayPathCostModel
from .affix_pairing import orient_affixes, refine_affixes, pair_affixes
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
    candidates: list
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
    profile = {}

    t = time()
    b_anno, y_anno = targets.get_series_loss_symbols()
    suffix_array, reversed_suffix_array = suffix_arrays
    if suffix_array is None:
        aligned_affixes = [trace(
                prod,
                [x for x in prod.graph if prod.graph.in_degree(x) == 0],
                params.cost_threshold,
                AnnotatedResiduePathCostModel(
                    prod,
                    left,
                    right,
                    targets,
                ),
            ) for (i, (prod, left, right)) in enumerate(zip(algn.prod_topology, algn.left_topology, algn.right_topology))]
        # construct the constrained path space of each aligned product topology.
        prefixes, suffixes, infixes = zip(*[orient_affixes(x, b_anno, y_anno) for x in aligned_affixes])
        # categorize as prefixes, suffixes, or infixes.
    else:
        reverse_aligned_affixes = [trace(
                prod,
                [x for x in prod.graph if prod.graph.in_degree(x) == 0],
                params.cost_threshold,
                SuffixArrayPathCostModel(
                    reversed_suffix_array,
                    prod,
                    left,
                    right,
                    targets,
                ),
            ) for (i, (prod, left, right)) in enumerate(zip(algn.prod_topology, algn.left_topology, algn.right_topology))]
        # construct path space of peptide suffixes from the suffix array path.
        forward_aligned_affixes = [trace(
                prod,
                [x for x in prod.graph if prod.graph.in_degree(x) == 0],
                params.cost_threshold,
                SuffixArrayPathCostModel(
                    suffix_array,
                    prod,
                    left,
                    right,
                    targets,
                ),
            ) for (i, (prod, left, right)) in enumerate(zip(algn.prod_topology, algn.left_topology, algn.right_topology))]
        # construct path space of peptide prefixes from the reversed suffix array.
        zip_afx = zip(reverse_aligned_affixes, forward_aligned_affixes)
        aligned_affixes, prefixes, suffixes, infixes = zip(*[refine_affixes(rev_afx, fwd_afx, b_anno, y_anno) for (rev_afx, fwd_afx) in zip_afx])
        # refine forward and reverse affixes into prefix, suffix, and infix categories.
    profile["trace"] = time() - t
    # generate alignments between low- and high-mz graphs and categorize into prefix and suffix alignments.
    if verbose:
        for (a,p,s,i) in zip(aligned_affixes,prefixes,suffixes,infixes):
            for (tag,afx) in (("prefix",p),("suffix",s),("infix",i)):
                for (x,y) in afx:
                    cost, __, anno = a[x][:3]
                    anno_res = [u[:,0] for u in anno]
                    anno_loss = [u[:,2] for u in anno]
                    term = anno_loss[-1][y]
                    print(f"{tag} {x} {y} {cost} {[v[0] for v in anno_res]} {term}")

    affix_pairs = [pair_affixes(pfx, sfx, aln_afx, prod, pvt) for (pfx, sfx, aln_afx, prod, pvt) in zip(prefixes,suffixes,aligned_affixes,algn.prod_topology,algn.pivot_topology)]
    # pair affixes.

    # TODO
    # sort and enumerate candidates.
    if verbose:
        print(profile)
    return EnumerationResult(
        aligned_affixes = aligned_affixes,
        prefixes = prefixes,
        suffixes = suffixes,
        infixes = infixes,
        affix_pairs = affix_pairs,
        candidates = [],
        _profile = profile,
    )
