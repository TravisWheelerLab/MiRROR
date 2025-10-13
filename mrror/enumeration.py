import dataclasses, json
from time import time
from typing import Self, Any

from .fragments.types import TargetMassStateSpace
from .graphs.dfs import dfs
from .alignment import AlignmentResult

@dataclasses.dataclass(slots=True)
class EnumerationResult:
    candidates: list

@dataclasses.dataclass(slots=True)
class EnumerationParams:
    cost_threshold: float
    # suffix_array: SuffixArray

    @classmethod
    def from_config(cls, cfg):
        return cls(
            cost_threshold = cfg['cost_threshold'],
        )

def enumerate_candidates(
    algn: AlignmentResult,
    targets: TargetMassStateSpace,
    params: EnumerationParams,
    verbose: bool = False,
) -> EnumerationResult:
    profile = {}

    t = time()
    aligned_paths = [dfs(
        topology = prod,
        sources = [x for x in prod.graph if prod.graph.in_degree(x) == 0],
        threshold = params.cost_threshold,
    ) for (prod, lo, hi) in zip(algn.sparse_prod, algn.lo_adj, algn.hi_adj)]
    profile["dfs"] = time() - t
    if verbose:
        print(aligned_paths)
        for (graph, pathspace) in zip(algn.sparse_prod,aligned_paths):
            pathspace = sorted(pathspace,key=lambda x: -x[0])
            for cost, path in pathspace:
                unraveled_path = [graph.unravel(x) for x in path]
                print(cost, [(int(u),int(w)) for (u,w) in unraveled_path])
    # generate alignments between low- and high-mz graphs.

    if verbose:
        print(profile)
    return EnumerationResult(
        candidates = [],
    )
