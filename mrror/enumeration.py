import dataclasses, json
from time import time
from typing import Self, Any
import itertools as it

from .io import serialize_dataclass, deserialize_dataclass, SerializableDataclass
from .fragments import TargetMassStateSpace
from .graphs.dfs import dfs
from .graphs.trace import trace
from .sequences.suffix_array import SuffixArray
from .costmodels import OrderedResiduePathCostModel, SuffixArrayPathCostModel
from .annotation import AnnotationResult
from .alignment import AlignmentResult

@dataclasses.dataclass(slots=True)
class EnumerationResult(SerializableDataclass):
    aligned_paths: list
    candidates: list
    _profile: dict[str,float]

@dataclasses.dataclass(slots=True)
class EnumerationParams(SerializableDataclass):
    cost_threshold: float
    suffix_array: SuffixArray

    @classmethod
    def from_config(cls, cfg):
        suf_path = cfg['suffix_array']
        # suffix_array = SuffixArray. TODO
        return cls(
            cost_threshold = cfg['cost_threshold'],
            suffix_array = None,
        )

def enumerate_candidates(
    anno: AnnotationResult,
    algn: AlignmentResult,
    targets: TargetMassStateSpace,
    params: EnumerationParams,
    verbose: bool = False,
) -> EnumerationResult:
    profile = {}

    t = time()
    aligned_paths = [trace(
            prod,
            [x for x in prod.graph if prod.graph.in_degree(x) == 0],
            params.cost_threshold,
            OrderedResiduePathCostModel(
                prod,
                left,
                right,
                targets,
            ),
        ) for (i, (prod, left, right)) in enumerate(zip(algn.prod_topology, algn.left_topology, algn.right_topology))]
    profile["trace"] = time() - t
    if verbose:
        pass
    # generate alignments between low- and high-mz graphs.

    if verbose:
        print(profile)
    return EnumerationResult(
        aligned_paths = aligned_paths,
        candidates = [],
        _profile = profile,
    )
