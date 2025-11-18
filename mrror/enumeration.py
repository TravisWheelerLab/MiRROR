import dataclasses, json
from time import time
from typing import Self, Any
import itertools as it

from .io import serialize_dataclass, deserialize_dataclass, SerializableDataclass
from .fragments import TargetMassStateSpace
from .graphs.types import PathSpace
from .graphs.dfs import dfs
from .graphs.trace import trace
from .sequences.suffix_array import SuffixArray
from .costmodels import OrderedResiduePathCostModel, SuffixArrayPathCostModel
from .annotation import AnnotationResult
from .alignment import AlignmentResult

@dataclasses.dataclass(slots=True)
class EnumerationResult(SerializableDataclass):
    affixes: list[PathSpace]   # everything
    prefixes: list[PathSpace]   # aligned peptide prefixes
    suffixes: list[PathSpace]   # aligned peptide suffixes
    infixes: list[PathSpace]    # alignments that could not be categorized as prefix or suffix
    candidates: list            # 
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
    suffix_array, reversed_suffix_array = suffix_arrays
    if suffix_array is None:
        affixes = [trace(
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
        # construct the constrained path space of each aligned product topology.
        prefixes = [PathSpace.empty() for _ in range(len(algn))]
        suffixes = [PathSpace.empty() for _ in range(len(algn))]
        infixes = [PathSpace.empty() for _ in range(len(algn))]
        # categorize as prefixes, suffixes, or infixes.
    else:
        prefixes = [trace(
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
        suffixes = [trace(
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
        infixes = [PathSpace.empty() for _ in range(len(algn))]
        # no infixes are produced with this method.
        
    profile["trace"] = time() - t
    if verbose:
        print(prefixes,suffixes,infixes)
    # generate alignments between low- and high-mz graphs.

    if verbose:
        print(profile)
    return EnumerationResult(
        affixes = affixes,
        prefixes = prefixes,
        suffixes = suffixes,
        infixes = infixes,
        candidates = [],
        _profile = profile,
    )
