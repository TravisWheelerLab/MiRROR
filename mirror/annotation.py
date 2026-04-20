import dataclasses, json
from time import time
from typing import Self, Any
# standard

from .util import normalize_dict
from .io import serialize_dataclass, deserialize_dataclass, SerializableDataclass
from .spectra.types import Peaks, AugmentedPeaks
from .fragments.types import FragmentStateSpace, ResidueStateSpace, TargetMasses, MultiResidueTargetMasses, PairResult, PivotResult, BoundaryResult
from .fragments.masses import construct_pair_target_masses, construct_boundary_target_masses
from .fragments.search import find_pairs, find_pivots, find_boundaries 
from .sequences.suffix_array import SuffixArray
from .sequences.queries import PeptideMassQueryEngine, all_kmers
from .graphs.types import SpectrumGraph, PivotGraph
from .evaluation.labeled_peaks import FragmentLabels
from .evaluation.spectrum_topology import construct_spectrum_topology
# local

import numpy as np
from omegaconf.dictconfig import DictConfig

@dataclasses.dataclass(slots=True)
class AnnotationResult(SerializableDataclass):
    peaks: Peaks
    pairs: PairResult
    pivots: PivotResult
    lower_boundaries: BoundaryResult
    upper_boundaries: list[BoundaryResult]
    fragment_labels: FragmentLabels
    lower_topology: list[SpectrumGraph]
    upper_topology: list[SpectrumGraph]
    pivot_topology: list[PivotGraph]
    symmetric_nodes: list[np.ndarray]
    # every list has len(self.pivots) items.
    
    _profile: dict[str,float] = None

    def __len__(self) -> int:
        return len(self.pivots)

    @classmethod
    def from_data(cls,
        peaks: Peaks,
        pairs: PairResult,
        pivots: PivotResult,
        lower_boundaries: BoundaryResult,
        upper_boundaries: list[BoundaryResult],
        fragment_labels: FragmentLabels,
        lower_topology: list[SpectrumGraph],
        upper_topology: list[SpectrumGraph],
        pivot_topology: list[PivotGraph],
        symmetric_nodes: list[np.ndarray],
        profile: dict[str,float],
    ) -> Self:
        assert len(pivots) == len(upper_boundaries)
        return cls(
            peaks = peaks.to_peaks(),
            pairs = pairs,
            pivots = pivots,
            lower_boundaries = lower_boundaries,
            upper_boundaries = upper_boundaries,
            fragment_labels = fragment_labels,
            lower_topology = lower_topology,
            upper_topology = upper_topology,
            pivot_topology = pivot_topology,
            symmetric_nodes = symmetric_nodes,
            _profile = profile,
        )

@dataclasses.dataclass(slots=True)
class AnnotationParams(SerializableDataclass):
    target_masses: list[TargetMasses]
    # target masses for pairs, lower boundaries, and reflected upper boundaries.
    max_k: int
    # .
    charges: np.ndarray
    # list of (positive) charges expected on fragments. typically [1,], [1,2,], or [1,2,3].
    query_tolerance: float
    # the radius around a query in which hits are collected.
    symmetry_tolerance: float
    # the max distance between a value and another reflected value such that they are considered symmetric.
    pivot_score_factor: float
    # tunes the pivot symmetry score threshold := max score * score factor. TODO - this might not work for noisier spectra.
    suffix_array: SuffixArray

    def _targets(self, i: int, k: int, stride=3) -> tuple[TargetMasses,int]:
        if k > self.max_k:
            raise ValueError(f"residue length {k} exceeds the maximum {self.max_k}.")
        index = i + ((k - 1) * stride)
        return (
            self.target_masses[index],
            index,
        )

    def pair_targets(self, k=1) -> tuple[TargetMasses,int]:
        return self._targets(0, k)

    def boundary_targets(self, k=1) -> tuple[TargetMasses,int]:
        return self._targets(1, k)

    def reflected_boundary_targets(self, k=1) -> tuple[TargetMasses,int]:
        return self._targets(2, k)

    @classmethod
    def from_config(cls,
        cfg: DictConfig,
        suffix_array: SuffixArray = None,
    ) -> Self:
        pair_fragment_space = FragmentStateSpace.from_config_to_pairs(cfg)
        residue_space = ResidueStateSpace.from_config(cfg)
        pair_targets = construct_pair_target_masses(
            residue_space,
            pair_fragment_space,
        )
        # target masses for fragments observed as the difference of two consecutive peaks.

        lower_boundary_fragment_space = FragmentStateSpace.from_config_to_boundaries(cfg)
        lower_boundary_targets = construct_boundary_target_masses(
            residue_space,
            lower_boundary_fragment_space,
        )
        # target masses for low-mz boundaries observed as single peaks.

        reflected_residue_space = ResidueStateSpace.from_config(cfg, reflect=True)
        reflected_upper_boundary_fragment_space = FragmentStateSpace.from_config_to_boundaries(cfg, reflect=True)
        reflected_upper_boundary_targets = construct_boundary_target_masses(
            reflected_residue_space,
            reflected_upper_boundary_fragment_space,
        )
        # target masses for high-mz boundaries observed as the reflections of single peaks.

        max_k = cfg.max_k
        multi_pair_targets = [None for _ in range(max_k - 1)]
        multi_lower_boundary_targets = [None for _ in range(max_k - 1)]
        multi_reflected_upper_boundary_targets = [None for _ in range(max_k - 1)]
        if not(suffix_array) is None and max_k > 1:
            for k in range(2, max_k + 1):
                operand = [pair_targets,] * (k - 1)
                multi_pair_targets[k - 2] = combine_target_masses(
                    [pair_targets,] + operand)
                multi_lower_boundary_targets[k - 2] = combine_target_masses(
                    [boundary_targets,] + operand)
                multi_reflected_upper_boundary_targets[k - 2] = combine_target_masses(
                    [reflected_upper_boundary_targets,] + operand)

        return cls(
            target_masses = [
                pair_targets,
                *multi_pair_targets,
                lower_boundary_targets,
                *multi_lower_boundary_targets,
                reflected_upper_boundary_targets,
                *multi_reflected_upper_boundary_targets,
            ],
            max_k = max_k,
            charges = np.array(cfg.charges),
            query_tolerance = cfg.query_tolerance,
            symmetry_tolerance = cfg.symmetry_tolerance,
            pivot_score_factor = cfg.pivot_score_factor,
            suffix_array = suffix_array,
        )

def annotate(
    peaks: Peaks,
    params: AnnotationParams,
    verbose: bool = False,
) -> AnnotationResult:
    profile = {}
    
    t = time()
    decharged_peaks = AugmentedPeaks.from_peaks(
        peaks,
        charges=params.charges,
    )
    profile["decharged_peaks"] = time() - t
    # construct augmented mz and target masses

    t = time()
    max_k = params.max_k
    pairs = [None for _ in range(max_k)]
    for i in range(max_k):
        k = i + 1
        pair_targets, pair_index = params.pair_targets(k)
        pairs[i] = find_pairs(
            peaks = decharged_peaks,
            tolerance = params.query_tolerance,
            targets = pair_targets,
            targets_index = pair_index,
        )
    single_residue_pairs = pairs[0]
    profile["pairs"] = time() - t
    # find pairs of peaks in the decharged spectrum whose difference matches a target in the pair target masses.
    
    t = time()
    lower_boundaries = [None for _ in range(max_k)]
    for i in range(max_k):
        k = i + 1
        lb_targets, lb_index = params.boundary_targets(k)
        lower_boundaries[i] = find_boundaries(
            peaks = decharged_peaks,
            tolerance = params.query_tolerance,
            targets = lb_targets,
            targets_index = lb_index,
        )
    profile["lower_boundaries"] = time() - t
    # find single peaks in the decharged spectrum whose mass matches a target in the boundary target masses.
    
    t = time()
    pivots = find_pivots(
        peaks = decharged_peaks,
        pairs = single_residue_pairs,
        query_tolerance = params.query_tolerance,
        symmetry_tolerance = params.symmetry_tolerance,
        score_factor = params.pivot_score_factor,
    )
    profile["pivots"] = time() - t
    # using pairs as seeds, find pivots as midpoints of peak quadruplets with mirror symmetry.

    profile["reflected_peaks"] = 0.
    profile["upper_boundaries"] = 0.
    p = len(pivots)
    reflected_peaks = [None for _ in range(p)]
    upper_boundaries = [[None for __ in range(max_k)] for _ in range(p)]
    for i in range(p):
        t = time()
        reflected_peaks[i] = AugmentedPeaks.from_peaks(
            peaks,
            charges=params.charges,
            pivot_point=pivots.cluster_points[i],
        )
        profile["reflected_peaks"] += time() - t
        # create a reflected peak list for each pivot cluster.

        for j in range(max_k):
            k = j + 1
            t = time()
            ub_targets, ub_index = params.reflected_boundary_targets(k)
            upper_boundaries[i][j] = find_boundaries(
                peaks = reflected_peaks[i],
                tolerance = params.query_tolerance,
                targets = ub_targets,
                targets_index = ub_index,
            )
            profile["upper_boundaries"] += time() - t
            # find single peaks in the reflected peak lists whose mass matches a target in the reflected boundary target masses.

    t = time()
    fragment_labels = FragmentLabels.from_results(
        peaks,
        [single_residue_pairs,],
        lower_boundaries,
        upper_boundaries,
        pivots,
    )
    profile["fragment_labels"] = time() - t
    # collect annotated fragments into a list of unique masses with aggregate properties.

    t = time()
    symmetric_nodes, lower_topology, upper_topology, pivot_topology = construct_spectrum_topology(
        fragment_labels,
        pivots,
        params.query_tolerance,
    )
    profile["spectrum_topology"] = time() - t
    # stitch together pairs, add a source node connected to boundary nodes, and a sink node to which pivot nodes connect. store pivot-crossing edges in a separate graph. augment pivot symmetries with equivalence between source and sink nodes.

    if verbose:
        print(json.dumps(profile, indent=4))
    return AnnotationResult.from_data(
        peaks,
        pairs,
        pivots,
        lower_boundaries,
        upper_boundaries,
        fragment_labels,
        lower_topology,
        upper_topology,
        pivot_topology,
        symmetric_nodes,
        profile,
    )
