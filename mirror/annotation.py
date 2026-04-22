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
    charges: np.ndarray
    # list of (positive) charges expected on fragments.
    query_tolerance: float
    # the radius around a query in which hits are collected.
    symmetry_tolerance: float
    # the max distance between a value and another reflected value such that they are considered symmetric.
    pivot_score_factor: float
    # tunes the pivot symmetry score threshold := max score * score factor. TODO - this might not work for noisier spectra.

    @classmethod
    def from_config(cls,
        cfg: DictConfig,
        suffix_array: SuffixArray = None,
    ) -> Self:
        return cls(
            charges = np.array(cfg.charges),
            query_tolerance = cfg.query_tolerance,
            symmetry_tolerance = cfg.symmetry_tolerance,
            pivot_score_factor = cfg.pivot_score_factor,
        )

def annotate(
    peaks: Peaks,
    params: AnnotationParams,
    pair_targets: list[TargetMasses],
    boundary_targets: list[TargetMasses],
    reflected_boundary_targets: list[TargetMasses],
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
    single_residue_pairs = find_pairs(
        peaks = decharged_peaks,
        tolerance = params.query_tolerance,
        targets = pair_targets[0],
        targets_index = 0,
    )
    pairs = [single_residue_pairs,]
    profile["pairs"] = time() - t
    # find pairs of peaks in the decharged spectrum whose difference matches a target in the pair target masses.
    
    t = time()
    nb = len(boundary_targets)
    lower_boundaries = [None for _ in range(nb)]
    for i in range(nb):
        targets = boundary_targets[i]
        lower_boundaries[i] = find_boundaries(
            peaks = decharged_peaks,
            tolerance = params.query_tolerance,
            targets = targets,
            targets_index = i,
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
    nrb = len(reflected_boundary_targets)
    reflected_peaks = [None for _ in range(p)]
    upper_boundaries = [[None for __ in range(nrb)] for _ in range(p)]
    for i in range(p):
        t = time()
        reflected_peaks[i] = AugmentedPeaks.from_peaks(
            peaks,
            charges=params.charges,
            pivot_point=pivots.cluster_points[i],
        )
        profile["reflected_peaks"] += time() - t
        # create a reflected peak list for each pivot cluster.

        for j in range(nrb):
            t = time()
            targets = reflected_boundary_targets[j]
            upper_boundaries[i][j] = find_boundaries(
                peaks = reflected_peaks[i],
                tolerance = params.query_tolerance,
                targets = targets,
                targets_index = j,
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
