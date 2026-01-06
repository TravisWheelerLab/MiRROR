import dataclasses, json
from time import time
from typing import Self, Any
# standard

from .util import normalize_dict
from .io import serialize_dataclass, deserialize_dataclass, SerializableDataclass
from .spectra.types import Peaks, AugmentedPeaks
from .fragments.types import FragmentStateSpace, ResidueStateSpace, MultiResidueStateSpace, TargetMasses, PairResult, PivotResult, BoundaryResult
from .fragments.masses import construct_pair_target_masses, construct_boundary_target_masses
from .fragments.search import find_pairs, find_pivots, find_boundaries 
from .sequences.suffix_array import SuffixArray
from .sequences.queries import PeptideMassQueryEngine, all_kmers
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
    tolerance: float
    
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
        tolerance: float,
        profile: dict[str,float],
    ) -> Self:
        assert len(pivots) == len(upper_boundaries)
        return cls(
            peaks = peaks.to_peaks(),
            pairs = pairs,
            pivots = pivots,
            lower_boundaries = lower_boundaries,
            upper_boundaries = upper_boundaries,
            tolerance = tolerance,
            _profile = profile,
        )

@dataclasses.dataclass(slots=True)
class AnnotationParams(SerializableDataclass):
    pair_target_masses: TargetMasses
    # target masses for pairs.
    # residues augmented by modifications and left and right losses.
    boundary_target_masses: TargetMasses
    # target masses for lower boundaries.
    # residues augmented by modifications and losses, and shifted by b and y series offsets.
    reflected_boundary_target_masses: TargetMasses
    # target masses for reflected upper boundaries.
    # residues augmented by reflected modifications and losses, 
    # and shifted by b and y series offsets.
    # TODO - multiresidue targets.
    charges: np.ndarray
    # list of (positive) charges expected on fragments. typically [1,], [1,2,], or [1,2,3].
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
        
        # TODO - multiresidue spaces and targets.

        charges = np.array(cfg.charges)
        query_tolerance = cfg['query_tolerance']
        symmetry_tolerance = cfg['symmetry_tolerance']
        pivot_score_factor = cfg['pivot_score_factor']
        # other params

        return cls(
            pair_targets,
            lower_boundary_targets,
            reflected_upper_boundary_targets,
            charges,
            query_tolerance,
            symmetry_tolerance,
            pivot_score_factor,
        )

def annotate(
    peaks: Peaks,
    params: AnnotationParams,
    verbose: bool = False,
) -> AnnotationResult:
    profile = {}

    if verbose:
        print(peaks)
    
    t = time()
    decharged_peaks = AugmentedPeaks.from_peaks(
        peaks,
        charges=params.charges,
    )
    profile["augment"] = time() - t
    if verbose:
        print(decharged_peaks)
    # construct augmented mz and target masses
    
    t = time()
    pairs = find_pairs(
        peaks = decharged_peaks,
        tolerance = params.query_tolerance,
        targets = params.pair_target_masses,
    )
    profile["pairs"] = time() - t
    if verbose:
        print(pairs)
    # find pairs of peaks in the decharged spectrum whose difference matches a target in the pair target masses.
    
    t = time()
    lower_boundaries = find_boundaries(
        peaks = decharged_peaks,
        tolerance = params.query_tolerance,
        targets = params.boundary_target_masses,
    )
    profile["lower_boundaries"] = time() - t
    if verbose:
        print(lower_boundaries)
    # find single peaks in the decharged spectrum whose mass matches a target in the boundary target masses.
    
    t = time()
    pivots = find_pivots(
        peaks = decharged_peaks,
        pairs = pairs,
        query_tolerance = params.query_tolerance,
        symmetry_tolerance = params.symmetry_tolerance,
        score_factor = params.pivot_score_factor,
    )
    profile["pivots"] = time() - t
    if verbose:
        print(pivots)
    # using pairs as seeds, find pivots as midpoints of peak quadruplets with mirror symmetry.
    
    t = time()
    reflected_peaks = [AugmentedPeaks.from_peaks(
            peaks,
            charges=params.charges,
            pivot_point=pivot_pt,
        ) for pivot_pt in pivots.cluster_points]
    profile["reflection"] = time() - t
    # create a reflected peak list for each pivot cluster.

    t = time()
    upper_boundaries = [find_boundaries(
            peaks = refl_peaks,
            tolerance = params.query_tolerance,
            targets = params.reflected_boundary_target_masses,
        ) for refl_peaks in reflected_peaks]
    profile["upper_boundaries"] = time() - t
    if verbose:
        print(upper_boundaries)
    # find single peaks in the reflected peak lists whose mass matches a target in the reflected boundary target masses.
    
    if verbose:
        print(json.dumps(profile, indent=4))
    return AnnotationResult.from_data(
        peaks,
        pairs,
        pivots,
        lower_boundaries,
        upper_boundaries,
        params.query_tolerance,
        profile,
    )
