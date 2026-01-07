import dataclasses, json
from time import time
from typing import Self, Any
# standard

from .util import normalize_dict
from .io import serialize_dataclass, deserialize_dataclass, SerializableDataclass
from .spectra.types import Peaks, AugmentedPeaks
from .fragments.types import FragmentStateSpace, ResidueStateSpace, MultiResidueStateSpace, TargetMasses, PairResult, PivotResult, BoundaryResult, FragmentMasses
from .fragments.masses import construct_pair_target_masses, construct_boundary_target_masses, construct_unique_fragment_masses
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
    fragment_masses: FragmentMasses
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
        fragment_masses: FragmentMasses,
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
            fragment_masses = fragment_masses,
            tolerance = tolerance,
            _profile = profile,
        )

@dataclasses.dataclass(slots=True)
class AnnotationParams(SerializableDataclass):
    target_masses: tuple[TargetMasses,TargetMasses,TargetMasses]
    max_k: int
    # pair_target_masses: TargetMasses
    # # target masses for pairs.
    # # residues augmented by modifications and left and right losses.
    # boundary_target_masses: TargetMasses
    # # target masses for lower boundaries.
    # # residues augmented by modifications and losses, and shifted by b and y series offsets.
    # reflected_boundary_target_masses: TargetMasses
    # # target masses for reflected upper boundaries.
    # # residues augmented by reflected modifications and losses, 
    # # and shifted by b and y series offsets.
    # # TODO - multiresidue targets.
    charges: np.ndarray
    # list of (positive) charges expected on fragments. typically [1,], [1,2,], or [1,2,3].
    query_tolerance: float
    # the radius around a query in which hits are collected.
    symmetry_tolerance: float
    # the max distance between a value and another reflected value such that they are considered symmetric.
    pivot_score_factor: float
    # tunes the pivot symmetry score threshold := max score * score factor. TODO - this might not work for noisier spectra.

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

        # TODO - multiresidue spaces and targets.

        return cls(
            target_masses = (
                pair_targets,
                lower_boundary_targets,
                reflected_upper_boundary_targets,
                # k=2 pair, lb, ub...
                # k=3 pair, lb, ub...
            ),
            max_k = 1,
            charges = np.array(cfg.charges),
            query_tolerance = cfg.query_tolerance,
            symmetry_tolerance = cfg.symmetry_tolerance,
            pivot_score_factor = cfg.pivot_score_factor,
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
    pair_targets, pair_index = params.pair_targets()
    pairs = find_pairs(
        peaks = decharged_peaks,
        tolerance = params.query_tolerance,
        targets = pair_targets,
        targets_index = pair_index,
    )
    profile["pairs"] = time() - t
    if verbose:
        print(pairs)
    # find pairs of peaks in the decharged spectrum whose difference matches a target in the pair target masses.
    
    t = time()
    lb_targets, lb_index = params.boundary_targets()
    lower_boundaries = find_boundaries(
        peaks = decharged_peaks,
        tolerance = params.query_tolerance,
        targets = lb_targets,
        targets_index = lb_index,
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
    ub_targets, ub_index = params.reflected_boundary_targets()
    upper_boundaries = [find_boundaries(
            peaks = refl_peaks,
            tolerance = params.query_tolerance,
            targets = ub_targets,
            targets_index = ub_index,
        ) for refl_peaks in reflected_peaks]
    profile["upper_boundaries"] = time() - t
    if verbose:
        print(upper_boundaries)
    # find single peaks in the reflected peak lists whose mass matches a target in the reflected boundary target masses.

    t = time()
    fragment_masses = construct_unique_fragment_masses(
        peaks,
        pairs,
        pivots,
        lower_boundaries,
        upper_boundaries,
    )
    profile["reindexing"] = time() - t
    # collect annotated fragments into a list of unique masses with aggregate properties.
    
    if verbose:
        print(json.dumps(profile, indent=4))
    return AnnotationResult.from_data(
        peaks,
        pairs,
        pivots,
        lower_boundaries,
        upper_boundaries,
        fragment_masses,
        params.query_tolerance,
        profile,
    )
