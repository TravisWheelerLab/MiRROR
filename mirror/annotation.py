import dataclasses, json
from time import time
from typing import Self, Any
# standard

from .util import normalize_dict
from .io import serialize_dataclass, deserialize_dataclass, SerializableDataclass
from .spectra.types import Peaks, AugmentedPeaks
from .fragments.types import FragmentStateSpace, ResidueStateSpace, TargetMasses, MultiResidueTargetMasses, PairResult, AxesResult, BoundaryResult, UniqueFragmentIndex
from .fragments.masses import construct_pair_target_masses, construct_boundary_target_masses
from .fragments.search import find_pairs, find_boundaries, find_axes_of_reflection, deduplicate_by_fragment_mass
from .sequences.suffix_array import SuffixArray
from .sequences.queries import all_kmers
from .graphs.types import SpectrumGraph, PivotGraph, SymmetricGraph
from .evaluation.spectrum_topology import construct_spectrum_graphs
# local

import numpy as np
from omegaconf.dictconfig import DictConfig

@dataclasses.dataclass(slots=True)
class AnnotationResult(SerializableDataclass):
    peaks: Peaks
    pairs: PairResult
    axes: AxesResult
    lower_boundaries: BoundaryResult
    upper_boundaries: list[BoundaryResult]
    unique_fragment_index: UniqueFragmentIndex
    lower_topology: list[SpectrumGraph]
    upper_topology: list[SpectrumGraph]
    pivot_topology: list[PivotGraph]
    symmetric_topology: list[SymmetricGraph]
    # every list has len(self.axes) items.
    
    _profile: dict[str,float] = None

    def __len__(self) -> int:
        return len(self.axes)

    @classmethod
    def from_data(cls,
        peaks: Peaks,
        pairs: PairResult,
        axes: AxesResult,
        lower_boundaries: BoundaryResult,
        upper_boundaries: list[BoundaryResult],
        unique_fragment_index: UniqueFragmentIndex,
        lower_topology: list[SpectrumGraph],
        upper_topology: list[SpectrumGraph],
        pivot_topology: list[PivotGraph],
        symmetric_topology: list[SymmetricGraph],
        profile: dict[str,float],
    ) -> Self:
        assert len(axes) == len(upper_boundaries)
        return cls(
            peaks.to_peaks(),
            pairs,
            axes,
            lower_boundaries,
            upper_boundaries,
            unique_fragment_index,
            lower_topology,
            upper_topology,
            pivot_topology,
            symmetric_topology,
            profile,
        )

@dataclasses.dataclass(slots=True)
class AnnotationParams(SerializableDataclass):
    charges: np.ndarray
    # list of (positive) charges expected on fragments.
    query_tolerance: float
    # the radius around a query in which hits are collected.
    symmetry_tolerance: float
    # the max distance between a value and another reflected value such that they are considered symmetric. NOTE - deprecated.
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
    anno_params: AnnotationParams,
    pair_targets: list[TargetMasses],
    boundary_targets: list[TargetMasses],
    reverse_boundary_targets: list[TargetMasses],
    verbose: bool = False,
) -> AnnotationResult:
    profile = {}
    
    t = time()
    decharged_peaks = AugmentedPeaks.from_peaks(
        peaks,
        charges=anno_params.charges,
    )
    profile["decharged_peaks"] = time() - t
    # construct augmented mz and target masses

    tolerance = anno_params.query_tolerance

    t = time()
    pair_results = find_pairs(
        decharged_peaks,
        pair_targets[0],
        tolerance,
    )
    profile["pairs"] = time() - t
    # find pairs of peaks in the decharged spectrum whose difference matches a target in the pair target masses.
    
    t = time()
    lower_boundary_results = find_boundaries(
        decharged_peaks,
        boundary_targets[0],
        tolerance,
    )
    profile["lower_boundaries"] = time() - t
    # find single peaks in the decharged spectrum whose mass matches a target in the boundary target masses.
    
    t = time()
    axes = find_axes_of_reflection(
        decharged_peaks,
        pair_results,
        tolerance,
        anno_params.pivot_score_factor,
    )
    profile["axes_of_reflection"] = time() - t
    # using pairs as seeds, find axes as midpoints of peak quadruplets with mirror symmetry.

    profile["reflected_peaks"] = 0.
    profile["upper_boundaries"] = 0.
    p = len(axes)
    reflected_peaks = [None for _ in range(p)]
    upper_boundaries = [None for _ in range(p)]
    for i in range(p):
        t = time()
        reflected_peaks[i] = AugmentedPeaks.from_peaks(
            peaks,
            charges=anno_params.charges,
            pivot_point=axes.cluster_points[i],
        )
        profile["reflected_peaks"] += time() - t
        # create a reflected peak list for each pivot cluster.
        t = time()
        upper_boundaries[i] = find_boundaries(
            reflected_peaks[i],
            reverse_boundary_targets[0],
            tolerance,
        )
        profile["upper_boundaries"] += time() - t
        # find single peaks in the reflected peak lists whose mass matches a target in the reflected boundary target masses.

    t = time()
    unique_fragment_index = deduplicate_by_fragment_mass(
        peaks,
        pair_results,
        lower_boundary_results,
        axes,
        upper_boundaries,
    )
    profile["deduplicate_by_fragment_mass"] = time() - t
    # create a compact, unified index into the array of unique fragment masses.

    t = time()
    spectrum_topology = construct_spectrum_graphs(
        unique_fragment_index,
        axes,
        tolerance,
    )
    profile["spectrum_topology"] = time() - t
    # for each axis, construct four graphs: lower and upper spectrum graphs from pairs connecting boundaries to axis nodes, a pivot graph representing edges connecting the lower and upper graphs, and a symmetry graph pairing nodes whose fragment masses are symmetric.

    if verbose:
        print(json.dumps(profile, indent=4))
    return AnnotationResult.from_data(
        peaks,
        pair_results,
        axes,
        lower_boundary_results,
        upper_boundaries,
        unique_fragment_index,
        lower_topology = spectrum_topology[0],
        upper_topology = spectrum_topology[1],
        pivot_topology = spectrum_topology[2],
        symmetric_topology = spectrum_topology[3],
        profile = profile,
    )
