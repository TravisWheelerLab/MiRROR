import dataclasses, json
from time import time
from typing import Self, Any
# standard

from .util import normalize_dict, consecutive_intervals
from .io import serialize_dataclass, deserialize_dataclass, SerializableDataclass
from .spectra.types import Peaks, AugmentedPeaks, SimulationLabeledPeaks
from .fragments.types import FragmentStateSpace, ResidueStateSpace, LossDistribution, TargetMasses, MultiResidueTargetMasses, PairResult, AxesResult, BoundaryResult, UniqueFragmentIndex, AnnotationIndex
from .fragments.masses import construct_pair_target_masses, construct_boundary_target_masses
from .fragments.search import find_pairs, find_boundaries, find_axes_of_reflection, deduplicate_by_fragment_mass, expand_annotations
from .sequences.suffix_array import SuffixArray
from .sequences.queries import all_kmers
from .graphs.types import SpectrumGraph, PivotGraph, SymmetricGraph
from .evaluation.spectrum_topology import SpectrumTopology, construct_spectrum_topology
from .evaluation.costmodels import SymmetricNodeCostModel, MassConstrainedPathCostModel
from .evaluation.peptide_mass_lookup import construct_peptide_mass_lookup, PeptideMassLookup
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
    annotation_index: AnnotationIndex
    spectrum_topology: SpectrumTopology
    node_lookup: PeptideMassLookup
    # every list has len(self.axes) items.
    
    _profile: dict[str,float] = None

    def __len__(self) -> int:
        return len(self.axes)

    @classmethod
    def from_data(
        cls,
        peaks: Peaks,
        pairs: PairResult,
        axes: AxesResult,
        lower_boundaries: BoundaryResult,
        upper_boundaries: list[BoundaryResult],
        unique_fragment_index: UniqueFragmentIndex,
        annotation_index: AnnotationIndex,
        spectrum_topology: SpectrumTopology,
        peptide_mass_lookup: PeptideMassLookup,
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
            annotation_index,
            spectrum_topology,
            peptide_mass_lookup,
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
    loss_distribution: LossDistribution,
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
        loss_distribution,
    )
    profile["deduplicate_by_fragment_mass"] = time() - t
    # create a compact, unified index into the array of unique fragment masses.

    t = time()
    annotation_index = expand_annotations(
        pair_results,
        pair_targets[0],
        lower_boundary_results,
        boundary_targets[0],
        upper_boundaries,
        reverse_boundary_targets[0],
        unique_fragment_index,
    )
    profile["annotation_index"] = time() - t

    t = time()
    spectrum_topology = construct_spectrum_topology(
        unique_fragment_index,
        annotation_index,
        axes,
        tolerance,
    )
    profile["spectrum_topology"] = time() - t
    # for each axis, construct four graphs: lower and upper spectrum graphs from pairs connecting boundaries to axis nodes, a pivot graph representing edges connecting the lower and upper graphs, and a symmetry graph pairing nodes whose fragment masses are symmetric.

    t = time()
    peptide_mass_lookup = construct_peptide_mass_lookup(
        decharged_peaks,
        loss_distribution,
        unique_fragment_index,
        annotation_index,
        tolerance,
    )

    if verbose:
        print(json.dumps(profile, indent=4))
    return AnnotationResult.from_data(
        peaks,
        pair_results,
        axes,
        lower_boundary_results,
        upper_boundaries,
        unique_fragment_index,
        annotation_index,
        spectrum_topology,
        peptide_mass_lookup,
        profile = profile,
    )

def derive_annotation_from_simulation(
    peaks: SimulationLabeledPeaks,
    anno_params: AnnotationParams,
    pair_targets: list[TargetMasses],
    boundary_targets: list[TargetMasses],
    reverse_boundary_targets: list[TargetMasses],
    loss_distribution: LossDistribution,
) -> AnnotationResult:
    tolerance = 0.01
    charges = np.array([x[0] for x in peaks.charge])
    mods = np.array([x[0] for x in peaks.mods])
    print("mods",mods)
    losses = np.array([x[0] for x in peaks.loss])
    print("losses",losses)
    if any(charges > 1):
        raise ValueError("Derived annotations are not available for simulations with multiple charge states.")
    pair_idx = np.array(peaks.pairs())
    pair_pos = peaks.position[pair_idx]
    pair_amino = [peaks.peptide[i:j] for (i,j) in pair_pos]
    amino_id_lookup = {a: i for (i,a) in enumerate(pair_targets[0].residue_space.amino_symbols)}
    mod_id_lookup = {a: i for (i,a) in enumerate(pair_targets[0].residue_space.modification_symbols)}
    loss_id_lookup = {a: i for (i,a) in enumerate(pair_targets[0].right_fragment_space.loss_symbols)}
    pair_anno_amino_id = np.array([amino_id_lookup[a] for a in pair_amino])
    pair_mod = [str(mods[i:j][0]) for (i,j) in pair_pos]
    pair_anno_mod_id = np.array([mod_id_lookup[a] for a in pair_mod])
    pair_left_loss = losses[pair_idx[:,0]]
    pair_anno_left_loss_id = np.array([loss_id_lookup[a] for a in pair_left_loss])
    pair_right_loss = losses[pair_idx[:,1]]
    pair_anno_right_loss_id = np.array([loss_id_lookup[a] for a in pair_right_loss])
    n_pairs = len(pair_idx)
    pair_results = PairResult.from_arrays(
        peak_index_pairs = pair_idx,
        augmented_peak_index_pairs = pair_idx,
        augmented_peak_charges = charges[pair_idx],
        query_masses = np.array(peaks.decharged_pair_masses()),
        hit_ranges = consecutive_intervals(n_pairs),
    )
    lbound_idx = np.array(peaks.lower_boundaries())
    lbound_pos = peaks.position[lbound_idx]
    lbound_amino = [peaks.peptide[i-1:i] for i in lbound_pos]
    lbound_anno_amino_id = np.array([amino_id_lookup[a] for a in lbound_amino])
    lbound_mod = [str(mods[i-1:i][0]) for i in lbound_pos]
    lbound_anno_mod_id = np.array([mod_id_lookup[a] for a in lbound_mod])
    lbound_loss = losses[lbound_idx]
    lbound_anno_left_loss_id = np.zeros(len(lbound_loss),dtype=int)
    lbound_anno_right_loss_id = np.array([loss_id_lookup[a] for a in lbound_loss])
    n_lbound = len(lbound_idx)
    lower_boundary_results = BoundaryResult.from_arrays(
        peak_indices = lbound_idx,
        augmented_peak_indices = lbound_idx,
        augmented_peak_charges = charges[lbound_idx],
        query_masses = np.array(peaks.decharged_lower_boundary_masses()),
        hit_ranges = n_pairs + consecutive_intervals(n_lbound),
    )
    ubound_idx = np.array(peaks.upper_boundaries())
    ubound_pos = peaks.position[ubound_idx]
    ubound_amino = [peaks.peptide[i:i+1] for i in ubound_pos]
    ubound_anno_amino_id = np.array([amino_id_lookup[a] for a in ubound_amino])
    ubound_mod = [str(mods[i:i+1][0]) for i in ubound_pos]
    ubound_anno_mod_id = np.array([mod_id_lookup[a] for a in ubound_mod])
    ubound_loss = losses[ubound_idx]
    ubound_anno_left_loss_id = np.zeros(len(ubound_loss),dtype=int)
    ubound_anno_right_loss_id = np.array([loss_id_lookup[a] for a in ubound_loss])
    n_ubound = len(ubound_idx)
    upper_boundaries = [
        BoundaryResult.from_arrays(
            peak_indices = ubound_idx,
            augmented_peak_indices = ubound_idx,
            augmented_peak_charges = charges[ubound_idx],
            query_masses = np.array(peaks.decharged_upper_boundary_masses()),
            hit_ranges = n_pairs + n_lbound + consecutive_intervals(n_ubound),
        ),
    ]
    amino_mass = pair_targets[0].residue_space.amino_masses
    modification_mass = pair_targets[0].residue_space.modification_masses
    loss_mass = pair_targets[0].right_fragment_space.loss_masses
    anno_amino_id = np.concat([pair_anno_amino_id,lbound_anno_amino_id,ubound_anno_amino_id])
    anno_amino_mass = amino_mass[anno_amino_id]
    anno_mod_id = np.concat([pair_anno_mod_id,lbound_anno_mod_id,ubound_anno_mod_id])
    anno_mod_mass = modification_mass[anno_mod_id]
    anno_left_loss_id = np.concat([pair_anno_left_loss_id,lbound_anno_left_loss_id,ubound_anno_left_loss_id])
    anno_left_loss_mass = loss_mass[anno_left_loss_id]
    anno_right_loss_id = np.concat([pair_anno_right_loss_id,lbound_anno_right_loss_id,ubound_anno_right_loss_id])
    anno_right_loss_mass = loss_mass[anno_right_loss_id]
    combined_states = np.stack([anno_amino_id,anno_left_loss_id,anno_right_loss_id,anno_mod_id],axis=1)
    synthetic_targets = TargetMasses(
        target_masses = anno_amino_mass + anno_mod_mass + anno_left_loss_mass - anno_right_loss_mass,
        target_states = combined_states,
        null_states = pair_targets[0].null_states,
        residue_space = pair_targets[0].residue_space,
        left_fragment_space = pair_targets[0].left_fragment_space,
        right_fragment_space = pair_targets[0].right_fragment_space,
    )
    axis = peaks.pivot
    m = len(peaks)
    k = len(peaks.peptide)
    symmetric_idx = np.array([
        (i,j)
        for i in range(m)
        for j in range(i + 1,m)
        if peaks.position[i] == (k - peaks.position[j]) and peaks.series[i] != peaks.series[j]
    ])
    print(symmetric_idx)
    axes = AxesResult.from_data(
        cluster_points = np.array([axis]),
        clusters = [np.arange(k),],
        scores = np.array([1.]),
        symmetries = [symmetric_idx,],
        symmetries_charges = [charges[symmetric_idx],],
        symmetries_peak_idx = [symmetric_idx,],
        axes_points = np.empty((0,),dtype=float),
        axes_indices = np.empty((0,4),dtype=int),
        axes_charges = np.empty((0,4),dtype=int),
        axes_peak_idx = np.empty((0,4),dtype=int),
    )
    unique_fragment_index = deduplicate_by_fragment_mass(
        peaks,
        pair_results,
        lower_boundary_results,
        axes,
        upper_boundaries,
        loss_distribution,
    )
    # create a compact, unified index into the array of unique fragment masses.

    annotation_index = expand_annotations(
        pair_results,
        synthetic_targets,
        lower_boundary_results,
        synthetic_targets,
        upper_boundaries,
        synthetic_targets,
        unique_fragment_index,
    )

    spectrum_topology = construct_spectrum_topology(
        unique_fragment_index,
        annotation_index,
        axes,
        tolerance,
    )
    peptide_mass_lookup = construct_peptide_mass_lookup(
        peaks,
        loss_distribution,
        unique_fragment_index,
        annotation_index,
        tolerance,
    )
    return AnnotationResult.from_data(
        peaks,
        pair_results,
        axes,
        lower_boundary_results,
        upper_boundaries,
        unique_fragment_index,
        annotation_index,
        spectrum_topology,
        peptide_mass_lookup,
        profile = None,
    )
