import dataclasses, json
from time import time
from typing import Self, Any
# standard

from .util import normalize_dict
from .io import serialize_dataclass, deserialize_dataclass, SerializableDataclass
from .spectra.types import Peaks, AugmentedPeaks
from .fragments import FragmentStateSpace, ResidueStateSpace, MultiResidueStateSpace, TargetMassStateSpace, PairResult, PivotResult, BoundaryResult, find_pairs, find_pivots, find_boundaries 
from .sequences.suffix_array import SuffixArray
from .sequences.queries import PeptideMassQueryEngine, all_kmers
# local

import numpy as np

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
    residue_space: ResidueStateSpace
    # residue masses augmented by modifications.
    dimer_space: ResidueStateSpace
    # double-residue masses augmented by modifications.
    trimer_space: ResidueStateSpace
    # triple-residue masses augmented by modifications.
    fragment_space: FragmentStateSpace
    # loss states.
    boundary_fragment_space: FragmentStateSpace
    # loss states shifted by the series (a,b,c,x,y,z) offsets.
    query_tolerance: float
    # the radius around a query in which hits are collected.
    symmetry_tolerance: float
    # the max distance between one value and another reflected value such that they are considered symmetric.
    pivot_score_factor: float
    # tunes the pivot symmetry score threshold := max score * score factor

    @classmethod
    def from_config(cls,
        cfg: dict[str,Any],
        suffix_array: SuffixArray = None,
    ) -> Self:
        res = cfg['residues']
        res_mass = np.array(res['masses'])
        res_sym = np.array(res['symbols'])
        mod = cfg['modifications']
        mod_mass = np.array(mod['masses'])
        mod_sym = np.array(mod['symbols'])
        mod_appl = [np.array(x) for x in mod['application']]
        mod_max_num = mod['max_num']
        mod_nulls = np.array(mod['nulls'])
        residue_space = ResidueStateSpace(
            res_mass,
            res_sym,
            mod_mass,
            mod_sym,
            mod_nulls,
            mod_appl,
            mod_max_num,
        )
        # residue space

        if not(suffix_array is None):
            engine = PeptideMassQueryEngine(
                residue_space,
                suffix_array,
                5,
            )
            dimer_masses, dimers = engine.query_kmers(2)
            trimer_masses, trimers = engine.query_kmers(3)
        else:
            dimer_masses, dimers, _ = all_kmers(residue_space, 2)
            trimer_masses, trimers, _ = all_kmers(residue_space, 3)
        dimer_space = MultiResidueStateSpace.from_nonunique_pairs(
            dimer_masses,
            dimers,
            #mod_mass,
            #mod_sym,
            #mod_appl,
            #mod_num * 2,
        )
        trimer_space = MultiResidueStateSpace.from_nonunique_pairs(
            trimer_masses,
            trimers,
            #=od_mass,
            #=od_sym,
            #=od_appl,
            #=od_num * 3,
        )
        # dimer and trimer spaces. restricted by a suffix array, if passed.
        
        loss = cfg['losses']
        loss_mass = np.array(loss['masses'])
        n_loss = len(loss_mass)
        loss_sym = np.array(loss['symbols'])
        loss_nulls = np.array(loss['nulls'])
        loss_appl = [np.array(x) for x in loss['application']]
        charges = np.array(cfg['charges'])
        fragment_space = FragmentStateSpace(
            loss_mass,
            loss_sym,
            loss_nulls,
            loss_appl,
            charges,
        )
        # fragment space for pair difference masses
        
        ser = cfg['series']
        ser_mass = np.array(ser['masses'])
        n_ser = len(ser_mass)
        ser_sym = np.array(ser['symbols'])
        boundary_mass = (ser_mass.reshape(n_ser,1) + loss_mass.reshape(1, n_loss)).flatten()
        boundary_sym = (ser_sym.reshape(n_ser,1) + ' ' + loss_sym.reshape(1, n_loss)).flatten()
        boundary_nulls = [x + (i * n_loss) for x in loss_nulls for i in range(n_ser)]
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO
        boundary_appl = sum([loss_appl] * n_ser, start=[])
        boundary_fragment_space = FragmentStateSpace(
            boundary_mass,
            boundary_sym,
            boundary_nulls,
            boundary_appl,
            charges,
        )
        # fragment space for boundary masses
        
        query_tolerance = cfg['query_tolerance']
        symmetry_tolerance = cfg['symmetry_tolerance']
        pivot_score_factor = cfg['pivot_score_factor']
        # other params

        return cls(
            residue_space,
            dimer_space,
            trimer_space,
            fragment_space,
            boundary_fragment_space,
            query_tolerance,
            symmetry_tolerance,
            pivot_score_factor,
        )

def annotate(
    peaks: Peaks,
    targets: TargetMassStateSpace,
    params: AnnotationParams,
    verbose: bool = False,
) -> AnnotationResult:
    profile = {}

    if verbose:
        print(peaks)
    
    t = time()
    decharged_peaks = AugmentedPeaks.from_peaks(
        peaks,
        charges=params.fragment_space.charges,
    )
    profile["augment"] = time() - t
    if verbose:
        print(decharged_peaks)
    # construct augmented mz and target masses
    
    t = time()
    pairs = find_pairs(
        peaks = decharged_peaks,
        tolerance = params.query_tolerance,
        targets = targets,
    )
    profile["pairs"] = time() - t
    if verbose:
        print(pairs)
    # peak pairs
    
    t = time()
    pivots = find_pivots(
        peaks = peaks,
        pairs = pairs,
        query_tolerance = params.query_tolerance,
        symmetry_tolerance = params.symmetry_tolerance,
        score_factor = params.pivot_score_factor,
    )
    profile["pivots"] = time() - t
    if verbose:
        print(pivots)
    # pivots
    
    t = time()
    lower_boundaries = find_boundaries(
        peaks = decharged_peaks,
        tolerance = params.query_tolerance,
        targets = targets,
    )
    profile["lower_boundaries"] = time() - t
    if verbose:
        print(lower_boundaries)
    # low-mz boundaries
    
    t = time()
    reflected_peaks = [AugmentedPeaks.from_peaks(
            peaks,
            charges=params.fragment_space.charges,
            pivot_point=pivot_pt,
        ) for pivot_pt in pivots.cluster_points]
    ## cluster points correspond to reflections
    upper_boundaries = [find_boundaries(
            peaks = refl,
            tolerance = params.query_tolerance,
            targets = targets,
        ) for refl in reflected_peaks]
    ## each reflected spectrum has its own set of right boundaries
    profile["upper_boundaries"] = time() - t
    if verbose:
        print(upper_boundaries)
    # high-mz boundaries
    
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
