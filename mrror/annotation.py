import dataclasses
from time import time
from typing import Self, Any
# standard

from .util import normalize_dict
from .spectra.types import Peaks, AugmentedPeaks
from .fragments.types import FragmentStateSpace, ResidueStateSpace, TargetMassStateSpace
from .fragments.pairs import PairResult, find_pairs
from .fragments.pivots import PivotResult, find_pivots
from .fragments.boundaries import BoundaryResult, find_boundaries 
# local

import numpy as np

@dataclasses.dataclass(slots=True)
class AnnotationResult:
    peaks: Peaks
    decharged_peaks: AugmentedPeaks
    reflected_peaks: list[AugmentedPeaks]
    pairs: PairResult
    pivots: PivotResult
    left_boundaries: BoundaryResult
    right_boundaries: list[BoundaryResult]
    
    _profile: dict[str,float] = None

    @classmethod
    def from_data(cls,
        peaks: Peaks,
        decharged_peaks: AugmentedPeaks,
        reflected_peaks: list[AugmentedPeaks],
        pairs: PairResult,
        pivots: PivotResult,
        left_boundaries: BoundaryResult,
        right_boundaries: list[BoundaryResult],
        profile: dict[str,float],
    ) -> Self:
        assert len(pivots) == len(right_boundaries) == len(reflected_peaks)
        return cls(
            peaks = peaks,
            decharged_peaks = decharged_peaks,
            reflected_peaks = reflected_peaks,
            pairs = pairs,
            pivots = pivots,
            left_boundaries = left_boundaries,
            right_boundaries = right_boundaries,
            _profile = profile,
        )

    @classmethod
    def read(cls,
        filepath: str,
    ) -> Self:
        with open(filepath, 'r') as f:
            anno = json.load(f)
            return cls(
                peaks = Peaks.from_data(**anno['peaks']),
                decharged_peaks = AugmentedPeaks.from_data(**anno['decharged_peaks']),
                reflected_peaks = [AugmentedPeaks.from_data(**x)
                    for x in anno['reflected_peaks']],
                pairs = PairResult.from_dict(anno['pairs']),
                pivots = PivotResult.from_dict(anno['pivots']),
                left_boundaries = BoundaryResult.from_dict(anno['left_boundaries']),
                right_boundaries = [BoundaryResult.from_dict(x)
                    for x in anno['right_boundaries']],
                _profile = anno['_profile'],
            )

    def write(self,
        filepath: str,
    ) -> None:
        with open(filepath, 'w') as f:
            anno = dataclasses.asdict(self)
            json.dump(anno, f, indent = 4)

@dataclasses.dataclass(slots=True)
class AnnotationParams:
    residue_space: ResidueStateSpace
    # residue masses augmented by modifications.
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
    ) -> Self:
        res = cfg['residues']
        res_mass = np.array(res['masses'])
        res_sym = np.array(res['symbols'])
        mod = cfg['modifications']
        mod_mass = np.array(mod['masses'])
        mod_sym = np.array(mod['symbols'])
        mod_appl = [np.array(x) for x in mod['application']]
        mod_num = mod['max_num']
        residue_space = ResidueStateSpace(
            res_mass,
            res_sym,
            mod_mass,
            mod_sym,
            mod_appl,
            mod_num,
        )
        # residue space
        
        loss = cfg['losses']
        loss_mass = np.array(loss['masses'])
        n_loss = len(loss_mass)
        loss_sym = np.array(loss['symbols'])
        loss_appl = [np.array(x) for x in loss['application']]
        charges = np.array(cfg['charges'])
        fragment_space = FragmentStateSpace(
            loss_mass,
            loss_sym,
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
        boundary_appl = sum([loss_appl] * n_ser, start=[])
        boundary_fragment_space = FragmentStateSpace(
            boundary_mass,
            boundary_sym,
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
    
    t = time()
    decharged_peaks = AugmentedPeaks.from_peaks(
        peaks,
        charges=params.fragment_space.charges,
    )
    profile["augment"] = time() - t
    # construct augmented mz and target masses
    
    t = time()
    pairs = find_pairs(
        peaks = decharged_peaks.mz,
        tolerance = params.query_tolerance,
        target_masses = targets.pair_masses,
    )
    profile["pairs"] = time() - t
    if verbose:
        print(pairs)
    # peak pairs
    
    t = time()
    pivots = find_pivots(
        peaks = peaks.mz,
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
    left_boundaries = find_boundaries(
        peaks = decharged_peaks.mz,
        tolerance = params.query_tolerance,
        target_masses = targets.boundary_masses,
    )
    profile["left_boundaries"] = time() - t
    if verbose:
        print(left_boundaries)
    # low-mz boundaries
    
    t = time()
    reflected_peaks = [AugmentedPeaks.from_peaks(
            peaks,
            charges=params.fragment_space.charges,
            pivot_point=pivot_pt,
        ) for pivot_pt in pivots.cluster_points]
    ## cluster points correspond to reflections
    right_boundaries = [find_boundaries(
            peaks = refl.mz,
            tolerance = params.query_tolerance,
            target_masses = targets.boundary_masses,
        ) for refl in reflected_peaks]
    ## each reflected spectrum has its own set of right boundaries
    profile["right_boundaries"] = time() - t
    if verbose:
        print(right_boundaries)
    # high-mz boundaries
    
    if verbose:
        print(profile, normalize_dict(profile))
    return AnnotationResult.from_data(
        peaks,
        decharged_peaks,
        reflected_peaks,
        pairs,
        pivots,
        left_boundaries,
        right_boundaries,
        profile,
    )
