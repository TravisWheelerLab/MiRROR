import dataclasses, re, enum
import itertools as it
from typing import Self

import editdistance
import numpy as np
import mzspeclib as mzlib
import mzpaf
from tabulate import tabulate

from .util import merge_compare, merge_compare_fuzzy_unique, merge_compare_exact_unique, first_match
from .spectra.types import Peaks
from .spectra.simulation import oms, generate_fragment_spectrum, list_mz, list_intensity, list_ion_data, simulate_pivot, DEFAULT_PARAM, COMPLEX_PARAM
from .fragments.types import HYDROGEN_MASS, PairResult, PivotResult, BoundaryResult, FragmentMasses

from .annotation import AnnotationResult, AnnotationParams
from .alignment import AlignmentResult, AlignmentParams
from .enumeration import EnumerationResult, EnumerationParams

class AnnotationType(enum.Enum):
    FRAGMENTS = 0
    CANDIDATE = 1
    SIMULATION = 2
    BENCHMARK = 3

@dataclasses.dataclass(slots=True)
class ComparedAnnotations:
    """The result of comparing two AnnotatedPeaks objects A, B as constructed by A.compare(B).
    - comp_type: tuple, a pair of AnnotationType enums which record the manner in which A and B were constructed.
    - peptide_err: int, an edit distance between peptide sequences.
    - pivot_err: float, signed difference calculated as A.pivot - B.pivot.
    - mz_err: int, the number of un-aligned peaks in A.mz calculated as len(A) - len(alignment).
    - alignment: (n,2) int, index pairs aligning A.mz to B.mz.
    - int_err: (n,) float, difference in intensity between aligned peaks calculated as A.intensity[alignment[:,0]] - B.intensity[alignment[:,1]].
    - series_err: (n,) bool, comparison in series between aligned peaks.
    - pos_err: (n,) int, difference in position between aligned peaks.
    - charge_err: (n,2) int, optimal matches between charges of aligned peaks.
    - loss_aln: (n,2) int, optimal matches between losses of aligned peaks.
    - mods_aln: (n,2) int, optimal matches between modifications of aligned peaks."""
    comp_type: tuple[AnnotationType,AnnotationType]
    peptide_err: int
    pivot_err: float
    mz_err: int
    alignment: np.ndarray   # [[int; 2]; n]
    int_err: np.ndarray     # [float; n]
    series_err: np.ndarray  # [bool; n]
    pos_err: np.ndarray     # [int; n]
    charge_aln: np.ndarray  # [[int; 2]; n]
    loss_aln: np.ndarray    # [[int; 2]; n]
    mods_aln: np.ndarray    # [[int; 2]; n]

@dataclasses.dataclass(slots=True)
class AnnotatedPeaks(Peaks):
    anno_type: AnnotationType
    peptide: str
    pivot: float
    mz: np.ndarray              # [float; n]
    intensity: np.ndarray       # [float; n]
    series: np.ndarray          # [str; n]
    position: np.ndarray        # [int; n]
    charge: list[np.ndarray]    # [[int; _]; n]
    loss: list[np.ndarray]      # [[str; _]; n]
    mods: list[np.ndarray]      # [[str; _]; n]

    def __post_init__(self):
        peak_data = [self.mz, self.intensity, self.series, self.position, self.charge, self.loss]
        arr_lengths = [len(x) for x in peak_data]
        assert len(set(arr_lengths)) == 1 
        # all have the same length

    def _boundary(self, series, side):
        mask = self.series == series
        idx = np.arange(len(self))[mask]
        pos = self.position[mask]
        idx_mask = np.empty((0,),dtype=bool)
        if side=='left':
            idx_mask = pos == pos.min()
        elif side=='right':
            idx_mask = pos == pos.max()
        else:
            raise ValueError(f"unrecognized side \"{side}\". try \"left\" or \"right\".")
        return idx[idx_mask]

    def lower_boundaries(self) -> list[int]:
        return np.concat([
            self._boundary('b','left'),
            self._boundary('y','left'),
        ])
    
    def upper_boundaries(self) -> tuple[list[int],list[int]]:
        return np.concat([
            self._boundary('b','right'),
            self._boundary('y','right'),
        ])

    def pairs(self) -> list[tuple[int,int]]:
        return [
            (i, j) 
            for i in range(len(self)) 
            for j in range(len(self)) 
            if self.series[i] == self.series[j] and self.position[i] + 1 == self.position[j]
        ]

    def lower_boundary_masses(self) -> list[float]:
        return self.mz[self.lower_boundaries()]
    
    def upper_boundary_masses(self) -> list[float]:
        return self.mz[self.upper_boundaries()]
   
    def pair_masses(self) -> list[float]:
        return [
            self.mz[j] - self.mz[i] 
            for (i, j) in self.pairs()
        ]

    def decharged_mass(self, i: int) -> float:
        c = self.charge[i][0]
        return (self.mz[i] * c) - (HYDROGEN_MASS * (c - 1))

    def decharged_lower_boundary_masses(self) -> list[float]:
        return [self.decharged_mass(x) for x in self.lower_boundaries()]
    
    def decharged_upper_boundary_masses(self) -> list[float]:
        return [self.decharged_mass(x) for x in self.upper_boundaries()]
   
    def decharged_pair_masses(self) -> list[float]:
        return [
            self.decharged_mass(j) - self.decharged_mass(i) 
            for (i, j) in self.pairs()
        ]

    def tabulate(self, precision = 2) -> str:
        headers = ["pos","ser","m/z","charge","mass","loss","int"]
        table = sorted(zip(
            self.position.tolist(),
            self.series.tolist(),
            self.mz.round(precision).tolist(),
            [x[0].item() for x in self.charge],
            [self.decharged_mass(i).round(precision).item() for i in range(len(self))],
            [x[0].item() for x in self.loss],
            self.intensity.round(precision).tolist(),
        ), key=lambda x: x)
        return self.peptide + '\n' + tabulate(table, headers=headers)

    def tabulate_pair_indices(self, idx, precision=4) -> str:
        headers = ["idx","pos","series","m/z","mass","mass dif","charge","loss","residue"]
        pos = [(self.position[i].item(),self.position[j].item()) for (i,j) in idx]
        series = [self.series[i] for (i,_) in idx]
        mz = [(self.mz[i].round(precision).item(),self.mz[j].round(precision).item()) for (i,j) in idx]
        mass= [(self.decharged_mass(i).round(precision).item(),self.decharged_mass(j).round(precision).item()) for (i,j) in idx]
        massdif = [self.decharged_mass(j) - self.decharged_mass(i) for (i,j) in idx]
        charge = [(self.charge[i][0].item(),self.charge[j][0].item()) for (i,j) in idx]
        loss = [(self.loss[i][0].item(),self.loss[j][0].item()) for (i,j) in idx]
        residue = [(self.peptide[i:j] if ser == 'b' else self.peptide[::-1][i:j]) for ((i,j),ser) in zip(pos,series)]
        table = zip(idx,pos,series,mz,mass,massdif,charge,loss,residue)
        return tabulate(table,headers=headers)

    def tabulate_pairs(self, precision=4) -> str:
        idx = self.pairs()
        return self.tabulate_pair_indices(idx, precision=precision)
        
    def compare(self, other: Self, mz_tolerance: float = None) -> ComparedAnnotations:
        """Compare two AnnotatedPeaks objects along pivot, peptide, and peak annotations. Wrap the output as a ComparedAnnotations object. Runs with complexity that is linear to the sum length of each AnnotatedPeaks."""
        pivot_err = self.pivot - other.pivot
        peptide_err = editdistance.eval(self.peptide, other.peptide)
        if mz_tolerance is None:
            mz_tolerance = 1e-10 # if no tolerance is passed, make it large enough to account for floating point epsilons.
        aln = np.array(list(merge_compare(self.mz[::-1], other.mz[::-1], mz_tolerance, verbose=True)))
        aln = (
            len(self.mz) - aln[0][::-1] - 1,
            len(other.mz) - aln[1][::-1] - 1,
        )
        aln = np.c_[aln[0],aln[1]]
        n_peaks = len(self)
        n_aln = len(aln)
        mz_err = n_peaks - n_aln
        # compare mz, pivots, peptides.

        intensity_err = np.empty((n_aln,),dtype=float)
        series_err = np.empty((n_aln,),dtype=bool)
        pos_err = np.empty((n_aln,),dtype=int)
        charge_aln = np.empty((n_aln,2),dtype=int)
        loss_aln = np.empty((n_aln,2),dtype=int)
        # initialize the rest of the return types.

        if n_aln > 0:
            self_aln = aln[:,0]
            other_aln = aln[:,1]
            int_err = self.intensity[self_aln] - other.intensity[other_aln]
            series_err = self.series[self_aln] == other.series[other_aln]
            pos_err = self.position[self_aln] - other.position[other_aln]
            charge_aln = np.array([first_match(self.charge[i],other.charge[j]) for (i,j) in aln])
            loss_aln = np.array([first_match(self.loss[i],other.loss[j]) for (i,j) in aln])
        # compare annotations of aligned peaks.

        return ComparedAnnotations(
            comp_type = (self.anno_type, other.anno_type),
            pivot_err = pivot_err,
            peptide_err = peptide_err,
            mz_err = mz_err,
            alignment = aln,
            int_err = intensity_err,
            series_err = series_err,
            pos_err = pos_err,
            charge_aln = charge_aln,
            loss_aln = loss_aln,
            mods_aln = None,
        )

    @classmethod
    def from_data(cls, *args, **kwargs) -> Self:
        """Raises NotImplementedError. AnnotatedPeaks must be constructed from one of its specific constructors, so this generic Peaks constructor is disabled."""
        raise NotImplementedError("AnnotatedPeaks cannot be constructed by the Peaks classmethod from_data. Use an AnnotatedPeaks constructor: from_evaluation, from_simulation, or from_benchmark.")

    @classmethod
    def from_fragment_masses(
        cls,
        fragment_masses: FragmentMasses,
        anno_cfg: AnnotationParams,
    ) -> Self:
        n = len(fragment_masses.mass)
        charges = [None for _ in range(n)]
        losses = [None for _ in range(n)]
        assert len(fragment_masses.mass) == len(np.unique(fragment_masses.mass))
        for i in range(n):
            loss_arr = np.concat(fragment_masses.losses[i], dtype=int)
            target_idx_arr = np.concat(fragment_masses.target_indices[i], dtype=int)
            losses[i] = np.array([anno_cfg.target_masses[t].right_fragment_space.loss_symbols[l] for (t,l) in zip(target_idx_arr,loss_arr)])
            charges[i] = np.concat([
                [c] * len(fragment_masses.losses[i][j]) 
                for (j, c) in enumerate(fragment_masses.charges[i])
            ])
            feature_order = np.argsort(np.concat(fragment_masses.costs[i], dtype=float))
            losses[i] = losses[i][feature_order]
            charges[i] = charges[i][feature_order]
        return cls(
            anno_type = AnnotationType.FRAGMENTS,
            peptide = '',
            pivot = 0.,
            mz = fragment_masses.mass,
            intensity = fragment_masses.intensity,
            series = np.full_like(fragment_masses.mass, '', dtype=str),
            position = np.zeros_like(fragment_masses.mass, dtype=int),
            charge = charges,
            loss = losses,
            mods = [],
        )

    # TODO
    @classmethod
    def from_candidate(cls,
        peaks: Peaks,
        cluster: int,
        annotation: tuple[AnnotationResult,AnnotationParams],
        alignment: tuple[AlignmentResult,AlignmentParams],
        enumeration: tuple[EnumerationResult,EnumerationParams],
        candidate: int = None,
    ) -> Self:
        anno_res, anno_cfg = annotation
        algn_res, algn_cfg = alignment
        enmr_res, enmr_cfg = enumeration
        candidate_cluster = enmr_res.candidates[cluster]
        peptide = candidate_cluster.get_sequence(candidate)
        mods = [None for _ in peptide]
        # TODO mods
        mz, intensity, charge = candidate_cluster.get_peaks(
            candidate,
            peaks,
            anno_res.pairs,
            anno_res.lower_boundaries,
            anno_res.upper_boundaries[cluster],
            algn_res.prod_topology[cluster],
            algn_res.lower_topology[cluster],
            algn_res.upper_topology[cluster],
        )
        loss = [np.empty((0,),dtype=str) for _ in mz]
        # TODO loss
        mz_ord = np.argsort(mz)
        mz = mz[mz_ord]
        intensity = intensity[mz_ord]
        series = candidate_cluster.get_series(candidate)
        position = candidate_cluster.get_position(candidate)
        return cls(
            anno_type = AnnotationType.CANDIDATE,
            peptide = peptide,
            pivot = anno_res.pivots.cluster_points[cluster],
            mz = mz,
            intensity = intensity,
            series = series,
            position = position,
            charge = charge,
            loss = loss,
            mods = mods,
        )

    @classmethod
    def from_simulation(cls,
        peptide: str,
        param: oms.Param,
        charges: int,
    ) -> Self:
        pivot = simulate_pivot(peptide)
        # find the pivot

        spectrum = generate_fragment_spectrum(
            peptide,
            param,
            max_charge=charges,
        )
        mz = list_mz(spectrum)
        intensity = list_intensity(spectrum)
        # generate mz and intensity data
        
        ion_data = list_ion_data(spectrum)
        pattern = re.compile(r'^([abcxyz])(\d+)(?:-([^+]+))?(\++)$')
        series, position, loss, charge = (np.array(a) for a in zip(*[pattern.match(x).groups() for x in ion_data]))
        position = position.astype(int)
        charge = np.array([len(x) for x in charge])
        loss[loss == np.array(None)] = ''
        # retrieve ion annotations

        return cls(
            anno_type = AnnotationType.SIMULATION,
            peptide = peptide,
            pivot = pivot,
            mz = mz,
            intensity = intensity,
            series = series,
            position = position,
            charge = [np.array([x,]) for x in charge],
            loss = [np.array([x,]) for x in loss],
            mods = [np.empty((0,),dtype=str) for _ in mz],
        )

    @classmethod
    def from_benchmark(
        cls,
        dataset: mzlib.SpectrumLibrary,
        record_idx: int,
        analyte_id: int = '1',
    ) -> Self:
        spectrum = dataset[record_idx]
        # retrieve spectrum 

        peaks = spectrum.peak_list
        n = len(peaks)
        mz = np.empty((n,), dtype=float)
        intensity = np.empty((n,), dtype=float)
        series = np.empty((n,), dtype=str)
        position = np.empty((n,), dtype=int)
        charge = [None for _ in range(n)]
        loss = [None for _ in range(n)]
        for i in range(n):
            peak_mz, peak_intensity, peak_anno, _ = peaks[i]
            mz[i] = peak_mz
            intensity[i] = peak_intensity
            # unpack the i-th peak.

            n_anno = len(peak_anno)
            charge[i] = np.empty((n_anno,), dtype=int)
            loss[i] = np.empty((n_anno,), dtype='<U25')
            for j in range(n_anno):
                if isinstance(peak_anno[j], mzpaf.annotation.Unannotated):
                    series[i] = ''
                    position[i] = 0
                    charge[i][j] = 0
                    loss[i][j] = ''
                    # no annotation, denote with empty.

                elif isinstance(peak_anno[j], mzpaf.annotation.PeptideFragmentIonAnnotation):
                    charge[i][j] = peak_anno[j].charge
                    # collect all annotated charges.

                    loss_names = []
                    for l in peak_anno[j].neutral_losses:
                        name = l.name
                        if name == 'NH3':
                            loss_names.append('H3N1')
                        elif name == 'H2O':
                            loss_names.append('H2O1')
                        elif not(name[-1].isnumeric()):
                            loss_names.append(name + '1')
                        else:
                            loss_names.append(name)
                    loss[i][j] = '-'.join(loss_names)
                    # collect all neutral loss annotations,
                    # represented as hyphen-separated strings.

                    if j == 0:
                        series[i] = peak_anno[j].series
                        position[i] = peak_anno[j].position
                        # select series and position from the first annotation.
        # parse peak data

        analyte = spectrum.analytes[analyte_id]
        pivot = analyte.peptide.mass / 2
        peptide, mods = zip(*analyte.peptide.sequence)
        peptide = ''.join(peptide)
        # parse analyte
        
        return cls(
            anno_type = AnnotationType.BENCHMARK,
            peptide = peptide,
            pivot = pivot,
            mz = mz,
            intensity = intensity,
            series = series,
            position = position,
            charge = charge,
            loss = loss,
            mods = mods,
        )
