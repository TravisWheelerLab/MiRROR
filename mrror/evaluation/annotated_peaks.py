import dataclasses, re, enum
import itertools as it
from typing import Self

import editdistance
import numpy as np

from ..util import merge_compare_fuzzy_unique, merge_compare_exact_unique
from ..spectra.types import Peaks
from ..spectra.simulation import oms, generate_fragment_spectrum, list_mz, list_intensity, list_ion_data, simulate_pivot, DEFAULT_PARAM, COMPLEX_PARAM
from ..fragments.types import PairResult, PivotResult, BoundaryResult

from ..annotation import AnnotationResult, AnnotationParams
from ..alignment import AlignmentResult, AlignmentParams
from ..enumeration import EnumerationResult, EnumerationParams

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
        peak_data = [self.mz, self.intensity, self.series, self.position, self.charge, self.loss, self.mods]
        arr_lengths = [len(x) for x in peak_data]
        assert len(set(arr_lengths)) == 1 
        # all have the same length
        assert len(self.mz) == 2 * (len(self.peptide) - 1)
        # two peaks for every nonterminal symbol in the peptide.

    def _boundary(self, series, side):
        mask = self.series == series
        idx = np.arange(len(self))[mask]
        pos = self.position[mask]
        if side=='left':
            return idx[np.argmin(pos)]
        elif side=='right':
            return idx[np.argmax(pos)]
        else:
            raise ValueError(f"unrecognized side \"{side}\". try \"left\" or \"right\".")

    def lower_boundaries(self) -> tuple[int,int]:
        return (
            self._boundary('b','left'),
            self._boundary('y','left'),
        )
    
    def upper_boundaries(self) -> tuple[int,int]:
        return (
            self._boundary('b','right'),
            self._boundary('y','right'),
        )

    def pairs(self) -> list[tuple[int,int]]:
        return [
            (i, j) 
            for i in range(len(self)) 
            for j in range(i + 1, len(self)) 
            if self.series[i] == self.series[j] and self.position[i] + 1 == self.position[j]
        ]
   
    def compare(self, other: Self, mz_tolerance: float = None) -> ComparedAnnotations:
        """Compare two AnnotatedPeaks objects along pivot, peptide, and peak annotations. Wrap the output as a ComparedAnnotations object. Runs with complexity that is linear to the sum length of each AnnotatedPeaks."""
        pivot_err = self.pivot - other.pivot
        peptide_err = editdistance.eval(self.peptide, other.peptide)
        if mz_tolerance is None:
            mz_tolerance = 1e-10 # if no tolerance is passed, make it large enough to account for floating point epsilons.
        aln = np.array(list(merge_compare_fuzzy_unique(self.mz, other.mz, mz_tolerance)))
        n_peaks = len(self)
        n_aln = len(aln)
        mz_err = n_peaks - n_aln
        # compare mz, pivots, peptides.

        intensity_err = np.empty((n_aln,),dtype=float),
        charge_err = np.empty((n_aln,),dtype=int),
        series_err = np.empty((n_aln,),dtype=bool),
        pos_err = np.empty((n_aln,),dtype=int),
        loss_aln = np.empty((n_aln,2),dtype=int),
        mods_aln = np.empty((n_aln,2),dtype=int),
        # initialize the rest of the return types.

        if n_aln > 0:
            self_aln = aln[:,0]
            other_aln = aln[:,1]
            int_err = self.intensity[self_aln] - other.intensity[other_aln]
            series_err = self.series[self_aln] == other.series[other_aln]
            pos_err = self.position[self_aln] - other.position[other_aln]
            charge_aln = np.array([next(merge_compare_exact_unique(self.charge[i],other.charge[j]), (-1,-1)) for (i,j) in aln])
            loss_aln = np.array([next(merge_compare_exact_unique(self.loss[i],other.loss[j]), (-1,-1)) for (i,j) in aln])
            mods_aln = np.array([next(merge_compare_exact_unique(self.mods[i],other.mods[j]), (-1,-1)) for (i,j) in aln])
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
            mods_aln = mods_aln,
        )

    @classmethod
    def from_data(cls, *args, **kwargs) -> Self:
        raise NotImplementedError("AnnotatedPeaks cannot be constructed by the Peaks classmethod from_data. Use an AnnotatedPeaks constructor: from_evaluation, from_simulation, or from_benchmark.")

    # TODO
    @classmethod
    def from_evaluation(cls,
        peaks: Peaks,
        annotation: tuple[AnnotationResult,AnnotationParams],
        cluster: int,
        alignment: tuple[AlignmentResult,AlignmentParams] = None,
        enumeration: tuple[EnumerationResult,EnumerationParams] = None,
        candidate: int = None,
    ) -> Self:
        passed_options = [x is not None for x in [alignment, enumeration, candidate]]
        any_passed = any(passed_options)
        all_passed = all(passed_options)
        if any_passed and not(all_passed):
            raise ValueError("from_evaluation must be called either with only the annotation arg or with all four args.")
        else:
            if all_passed:
                anno_type = AnnotationType.CANDIDATE
                candidate_cluster = enumeration[0].candidates[cluster]
                peptide = candidate_cluster.get_sequence(candidate)
                mz, intensity, charge = candidate_cluster.get_peaks(
                    candidate,
                    peaks,
                    annotation[0].pairs,
                    annotation[0].lower_boundaries,
                    annotation[0].upper_boundaries[cluster],
                    alignment[0].prod_topology[cluster],
                    alignment[0].lower_topology[cluster],
                    alignment[0].upper_topology[cluster],
                )
                mz_ord = np.argsort(mz)
                mz = mz[mz_ord]
                intensity = intensity[mz_ord]
                series = candidate_cluster.get_series(candidate)
                position = candidate_cluster.get_position(candidate)
                # loss, mods = candidate_cluster.get_annotation(candidate)
                loss = np.zeros_like(mz)
                mods = np.zeros_like(mz)
            else:
                anno_type = AnnotationType.FRAGMENTS
                peptide = ""
                mz = annotation[0].peaks.mz
                intensity = annotation[0].peaks.intensity
                series = ['' for _ in mz]
                position = [-1 for _ in mz]
                charge = annotation[0].get_peak_charges(cluster)
                loss = annotation[0].get_peak_losses(cluster)
                mods = annotation[0].get_peak_modifications(cluster)
                # peptide, series, and position cannot be calculated without a peptide. the rest can be derived with less specificity from the AnnotationResult.
            return cls(
                anno_type = anno_type,
                peptide = peptide,
                pivot = annotation[0].pivots.cluster_points[cluster],
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

    # TODO
    @classmethod
    def from_benchmark(cls,
        # ... some form of MzSpecLib data
    ) -> Self:
        return None 
        # return cls(
        #     anno_type = Annotation.BENCHMARK,
        #     ...
        # )
