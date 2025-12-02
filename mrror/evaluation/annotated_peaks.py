import dataclasses, re
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

@dataclasses.dataclass(slots=True)
class ComparedAnnotations:
    """The result of comparing two AnnotatedPeaks objects A, B as constructed by A.compare(B).
    - peptide_err: int, an edit distance between peptide sequences.
    - pivot_err: float, signed difference calculated as A.pivot - B.pivot.
    - mz_err: int, the number of un-aligned peaks in A.mz calculated as len(A) - len(alignment).
    - alignment: (n,2) int, index pairs aligning A.mz to B.mz.
    - int_err: (n,) float, difference in intensity between aligned peaks calculated as A.intensity[alignment[:,0]] - B.intensity[alignment[:,1]].
    - charge_err: (n,) int, difference in charge between aligned peaks.
    - series_err: (n,) bool, comparison in series between aligned peaks.
    - pos_err: (n,) int, difference in position between aligned peaks.
    - loss_aln: (n,2) int, optimal matches between losses of aligned peaks.
    - mods_aln: (n,2) int, optimal matches between modifications of aligned peaks."""
    peptide_err: int
    pivot_err: float
    mz_err: int
    alignment: np.ndarray   # [[int; 2]; n]
    int_err: np.ndarray     # [float; n]
    charge_err: np.ndarray  # [int; n]
    series_err: np.ndarray  # [bool; n]
    pos_err: np.ndarray     # [int; n]
    loss_aln: np.ndarray    # [[int; 2]; n]
    mods_aln: np.ndarray    # [[int; 2]; n]

@dataclasses.dataclass(slots=True)
class AnnotatedPeaks(Peaks):
    peptide: str
    pivot: float
    mz: np.ndarray          # [float; n]
    intensity: np.ndarray   # [float; n]
    charge: np.ndarray      # [int; n]
    series: np.ndarray      # [str; n]
    position: np.ndarray    # [int; n]
    loss: list[np.ndarray]  # [[str; _]; n]
    mods: list[np.ndarray]  # [[str; _]; n]

    @classmethod
    def from_data(cls, *args, **kwargs) -> Self:
        raise NotImplementedError("AnnotatedPeaks cannot be constructed by the Peaks classmethod from_data. Use an AnnotatedPeaks constructor: from_evaluation, from_simulation, or from_benchmark.")

    # TODO
    @classmethod
    def from_evaluation(cls,
        annotation: tuple[AnnotationResult,AnnotationParams],
        alignment: tuple[AlignmentResult,AlignmentParams] = None,
        enumeration: tuple[EnumerationResult,EnumerationParams] = None,
        candidate: tuple[int,int] = None,
    ) -> Self:
        passed_options = [x is not None for x in [alignment, enumeration, candidate]]
        if any(passed_options) and not(all(passed_options)):
            raise ValueError("from_evaluation must be called either with only the annotation arg or with all four args.")
        return None

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
        ion_data_chunks = [(re.search(r'\d', x).start(), re.search(r'\W|_', x).start()) for x in ion_data]
        series = np.array([x[:i] for ((i,_),x) in zip(ion_data_chunks,ion_data)])
        position = np.array([int(x[i:j]) for ((i,j),x) in zip(ion_data_chunks,ion_data)])
        charge = np.array([len([c for c in x[j:] if c == '+']) for ((_,j),x) in zip(ion_data_chunks,ion_data)])
        # generate series, position, and charge data.

        print("ion data", ion_data)

        return cls(
            peptide = peptide,
            pivot = pivot,
            mz = mz,
            intensity = intensity,
            charge = charge,
            series = series,
            position = position,
            loss = [np.empty((0,),dtype=str) for _ in mz],
            mods = [np.empty((0,),dtype=str) for _ in mz],
        )

    # TODO
    @classmethod
    def from_benchmark(cls,
        # ... some form of MzSpecLib data
    ) -> Self:
        return None

    def compare(self, other: Self, mz_tolerance: float = None):
        pivot_err = self.pivot - other.pivot
        peptide_err = editdistance.eval(self.peptide, other.peptide)
        if mz_tolerance is None:
            mz_tolerance = min([q - p for (p,q) in it.pairwise(self.mz)]) / 2
            # if no tolerance is passed, set it to half of the smallest distance between peaks in self.mz.
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
            charge_err = self.charge[self_aln] - other.charge[other_aln]
            series_err = self.series[self_aln] == other.series[other_aln]
            pos_err = self.position[self_aln] - other.position[other_aln]
            loss_aln = np.array([next(merge_compare_exact_unique(self.loss[i],other.loss[j]), (-1,-1)) for (i,j) in aln])
            mods_aln = np.array([next(merge_compare_exact_unique(self.mods[i],other.mods[j]), (-1,-1)) for (i,j) in aln])
        # compare annotations of aligned peaks.

        return ComparedAnnotations(
            pivot_err = pivot_err,
            peptide_err = peptide_err,
            mz_err = mz_err,
            alignment = aln,
            int_err = intensity_err,
            charge_err = charge_err,
            series_err = series_err,
            pos_err = pos_err,
            loss_aln = loss_aln,
            mods_aln = mods_aln,
        )
