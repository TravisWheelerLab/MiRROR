import dataclasses, re, enum
import itertools as it
from typing import Self, Iterable, Iterator

from ..util import decharge_peaks, HYDROGEN_MASS, merge_compare, merge_compare_fuzzy_unique, merge_compare_exact_unique, first_match
from .simulation import oms, generate_fragment_spectrum, list_mz, list_intensity, list_ion_data, simulate_pivot, DEFAULT_PARAM, COMPLEX_PARAM

import editdistance
import numpy as np
import mzspeclib as mzlib
import mzpaf
from pyteomics import mgf
from tabulate import tabulate

@dataclasses.dataclass(slots=True)
class SpectraParams:
    initial_b: bool

    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(
            initial_b = cfg['initial_b']
        )

@dataclasses.dataclass(slots=True)
class Peaks:
    mz: np.ndarray
    # [float; n]
    intensity: np.ndarray
    # [float; n]

    def __len__(self) -> int:
        return len(self.mz)

    def to_peaks(self) -> Self:
        return Peaks(
            mz = self.mz,
            intensity = self.intensity,
        )

    @classmethod
    def from_mgf(
        cls,
        mgf: mgf.IndexedMGF,
    ) -> Self:
        return cls(
            mz = mgf['m/z array'],
            intensity = mgf['intensity array'],
        )

    @classmethod
    def from_data(
        cls,
        mz: Iterable,
        intensity: Iterable = None,
    ) -> Self:
        return cls(
            mz = np.array(mz),
            intensity = np.array(intensity) if intensity else np.ones_like(mz),
        )

@dataclasses.dataclass(slots=True)
class SimulatedPeaks(Peaks):
    mz: np.ndarray          # [float; n]
    intensity: np.ndarray   # [float; n]
    series: np.ndarray      # [str; n]
    position: np.ndarray    # [int; n]
    charge: np.ndarray      # [str; n]
    pivot: float
    peptide: str

    def _boundary(self, series, side):
        mask = self.series == series
        idx = np.arange(len(self))[mask]
        pos = self.position[mask]
        if side=='left':
            opt = np.argmin(pos)
        elif side=='right':
            opt = np.argmax(pos)
        return idx[opt]

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
        return [(i, j) for i in range(len(self)) for j in range(i + 1, len(self)) if self.series[i] == self.series[j] and self.position[i] + 1 == self.position[j]]

    def _paths(self, series) -> Iterator[list[int]]:
        mask = self.series == series
        idx = np.arange(len(self))[mask].tolist()
        pos = self.position[mask]
        mz = self.mz[mask]
        path = [idx[0]]
        prev_pos = pos[0]
        prev_mz = mz[0]
        for (i,curr_pos,curr_mz) in zip(idx[1:],pos[1:],mz[1:]):
            if curr_pos != prev_pos + 1 or prev_mz < self.pivot < curr_mz:
                yield path
                path = [i]
            else:
                path.append(i)    
            prev_pos = curr_pos
            prev_mz = curr_mz
        yield path

    def paths(self):
        return (
            list(self._paths('b')),
            list(self._paths('y')),
        )

    def _alignments(self):
        # naive, doesn't work in most cases. 
        # todo - rewrite for transformed spectra
        b_paths, y_paths = self.paths()
        for bp in b_paths:
            for yp in y_paths:
                yp = yp[::-1]
                if all(self.position[bp] == len(self.peptide) - self.position[yp]):
                    if all(self.position[bp] < self.position[yp]):
                        yield list(zip(bp,yp))
                    else:
                        yield list(zip(yp,bp))

    def alignments(self):
        return list(self._alignments())

    def zip(self) -> list[tuple[float,float,str,int,str]]:
        return list(zip(
                np.round(self.mz,3).tolist(),
                np.round(self.intensity,3).tolist(),
                self.series.tolist(),
                self.position.tolist(),
                self.charge.tolist(),
            ))

    def get_mask(
        self,
        series: str = None,
        charge: str = None,
    ):
        mask = np.full_like(self.mz, True, dtype=np.bool)
        if series:
            mask = np.logical_and(mask, self.series == series)
        if charge:
            mask = np.logical_and(mask, self.charge == charge)
        return mask

    def subpeaks(
        self,
        mask: np.ndarray,
    ) -> Self:
        return type(self)(
            self.mz[mask],
            self.intensity[mask],
            self.series[mask],
            self.position[mask],
            self.charge[mask],
            self.pivot,
            self.peptide,
        )

    def truncate(
        self,
        depth: int,
        initial: bool = False,
        terminal: bool = False,
        series: str = None,
        charge: str = None,
    ) -> Self:
        """Drop some number of the first and/or last ions of a given series and/or charge."""
        if not(initial or terminal):
            initial = True
            terminal = True
        n = len(self.peptide)
        feature_mask = np.bitwise_not(self.get_mask(series=series, charge=charge))
        trunc_mask = self.get_mask()
        if initial:
            trunc_mask = np.logical_and(trunc_mask, depth < self.position)
        if terminal:
            trunc_mask = np.logical_and(trunc_mask, (n - depth) > self.position)
        return self.subpeaks(np.logical_or(feature_mask,trunc_mask))

    def occlude(
        self,
        pos,
        series: str = None,
        charge: str = None,
    ) -> Self:
        feature_mask = np.bitwise_not(self.get_mask(series=series, charge=charge))
        occl_mask = self.get_mask()
        if type(pos) is int:
            pos = [pos]
        for x in pos:
            occl_mask = np.logical_and(occl_mask, self.position != x)
        return self.subpeaks(np.logical_or(feature_mask, occl_mask))

    def insert(
        self,
        mz: np.ndarray,
        intensity: np.ndarray = None,
    ) -> Self:
        if intensity is None:
            intensity = np.full_like(mz, np.mean(self.intensity))
        series = np.full_like(mz, '.', dtype=str)
        position = np.full_like(mz, 0, dtype=int)
        charge = np.full_like(mz, '.', dtype=str)
        new_mz = np.concat([self.mz, mz])
        new_intensity = np.concat([self.intensity, intensity])
        new_series = np.concat([self.series, series])
        new_position = np.concat([self.position, position])
        new_charge = np.concat([self.charge, charge])
        order = np.argsort(new_mz)
        return type(self)(
            new_mz[order],
            new_intensity[order],
            new_series[order],
            new_position[order],
            new_charge[order],
            self.pivot,
            self.peptide,
        )

    def corrupt(
        self,
        snr: float = None,
        num: int = None,
        padding: float = 200.,
    ) -> Self:
        if snr:
            num = int(len(self) / snr)
        elif num is None:
            num = len(self)
        noise = np.random.uniform(max(0, self.mz.min() - padding), self.mz.max() + padding, num)
        return self.insert(noise)

    # TODO 2: should probably use an enum for this
    @classmethod
    def from_peptide(cls,
        peptide: str,
        mode: str = 'simple',
        initial_b: bool = False,
    ) -> Self:
        param = COMPLEX_PARAM if mode == "complex" else DEFAULT_PARAM
        if initial_b:
            param.setValue("add_first_prefix_ion", "true")
        spectrum = generate_fragment_spectrum(
            peptide,
            param,
            max_charge= 1 if mode == "simple" else 3,
        )
        ion_data = list_ion_data(spectrum)
        ion_data_chunks = [(re.search(r'\d', x).start(), re.search(r'\W|_', x).start()) for x in ion_data]
        return cls(
            mz = list_mz(spectrum),
            intensity = list_intensity(spectrum),
            series = np.array([x[:i] for ((i,_),x) in zip(ion_data_chunks,ion_data)]),
            position = np.array([int(x[i:j]) for ((i,j),x) in zip(ion_data_chunks,ion_data)]),
            charge = np.array([x[j:] for ((_,j),x) in zip(ion_data_chunks,ion_data)]),
            pivot = simulate_pivot(peptide),
            peptide = peptide,
        )

@dataclasses.dataclass(slots=True)
class PeaksDataset:
    _data: list[Peaks]
    _input: str

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i) -> Peaks:
        return self._data[i]

    def __iter__(self) -> Iterator[Peaks]:
        return self._data.__iter__()
    
    @classmethod
    def from_peptide(cls,
        input: str,
        initial_b: bool = False,
    ) -> Self:
        return cls(
            _data = [SimulatedPeaks.from_peptide(*input.split('::'), initial_b=initial_b)],
            _input = input,
        )

    @classmethod
    def from_fasta(cls,
        input: str, 
        **kwargs,
    ) -> Self:
        pass
        
    @classmethod
    def from_mgf(cls,
        input: str, 
        **kwargs,
    ) -> Self:
        pass
        
    @classmethod
    def from_mzlib(cls,
        input: str, 
        **kwargs,
    ) -> Self:
        pass
        
@dataclasses.dataclass(slots=True)
class AugmentedPeaks(Peaks):
    mz: np.ndarray
    # [float; n]
    intensity: np.ndarray
    # [float; n]
    indices: np.ndarray
    # [int; n]
    charges: np.ndarray
    # [int; n]

    def get_original_indices(self,
        augmented_indices: np.ndarray,
    ) -> np.ndarray:
        return self.indices[augmented_indices]

    def get_augmenting_charges(self,
        augmented_indices: np.ndarray,
    ) -> np.ndarray:
        return self.charges[augmented_indices]

    @classmethod
    def from_data(cls,
        mz: Iterable[float],
        intensity: Iterable[float],
        indices: Iterable[int],
        charges: Iterable[int],
    ) -> Self:
        return cls(
            mz = np.array(mz),
            intensity = np.array(mz),
            indices = np.array(indices),
            charges = np.array(charges),
        )

    @classmethod
    def from_peaks(cls,
        peaks: Peaks,
        charges: np.ndarray = None,
        pivot_point: float = None,
    ) -> Self:
        reflection = None
        if pivot_point:
            reflector = 2 * pivot_point
            reflection = lambda x: reflector - x
        # construct the optional reflection

        mz, deindexer, charge_table = decharge_peaks(
            peaks.mz,
            charges,
            transformation=reflection,
        )
        # decharge peaks

        intensity = peaks.intensity[deindexer]
        # remap intensity to the new peaks

        return cls(
            mz = mz,
            intensity = intensity,
            indices = deindexer,
            charges = charge_table
        )

class LabelType(enum.Enum):
    FRAGMENTS = 0
    CANDIDATE = 1
    SIMULATION = 2
    BENCHMARK = 3

@dataclasses.dataclass(slots=True)
class ComparedLabels:
    """The result of comparing two LabeledPeaks objects A, B as constructed by A.compare(B).
    - comp_type: tuple, a pair of LabelType enums which record the manner in which A and B were constructed.
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
    comp_type: tuple[LabelType,LabelType]
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
class AbstractLabeledPeaks(Peaks):
    label_type: LabelType
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

    def _boundary(
        self,
        series,
        side,
        k: int = None,
    ):
        mask = self.series == series
        idx = np.arange(len(self))[mask]
        pos = self.position[mask]
        idx_mask = np.empty((0,),dtype=bool)
        if side=='left':
            idx_mask = (pos == pos.min()) if (k is None) else (pos == k)
        elif side=='right':
            idx_mask = (pos == pos.max()) if (k is None) else (pos == len(self.peptide) - k)
        else:
            raise ValueError(f"unrecognized side \"{side}\". try \"left\" or \"right\".")
        return idx[idx_mask]

    def lower_boundaries(
        self,
        k: int = None,
    ) -> list[int]:
        return np.concat([
            self._boundary('b','left', k),
            self._boundary('y','left', k),
        ])
    
    def upper_boundaries(
        self,
        k: int = None,
    ) -> tuple[list[int],list[int]]:
        return np.concat([
            self._boundary('b','right', k),
            self._boundary('y','right', k),
        ])

    def pairs(
        self,
        k: int = 1,
    ) -> list[tuple[int,int]]:
        return [
            (i, j) 
            for i in range(len(self)) 
            for j in range(len(self)) 
            if self.series[i] == self.series[j] and self.position[i] + k == self.position[j]
        ]

    def lower_boundary_masses(self) -> list[float]:
        return self.mz[self.lower_boundaries()]
    
    def upper_boundary_masses(self) -> list[float]:
        return self.mz[self.upper_boundaries()]
   
    def pair_masses(
        self,
        k: int = 1,
    ) -> list[float]:
        return [
            self.mz[j] - self.mz[i] 
            for (i, j) in self.pairs(k)
        ]

    def decharged_mass(self, i: int) -> float:
        c = self.charge[i][0]
        return (self.mz[i] * c) - (HYDROGEN_MASS * (c - 1))

    def decharged_lower_boundary_masses(self) -> list[float]:
        return [self.decharged_mass(x) for x in self.lower_boundaries()]
    
    def decharged_upper_boundary_masses(self) -> list[float]:
        return [self.decharged_mass(x) for x in self.upper_boundaries()]
   
    def decharged_pair_masses(
        self,
        k: int = 1,
    ) -> list[float]:
        return [
            self.decharged_mass(j) - self.decharged_mass(i) 
            for (i, j) in self.pairs(k)
        ]

    def tabulate(self, precision = 4) -> str:
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
        return f"peptide={self.peptide}\n" + tabulate(table, headers=headers)

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
        
    def compare(self, other: Self, mz_tolerance: float = None) -> ComparedLabels:
        """Compare two LabeledPeaks objects along pivot, peptide, and peak annotations. Wrap the output as a ComparedLabels object. Runs with complexity that is linear to the sum length of each LabeledPeaks."""
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

        return ComparedLabels(
            comp_type = (self.label_type, other.label_type),
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
        """Raises NotImplementedError. *LabeledPeaks must be constructed from a subclass's classmethod constructor, so this generic Peaks constructor is disabled."""
        raise NotImplementedError("LabeledPeaks objects cannot be constructed with the Peaks classmethod from_data. Use a subclass classmethod constructor instead, e.g.: SimulationLabeledPeaks.from_oms or BenchmarkLabeledPeaks.from_mzlib.")
 
class SimulationLabeledPeaks(AbstractLabeledPeaks):
    @classmethod
    def from_oms(cls,
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
        ## this regex pattern is the only AI-generated code in MiRROR. it has been tested extensively.
        series, position, loss, charge = (np.array(a) for a in zip(*[pattern.match(x).groups() for x in ion_data]))
        position = position.astype(int)
        charge = np.array([len(x) for x in charge])
        loss[loss == np.array(None)] = ''
        # retrieve ion annotations

        return cls(
            label_type = LabelType.SIMULATION,
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

class BenchmarkLabeledPeaks(AbstractLabeledPeaks):
    @classmethod
    def from_mzlib(
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
                    # no annotation, denote with zeroes and empty strings.

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
            label_type = LabelType.BENCHMARK,
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
