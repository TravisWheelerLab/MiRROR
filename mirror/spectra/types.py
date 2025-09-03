from abc import ABC, abstractclassmethod
from typing import Any, Iterator, Iterable, Self, Callable
from dataclasses import dataclass
from copy import deepcopy
import statistics as stat
import itertools as it
import warnings

import numpy as np
import pyopenms as oms
from mzpaf import PeptideFragmentIonAnnotation, Unannotated
from mzspeclib.validate import ValidationWarning as mzlib_ValidationWarning
from pyopenms import MSSpectrum

from ..io import mgf, read_mgf, mzlib, read_mzlib
from ..util import merge_in_order, interleave

from .simulation import generate_fragment_spectrum, DEFAULT_PARAM, COMPLEX_PARAM

@dataclass
class SpectrumParams:
    # spectrum preparation
    # peak detection
    # peak annotation
    # spectrum alignment
    score_model: str
    score_threshold: float

class MzArray:
    """Wrapper for a numpy array containing relative frequency values, 
    associating a range of m/z values to the index range of the array.
    Represents the spectrum as a map from m/z to relative frequency.
    Constructed via the prepare_spectrum function."""

    def __init__(self, 
        data: np.ndarray,
        resolution: float,
        min_mz: float,
        max_mz: float,
    ):
        self.data = data
        self.resolution = resolution
        self.min = min_mz
        self.max = max_mz

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i: int):
        return self.data[i]

    def convert_mz_to_index(self, mz: float):
        normalized_mz = (mz - self.min) / self.max
        n = len(self)
        return np.floor(normalized_mz * n)

    def get_intensity(self, mz: float):
        return self[self.convert_mz_to_index(mz)]

class PeakList:
    """A collection of peaks sorted by m/z.
    Constructed via from_mzlib and from_mgf class methods."""

    def __init__(self,
        mz: np.ndarray, 
        intensity: np.ndarray,
    ):
        n = len(mz)
        assert len(intensity) == n
        self.n = n
        self.mz = mz
        self.intensity = intensity

    @classmethod
    def from_mz(cls,
        mz: np.ndarray,
    ) -> Self:
        return cls(
            mz = mz,
            intensity = np.ones_like(mz))

    @classmethod
    def from_mzlib(cls, 
        dataset: mzlib.SpectrumLibrary,
        i: int,
    ) -> Self:
        """Given a spectrum library `dataset` and an index `i`,
        retrieve the `i`th spectrum and use it to construct a PeakList."""
        spectrum = dataset[i]
        mz, intensity, _, __ = zip(*spectrum.peak_list)
        return cls(
            mz = mz, 
            intensity = intensity)
        
    @classmethod
    def from_mgf(cls, 
        dataset: mgf.IndexedMGF, 
        i: int,
    ) -> Self:
        """Given an MGF `dataset` and an index `i`,
        retrieve the `i`th spectrum and use it to construct a PeakList."""
        spectrum = dataset[i]
        mz = spectrum['m/z array']
        intensity = spectrum['intensity array']
        return cls(
            mz = mz, 
            intensity = intensity)

    def with_transformation(self,
        func: Callable,
    ) -> Self:
        new_peaklist = deepcopy(self)
        new_peaklist.mz = func(self.mz)
        return new_peaklist

    def __len__(self) -> int:
        return self.n
    
    def __getitem__(self, index):
        return self.mz[index]
    
    def __iter__(self) -> Iterator:
        return self.mz.__iter__()
    
    def get_intensity(self, i: int) -> float:
        """The intensity of the `i`th peak."""
        return self.intensity[i]

class AnnotatedPeakList(PeakList):
    """A PeakList with additional data associated to peaks.
    Each annotation describes charge state, losses, and post-translation modification.
    Each peak may have multiple annotations."""

    def __init__(self,
        mz: np.ndarray, 
        intensity: np.ndarray,
        charge: list,
        losses: list,
        metadata: dict,
    ):
        n = len(mz)
        assert len(intensity) == len(charge) == n
        self.charge = charge
        self.losses = losses
        self.metadata = metadata
        super(AnnotatedPeakList, self).__init__(mz, intensity)
    
    def __repr__(self):
        metadata_repr = '\n'.join(f"- {key}: {list(val)}" for key, val in self.metadata.items())
        return f"AnnotatedPeakList:\nmz: {self.mz}\nintensity: {self.intensity}\ncharge: {self.charge}\nlosses: {self.losses}\nmetadata:\n{metadata_repr}"

    def get_charge(self, i: int) -> int:
        """Charge annotations of the `i`th peak."""
        return self.charge[i]
    
    def get_losses(self, i: int) -> int:
        """Loss annotations of the `i`th peak."""
        return self.losses[i]

    def get_metadata_keys(self):
        return self.metadata.keys()
            
    def get_metadata(self, i: int, key: str) -> Any:
        """Metadata for `key` of the `i`th peak."""
        return self.metadata[key][i]

    def get_states(self, i: int, metadata_keys = []) -> set:
        """The set of annotation vector(s) for the `i`th peak, including metadata if the `metadata_keys` kwarg is passed."""
        charges = self.get_charge(i)
        losses = self.get_losses(i)
        metadata = [self.get_metadata(i, k) for k in metadata_keys]
        states = [charges, losses, *metadata]
        return set(zip(*states))

    def compare_peak_annotations(self, other, i: int, metadata_keys):
        # collate self
        s_data = set(self.get_states(i, metadata_keys))
        # collate other
        o_data = set(other.get_states(i, metadata_keys))
        # compare
        matches = s_data.intersection(o_data)
        misses = s_data.symmetric_difference(o_data)
        return matches, misses

    def compare(self, other):
        if len(self) != len(other):
            raise ValueError("Unequal number of peaks; cannot directly compare annotations for distinct peak lists.")
        if self.mz != other.mz:
            raise ValueError("Mismatched m/z values; cannot directly compare annotations for distinct peak lists.")
        common_keys = set(self.get_metadata_keys()).intersection(other.get_metadata_keys())
        matches, misses = zip(*map(
            lambda i: self.compare_peak_annotations(other, i, common_keys), 
            range(len(self))))
        return matches, misses

class BenchmarkPeakList(AnnotatedPeakList):
    """An annotated peak list with a target peptide.
    Implements constructors from_mzlib, from_mgf, from_9species, from_simulation."""

    def __init__(self,
        peptide: str,
        mz: np.ndarray, 
        intensity: np.ndarray,
        charge: list,
        losses: list,
        series: list,
        position: list,
        *args,
        **kwargs,
    ):
        self.peptide = peptide
        self.m = len(peptide)
        self.series = series
        self.position = position
        super(BenchmarkPeakList, self).__init__(mz, intensity, charge, losses, **kwargs)
        self.y_idx = sorted(self._series_peak_indices('y'), key = lambda i: self.get_position(i)[0])
        self.y_pos = [int(self.get_position(i)) for i in self.y_idx]
        self.b_idx = sorted(self._series_peak_indices('b'), key = lambda i: self.get_position(i)[0])
        self.b_pos = [int(self.get_position(i)) for i in self.b_idx]
        # bin peak indices by their peptide index as calculated from their series and position. peak b1 has peptide index 0, peak y1 has peptide index n, b2 has idx 1, y2 has index n - 1, etc. series data is retained via the inner dictionaries, because we don't want to pair across series even when they have the same index.
        self.peak_idx_by_peptide_idx = [{'b':[],'y':[]} for _ in range(len(peptide) + 1)]
        for i in range(len(self.mz)):
            series = self.series[i]
            position = int(self.position[i])
            if series == 'b':
                idx = position
            elif series == 'y':
                idx = self.m - position
            # print(f"peak:{i} series:{series} pos:{position} idx:{idx}")
            self.peak_idx_by_peptide_idx[idx][series].append(i)

    def get_pairs(self,
        l: int,
        r: int,
    ) -> Iterator[tuple[int,int,str]]:
        """Given left and right peptide indices, return all (left peak index, right peak index, residue) data where residue is the character given by peptide[l:r] == peptide[l:l+1]. Does not (yet) support positions with a gap greater than one, i.e., r - l > 1 and `residue` is a kmer."""
        if r != l + 1:
            raise ValueError(f"indices {l}, {r} are not consecutive; this method does not yet support kmer queries.")
        else:
            l_peaks = self.peak_idx_by_peptide_idx[l]
            r_peaks = self.peak_idx_by_peptide_idx[r]
            l_b = l_peaks['b']
            r_b = r_peaks['b']
            l_y = l_peaks['y']
            r_y = r_peaks['y']
            residue = self.peptide[l:r]
            for l_idx in l_b:
                for r_idx in r_b:
                    # dif = self.mz[r_idx] - self.mz[l_idx]
                    yield (l_idx, r_idx, residue)#, dif)
            for l_idx in l_y:
                for r_idx in r_y:
                    # dif = self.mz[l_idx] - self.mz[r_idx]
                    yield (l_idx, r_idx, residue)#, dif)

    # def get_pivots(self) -> Iterator[tuple[float,Any]]:
    #     """Identify overlap and virtual pivots in the spectrum. Virtual pivots are only returned if no overlap pivots can be found. Overlap pivots take the form (pivot point, (pair, pair)) while virtual pivots take the form (pivot point, None)."""

    #     midpoints = []
    #     # step 1 - look for overlap pivots
    #     c = 0
    #     for (l,r) in it.pairwise(range(self.m + 1)):
    #         pairs = list(self.get_pairs(l,r))
    #         for (i, (l1, r1, res)) in enumerate(pairs):
    #             for (l2, r2, _) in pairs[i + 1:]: # all residues are the same in a call to get_pairs
    #                 pivot_point = (self[l1] + self[l2] + self[r1] + self[r2]) / 4
    #                 midpoints.append(pivot_point)
    #                 if (l1 < l2 < r1 < r2) or (l2 < l1 < r2 < r1):
    #                     c += 1
    #                     yield (
    #                         round(pivot_point, 2),
    #                         ((l1,r1),(l2,r2)))
    #     if c == 0:
    #         # step 2 - when there are no overlap pivots, return the mode of virtual pivots.
    #         midpoints = [round(x, 2) for x in midpoints]
    #         yield (
    #             stat.mode(midpoints),
    #             None)

    @staticmethod
    def _get_extremal(arr: Iterator[int], weights: Iterable[int], extrema: int, reverse: bool = False) -> Iterator[int]:
        enum_arr = list(enumerate(arr))
        if reverse:
            enum_arr = enum_arr[::-1]
        for (i, val) in enum_arr:
            if weights[i] == extrema:
                yield val
            else:
                return

    def get_prefix(self, peak_idx) -> str:
        series = self.get_series(peak_idx)
        position = int(self.get_position(peak_idx))
        if series == 'b':
            idx = position
            return self.peptide[:idx]
        elif series == 'y':
            idx = self.m - position
            return self.peptide[idx:]

    def get_suffix(self, peak_idx) -> str:
        series = self.get_series(peak_idx)
        position = int(self.get_position(peak_idx))
        if series == 'b':
            idx = position
            return self.peptide[idx:]
        elif series == 'y':
            idx = self.m - position
            return self.peptide[:idx]

    def get_left_boundaries(self) -> Iterator[tuple[int,str]]:
        """Scan the low end of the spectrum for peaks whose annotated position is minimal w.r.t. its series; returns all the b ions whose position is less than or equal to any other b ion, and likewise all y ions whose position is less than or equal to any other y ion."""
        return [(peak_idx, self.get_prefix(peak_idx)) for peak_idx in it.chain(
            self._get_extremal(self.b_idx, self.b_pos, self.b_pos[0], reverse = False),
            self._get_extremal(self.y_idx, self.y_pos, self.y_pos[0], reverse = False))]

    def get_right_boundaries(self) -> Iterator[tuple[int,str]]:
        """Scan the high end of the spectrum for peaks whose annotated position is maximal w.r.t. its series; returns all the b ions whose position is greater  than or equal to any other b ion, and likewise all y ions whose position is greater than or equal to any other y ion."""
        return [(peak_idx, self.get_suffix(peak_idx)) for peak_idx in it.chain(
            self._get_extremal(self.b_idx, self.b_pos, self.b_pos[-1], reverse = True),
            self._get_extremal(self.y_idx, self.y_pos, self.y_pos[-1], reverse = True))]
            
    def from_mzlib(cls, 
        dataset: mzlib.SpectrumLibrary,
        i: int,
    ):
        """Given a spectrum library `dataset` and an index `i`,
        retrieve the `i`th spectrum and use it to construct a BenchmarkPeakList.
        The resulting object will have metadata for 'mass_error', 'series', 'position', and 'losses'."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spectrum = dataset[i]
            # peptide sequence
            analyte = list(spectrum.analytes.values())[0]
            seq = ''.join(map(
                lambda x: x[0],
                analyte.peptide.sequence))
            # peak data
            mz, intensity, annotations, _ = zip(*spectrum.peak_list)
            # peak annotations
            def parse_annotation(annotation):
                if type(annotation) == PeptideFragmentIonAnnotation:
                    return (
                        [f"+{annotation.charge}"], 
                        [annotation.mass_error.mass_error], 
                        [annotation.series], 
                        [annotation.position], 
                        [tuple([''] if len(annotation.neutral_losses) == 0 else [loss.name for loss in annotation.neutral_losses])])
                else:
                    return (
                        ['+1'],
                        [0.],
                        [''],
                        [-1],
                        [tuple([''])])
            charge, mass_error, series, position, losses = zip(*map(
                lambda x: parse_annotation(x[0]),
                annotations))
            # done
            return cls(
                peptide = seq,
                mz = list(mz), 
                intensity = list(intensity),
                charge = list(charge),
                losses = losses,
                metadata = {
                    "mass_error": mass_error,
                    "series": list(series),
                    "position": list(position)})

    @classmethod
    def from_mgf(cls, 
        dataset: mgf.IndexedMGF, 
        i: int,
    ):
        """Given an MGF `dataset` and an index `i`,
        retrieve the `i`th spectrum and use it to construct a BenchmarkPeakList.
        The resulting object will not have metadata."""
        spectrum = dataset[i]
        # peptide sequence
        seq = spectrum['params']['seq']
        # peak data
        mz = spectrum['m/z array']
        n = len(mz)
        intensity = spectrum['intensity array']
        # peak annotations
        charge = [[c] for c in spectrum['charge array']]
        losses = [[['']] for _ in range(n)]
        # done
        return cls(
            peptide = seq,
            mz = mz, 
            intensity = intensity,
            charge = charge,
            losses = losses,
            metadata = {})

    @classmethod
    def from_9species(cls, 
        dataset: mzlib.SpectrumLibrary,
        i: int,
    ):
        """Given a spectrum library `dataset` and an index `i`,
        retrieve the `i`th spectrum and use it to construct a BenchmarkPeakList.
        The resulting object will have metadata for 'mass_error'."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spectrum = dataset[i]
            # peptide sequence
            analyte = list(spectrum.analytes.values())[0]
            seq = ''.join(map(
                lambda x: x[0],
                analyte.peptide.sequence))
            # peak data
            mz, intensity, annotations, _ = zip(*spectrum.peak_list)
            # peak annotations
            def parse_annotation(annotation):
                if type(annotation) == PeptideFragmentIonAnnotation:
                    return (
                        [f"+{annotation.charge}"], 
                        [annotation.mass_error.mass_error], 
                        [annotation.series], 
                        [annotation.position], 
                        [tuple([''] if len(annotation.neutral_losses) == 0 else [loss.name for loss in annotation.neutral_losses])])
                else:
                    return (
                        ['+1'],
                        [0.],
                        [''],
                        [-1],
                        [tuple([''])])
            charge, mass_error, series, position, losses = zip(*map(
                lambda x: parse_annotation(x[0]),
                annotations))
            # done
            return cls(
                peptide = seq,
                mz = mz, 
                intensity = intensity,
                charge = list(charge),
                losses = list(losses),
                series = list(series),
                position = list(position),
                metadata = {"mass_error": mass_error})

    @staticmethod
    def parse_oms_stringdata(data: bytes):
        text = data.decode()
        series = text[0]
        loss_start = text.find('-')
        charge_start = text.find('+')
        if loss_start == -1:
            loss = ''
            pos_end = charge_start
        else:
            loss = text[loss_start + 1: charge_start]
            pos_end = loss_start
        charge = len(text) - charge_start
        position = text[1:pos_end]
        return charge, loss, series, position

    @classmethod
    def from_oms_spectrum(cls,
        peptide: str,
        spectrum: MSSpectrum,
    ) -> Self:
        mz, intensity = spectrum.get_peaks()
        charge, losses, series, position = zip(*[
            cls.parse_oms_stringdata(bytestr) for bytestr in spectrum.getStringDataArrays()[0]])
        return cls(
            peptide = peptide,
            mz = mz,
            intensity = intensity,
            charge = charge,
            losses = losses,
            series = series,
            position = position,
            metadata = {})

    @classmethod
    def from_simulation(cls,
        peptide: str,
        mode: str,
        max_charge: int,
    ) -> Self:
        if mode == "simple":
            return cls.from_oms_spectrum(
                peptide = peptide,
                spectrum = generate_fragment_spectrum(peptide, DEFAULT_PARAM, max_charge = max_charge))
        elif mode == "complex":
            return cls.from_oms_spectrum(
                peptide = peptide,
                spectrum = generate_fragment_spectrum(peptide, COMPLEX_PARAM, max_charge = max_charge))
        else:
            raise ValueError(f"unrecognized simulation mode: {mode}")
        
    def __repr__(self):
        annotation_repr = super(BenchmarkPeakList, self).__repr__()
        peptide_repr = f"peptide: {self.peptide}"
        return '\n'.join([annotation_repr, peptide_repr])
    
    def get_peptide(self):
        return self.peptide

    def get_series(self, i: int):
        return self.series[i]
    
    def get_position(self, i: int):
        return self.position[i]

    def get_state(self, i: int, metadata_keys = []) -> tuple:
        """The set of annotation vector(s) for the `i`th peak, including metadata if the `metadata_keys` kwarg is passed."""
        charge = self.get_charge(i)[0]
        losses = self.get_losses(i)[0]
        series = self.get_series(i)[0]
        position = self.get_position(i)[0]
        metadata = [self.get_metadata(i, k) for k in metadata_keys]
        return (charge, losses, series, position, *metadata)
    
    def _series_peak_indices(self, series: str):
        for i in range(len(self)):
            if self.get_series(i)[0] == series:
                yield i
    
    def get_b_series_peaks(self):
        return self.b_idx
    
    def get_y_series_peaks(self):
        return self.y_idx
