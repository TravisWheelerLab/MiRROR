from abc import ABC, abstractclassmethod
from typing import Any, Iterable
from dataclasses import dataclass
import warnings

import numpy as np
import pyopenms as oms
from mzpaf import PeptideFragmentIonAnnotation, Unannotated
from mzspeclib.validate import ValidationWarning as mzlib_ValidationWarning

from ..io import mgf, read_mgf, mzlib, read_mzlib

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
        spectrum_array: np.ndarray,
        resolution: float,
        min_mz: float,
        max_mz: float,
    ):
        self.data = spectrum_array
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
        self._n = n
        self.mz = mz
        self.intensity = intensity

    @classmethod
    def from_mzlib(cls, 
        dataset: mzlib.SpectrumLibrary,
        i: int,
    ):
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
    ):
        """Given an MGF `dataset` and an index `i`,
        retrieve the `i`th spectrum and use it to construct a PeakList."""
        spectrum = dataset[i]
        mz = spectrum['m/z array']
        intensity = spectrum['intensity array']
        return cls(
            mz = mz, 
            intensity = intensity)

    def __len__(self) -> int:
        return self._n
    
    def __getitem__(self, i: int) -> float:
        return self.mz[i]
    
    def __iter__(self) -> Iterable:
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
    Constructed via from_mzlib, and from_mgf class methods 
    or the simulate_peaks_from_peptide function of spectra.simulation."""

    def __init__(self,
        peptide: str,
        *args,
        **kwargs,
    ):
        self.peptide = peptide
        super(BenchmarkPeakList, self).__init__(*args, **kwargs)
    
    def __repr__(self):
        annotation_repr = super(BenchmarkPeakList, self).__repr__()
        peptide_repr = f"peptide: {self.peptide}"
        return '\n'.join([annotation_repr, peptide_repr])
    
    def get_peptide(self):
        return self.peptide

    @classmethod
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

class NineSpeciesBenchmarkPeakList(BenchmarkPeakList):
    """A benchmark peak list with peptide as well as 
    series and position annotations on peaks. Can be 
    constructed from the mzlib files of the nine-
    -species benchmark."""

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
        self.series = series
        self.position = position
        super(BenchmarkPeakList, self).__init__(mz, intensity, charge, losses, *args, **kwargs)
    
    def get_series(self, i: int):
        return self.series[i]
    
    def get_position(self, i: int):
        return self.position[i]

    def get_states(self, i: int, metadata_keys = []) -> set:
        """The set of annotation vector(s) for the `i`th peak, including metadata if the `metadata_keys` kwarg is passed."""
        charge = self.get_charge(i)[0]
        losses = self.get_losses(i)[0]
        series = self.get_series(i)[0]
        position = self.get_position(i)[0]
        metadata = [self.get_metadata(i, k) for k in metadata_keys]
        return set([(charge, losses, series, position, *metadata)])
    
    def get_series_peaks(self, series: str):
        for i in range(len(self)):
            if self.get_series(i)[0] == series:
                yield i
    
    def get_b_series_peaks(self):
        return list(self.get_series_peaks('b'))
    
    def get_y_series_peaks(self):
        return list(self.get_series_peaks('y'))

    @classmethod
    def from_mzlib(cls, 
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