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
    a: float
    b: float
    # peak detection
    x: float
    y: float
    # peak annotation
    z: float
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
        self._mz = mz
        self._intensity = intensity

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

    def __len__(self):
        return self._n
    
    def __getitem__(self, i: int):
        return self._mz[i]
    
    def get_intensity(self, i: int):
        """Return the intensity of the `i`th peak."""
        return self._intensity[i]

class AnnotatedPeakList(PeakList):
    """A PeakList with additional data associated to peaks.
    Each annotation describes charge state, losses, and post-translation modification.
    Each peak may have multiple annotations. Peaks with multiple matched associations."""

    def __init__(self,
        mz: np.ndarray, 
        intensity: np.ndarray,
        charge: np.ndarray,
        metadata: dict,
    ):
        n = len(mz)
        assert len(intensity) == len(charge) == n
        self._charge = charge
        self._metadata = metadata
        super(AnnotatedPeakList, self).__init__(mz, intensity)
    
    def get_charge(self, i: int):
        """Return the charge state of the `i`th peak."""
        return self._charge[i]
    
    def get_metadata(self, i: int, key: str):
        """Return the metadata for `key` of the `i`th peak."""
        return self._metadata[key][i]

class BenchmarkPeakList(AnnotatedPeakList):
    """An annotated peak list in the format of the 9-species benchmark.
    Constructed via from_mzlib, and from_mgf class methods 
    or the simulate_peaks_from_peptide function of spectra.simulation."""

    def __init__(self,
        peptide: str,
        *args,
        **kwargs,
    ):
        self._peptide = peptide
        super(BenchmarkPeakList, self).__init__(*args, **kwargs)
    
    def get_peptide(self):
        return self._peptide

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
                        annotation.charge, 
                        annotation.mass_error, 
                        annotation.series, 
                        annotation.position, 
                        annotation.neutral_losses)
                else:
                    return (
                        None,
                        None,
                        None,
                        None,
                        None)

            charge, mass_error, series, position, losses = zip(*map(
                lambda x: parse_annotation(x[0]),
                annotations))
            # done
            return cls(
                peptide = seq,
                mz = mz, 
                intensity = intensity,
                charge = charge,
                metadata = {
                    "mass_error": mass_error,
                    "series": series,
                    "position": position,
                    "losses": losses})
        
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
        intensity = spectrum['intensity array']
        # peak annotations
        charge = spectrum['charge array']
        # done
        return cls(
            peptide = seq,
            mz = mz, 
            intensity = intensity,
            charge = charge,
            metadata = {})
