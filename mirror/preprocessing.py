import multiprocessing
from math import ceil

import numpy as np
import mzspeclib as mzlib
import pyopenms as oms

from .types import Iterator, Any
from .util import add_tqdm

#=============================================================================#
# preprocessing for mzSpecLib spectra

class MzSpecLib:
    """A wrapper for an mzspeclib.SpectrumLibrary object. Implements getters for length, peaks (i.e., spectra), and annotated peptides (i.e., analytes)"""

    def __init__(self, lib: mzlib.SpectrumLibrary):
        self._spectrum_library = lib
    
    def __len__(self):
        """The number of spectra in the record.

            len(self._spectrum_library)"""
        return len(self._spectrum_library)
    
    def _get_spectrum(self,
        idx: int
    ) -> mzlib.Spectrum:
        if idx >= len(self):
            raise ValueError(f"{idx} exceeds library size {len(self)}")
        return self._spectrum_library.get_spectrum(idx)

    def get_peaks(self,
        idx: int
    ) -> tuple[np.ndarray, np.ndarray, list, list]:
        """Return the intensity, peaks, and metadata as a 4-tuple given the index of the spectrum."""
        peaks = self._get_spectrum(idx).peak_list
        mz, intensity, metadata, etc = zip(*peaks)
        return np.array(list(intensity)), np.array(list(mz)), list(metadata), list(etc)

    def _get_peptides(self,
        idx: int
    ) -> Iterator[list[tuple[str, Any]]]:
        for analyte in self._get_spectrum(idx).analytes.values():
            yield analyte.peptide.sequence
    
    def get_peptides(self,
        idx: int
    ) -> list[str]:
        """Return the peptides (analytes) as str types, given the index of the spectrum."""
        return [''.join(res[0] for res in pep_list) for pep_list in self._get_peptides(idx)]

#=============================================================================#
# preprocessing for oms.MSExperiment interface to mzML spectra

def accumulate_bins(data: list[float], hist_arr: np.array, n_bins: int, max_val: int):
    hist_arr += np.histogram(data, bins = n_bins, range = (0, max_val))[0]

def generate_bins(data_iterable, n_bins: int, max_val: int):
    #for idx in add_tqdm(idx_range):
    hist_arr = np.zeros(n_bins)
    for data in add_tqdm(data_iterable):
        hist_arr += np.histogram(data, bins = n_bins, range = (0, max_val))[0]
    return hist_arr

def create_spectrum_bins(
    exp: oms.MSExperiment, 
    max_mz: int, 
    resolution: float, 
    parallelize = False, 
    n_processes = 4, 
    verbose = False
) -> np.ndarray:
    """Bin the peaks of a pyopenms.MSExperiment object.
    
    :exp: a pyopenms.MSExperiment object.
    :max_mz: the maximum allowed mz value.
    :resolution: the number of bins per mz. max_mz / resolution = n_bins.
    :parallelize: boolean flag for parallelization. Defaults to false.
    :n_processes: int value specifying the max number of processes to parallelize over.
    :verbose: boolean flag. If true, bins will be summarized to standard out."""
    n_bins = int(max_mz / resolution)
    n_spectra = exp.getNrSpectra()
    spectrum_bins = np.zeros(n_bins)
    if parallelize:
        with multiprocessing.Pool() as pool:
            n_elts_per_process = ceil(n_spectra / n_processes)
            range_iterable = (range(i * n_elts_per_process, min(n_spectra - 1, (i + 1) * n_elts_per_process)) for i in range(n_processes))
            data_iterable = ([exp.getSpectrum(spectrum_idx).get_peaks()[0] for spectrum_idx in indices] for indices in range_iterable)
            nbin_iterable = (n_bins for _ in range(n_processes))
            maxmz_iterable = (max_mz for _ in range(n_processes))
            bins_by_process = pool.starmap(
                generate_bins,
                zip(data_iterable, nbin_iterable, maxmz_iterable),
            )
            range_iterable = ((i * n_elts_per_process, min(n_spectra - 1, (i + 1) * n_elts_per_process)) for i in range(n_processes))
            for ((lo,hi),b) in zip(range_iterable,bins_by_process):
                if verbose:
                    print('\t', (lo,hi), sum(b == 0), '/', len(b))
                spectrum_bins += b
    else:
        for spectrum_idx in add_tqdm(range(n_spectra)):
            accumulate_bins(
                exp.getSpectrum(spectrum_idx).get_peaks()[0],
                spectrum_bins,
                n_bins,
                max_mz)
    if verbose:
        print(sum(spectrum_bins == 0), '/', len(spectrum_bins))
    return spectrum_bins

# reducing raw bin data into a set of peaks

def filter_spectrum_bins(
    spectrum_bins: np.ndarray,
    max_mz: int,
    resolution: float,
    binned_frequency_threshold: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Associates the output of create_spectrum_bins to arrays of frequencies (≈ intensities) and peak mz values.
    
    :spectrum_bins: np.ndarray, the output of create_spectrum_bins.
    :max_mz: int, maximum allowed mz value.
    :resolution: float, the number of bins per mz. max_mz / resolution = n_bins.
    :binned_frequency_threshold: the threshold, above which bins are counted as peaks."""
    bin_mask = spectrum_bins > binned_frequency_threshold
    n_reduced_peaks = sum(bin_mask)
    reduced_frequencies = spectrum_bins[bin_mask]
    reduced_mz = np.arange(np.ceil(max_mz / resolution, int))[bin_mask] * resolution
    return reduced_frequencies, reduced_mz

#=============================================================================#
# clustering for PTMs - optional, but without this step 
# the candidate set may be too large to enumerate.