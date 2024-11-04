import multiprocessing
from math import ceil
import numpy as np

from .util import add_tqdm
from .io import oms

def accumulate_bins(data: list[float], hist_arr: np.array, n_bins: int, max_val: int):
    hist_arr += np.histogram(data, bins = n_bins, range = (0, max_val))[0]

def generate_bins(data_iterable, n_bins: int, max_val: int):
    #for idx in add_tqdm(idx_range):
    hist_arr = np.zeros(n_bins)
    for data in add_tqdm(data_iterable):
        hist_arr += np.histogram(data, bins = n_bins, range = (0, max_val))[0]
        #accumulate_bins(
        #    data,
        #    hist_arr,
        #    n_bins,
        #    max_val
        #)
    return hist_arr

def create_bins(exp: oms.MSExperiment, max_mz: int, resolution: float, parallelize = False, n_processes = 4):
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
                print('\t', (lo,hi), sum(b == 0), '/', len(b))
                spectrum_bins += b
    else:
        for spectrum_idx in add_tqdm(range(n_spectra)):
            accumulate_bins(
                exp.getSpectrum(spectrum_idx).get_peaks()[0],
                spectrum_bins,
                n_bins,
                max_mz)
    print(sum(spectrum_bins == 0), '/', len(spectrum_bins))
    return spectrum_bins