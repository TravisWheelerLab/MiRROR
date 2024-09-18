import statistics as stat
import numpy as np
import multiprocessing
from time import time

from util import *
from search import *

def load_fasta_records_as_str(path_to_fasta):
    fasta_records = load_fasta_records(path_to_fasta)
    return [str(record.seq) for record in fasta_records]

def generate_spectrum_and_list_mz(seq):
    return list_mz(generate_default_fragment_spectrum(seq))

def locate_pivot_point(spectrum, tolerance):
    pivots_overlap, pivots_disjoint = search(spectrum, tolerance, AMINO_MASS_MONO)
    return pivots_overlap
    # todo: check overlap
    # if overlap fails with outputs, try to recover nested structure.
    # if nested structure is not present, or overlap has no outputs, 
    # use the mode of disjoint.

def check_pivot(pivot, spectrum):
    center = np.mean(pivot)
    n_sym, n_tot = measure_mirror_symmetry(spectrum, center)
    return n_sym / n_tot > 0.9

def main(argv):
    mode = argv[1]
    if mode == "test":
        path_fasta = argv[2]
        tolerance = float(argv[3]) if len(argv) == 4 else 2 * AVERAGE_MASS_DIFFERENCE
        pool = multiprocessing.Pool()

        print(f"⚙\tloading fasta records from `{path_fasta}`...")
        init_t = time()
        fasta_sequences = load_fasta_records_as_str(path_fasta)
        n_seq = len(fasta_sequences)
        elapsed_t = time() - init_t
        print(f"read {n_seq} sequences in {elapsed_t} seconds.")

        print("⚙\tsimulating trypsin digest...")
        init_t = time()
        tryptic_peptides = pool.map(digest_trypsin, fasta_sequences)
        tryptic_peptides = collapse_second_order_list(tryptic_peptides)
        n_pep = len(tryptic_peptides)
        elapsed_t = time() - init_t
        print(f"separated {n_pep} peptides in {elapsed_t} seconds.")

        print("⚙\tgenerating fragment spectra...")
        init_t = time()
        viable_peptides = filter(lambda seq: 'X' not in seq, tryptic_peptides)
        spectra = pool.map(generate_spectrum_and_list_mz, viable_peptides)
        n_spec = len(spectra)
        elapsed_t = time() - init_t
        print(f"synthesized {n_spec} spectra in {elapsed_t} seconds.")

        print("⚙\tlocating pivot points...")
        init_t = time()
        tolerances = [tolerance] * n_spec
        pivots = pool.starmap(locate_pivot_point, zip(spectra, tolerances))
        num_successes = sum(pool.starmap(check_pivot, zip(pivots, spectra)))
        elapsed_t = time() - init_t
        print(f"solved {num_successes} / {len(spectra)} ({100*num_successes/len(spectra)}%) of pivot point locations in {elapsed_t} seconds.")
        
    else:
        # todo: load spectrum from .mzML and .mzMLb
        print(f"unsupported mode `{mode}`")

if __name__ == "__main__": from sys import argv; main(argv)