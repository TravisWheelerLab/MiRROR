import multiprocessing
import statistics
from time import time
from random import shuffle

import numpy as np

from util import *
from search import *

def load_fasta_records_as_str(path_to_fasta):
    fasta_records = load_fasta_records(path_to_fasta)
    return [str(record.seq) for record in fasta_records]

def generate_spectrum_and_list_mz(seq):
    return list_mz(generate_default_fragment_spectrum(seq))

def locate_pivot_point(spectrum, tolerance):
    pivots_overlap = search(spectrum, "overlap", AMINO_MASS_MONO, tolerance)
    if len(pivots_overlap) == 0:
        pivots_disjoint = search(spectrum, "disjoint", AMINO_MASS_MONO, tolerance)
        pivot_centers = [np.mean(pivot.data) for pivot in pivots_disjoint]
        return statistics.mode(pivot_centers)
    else:    
        pivot_centers = [np.mean(pivot.data) for pivot in pivots_overlap]
        pivot_scores = [measure_mirror_symmetry(spectrum, center) for center in pivot_centers]
        return pivot_centers[np.argmax(pivot_scores)]

def check_pivot(pivot, peptide, spectrum):
    b = get_b_ion_series(peptide)
    y = get_y_ion_series(peptide)
    true_pivot = [*b[0:2],*y[-3:-1]]
    true_pivot_location = np.mean(true_pivot)
    return abs(true_pivot_location - pivot) < 0.01

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
        tryptic_peptides = pool.map(digest_trypsin, add_tqdm(fasta_sequences))
        tryptic_peptides = collapse_second_order_list(tryptic_peptides)
        tryptic_peptides = list(filter(lambda seq: 'X' not in seq, tryptic_peptides))
        shuffle(tryptic_peptides)
        n_pep = len(tryptic_peptides)
        elapsed_t = time() - init_t
        print(f"separated {n_pep} peptides in {elapsed_t} seconds.")

        print("⚙\tgenerating fragment spectra...")
        init_t = time()
        spectra = pool.map(generate_spectrum_and_list_mz, add_tqdm(tryptic_peptides))
        n_spec = len(spectra)
        elapsed_t = time() - init_t
        print(f"synthesized {n_spec} spectra in {elapsed_t} seconds.")

        print("⚙\tlocating pivot points...")
        init_t = time()
        spectra_with_tolerances = list(zip(spectra, [tolerance] * n_spec))
        pivots = pool.starmap(locate_pivot_point, add_tqdm(spectra_with_tolerances, description="processing"))
        pivots_peptides_spectra = list(zip(pivots, tryptic_peptides, spectra))
        num_successes = sum(pool.starmap(check_pivot, add_tqdm(pivots_peptides_spectra, description="validating")))
        elapsed_t = time() - init_t
        percent_success = round(100*num_successes/len(spectra), 3)
        print(f"solved {num_successes} / {len(spectra)} ({percent_success}%) of pivot point locations in {elapsed_t} seconds.")
    else:
        # todo: load spectrum from .mzML and .mzMLb
        print(f"unsupported mode `{mode}`")

if __name__ == "__main__": from sys import argv; main(argv)