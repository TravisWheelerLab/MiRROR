import multiprocessing
import statistics
import os
from time import time
from random import shuffle

import numpy as np

from util import *

from search import search as gap_driven_search
from agnostic_search import search as agnostic_search

def load_fasta_records_as_str(path_to_fasta):
    fasta_records = load_fasta_records(path_to_fasta)
    return [str(record.seq) for record in fasta_records]

def generate_spectrum_and_list_mz(seq):
    return list_mz(generate_default_fragment_spectrum(seq))

def locate_pivot_point(spectrum, tolerance, searchmode):
    if searchmode == "gap-driven":
        search_fn = gap_driven_search
    elif searchmode == "gap-agnostic":
        search_fn = agnostic_search
    pivots_overlap = search_fn(spectrum, "overlap", AMINO_MASS_MONO, tolerance)
    if len(pivots_overlap) == 0:
        pivots_disjoint = search_fn(spectrum, "disjoint", AMINO_MASS_MONO, tolerance)
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

def run_validation_test(argv, searchmode):
    print(f"search mode: {searchmode}")
    path_fasta = argv[2]
    tolerance = float(argv[3]) if len(argv) == 4 else 2 * AVERAGE_MASS_DIFFERENCE
    pool = multiprocessing.Pool()

    print(f"⚙\tloading fasta records from `{path_fasta}`...")
    init_t = time()
    fasta_sequences = load_fasta_records_as_str(path_fasta)
    num_seq = len(fasta_sequences)
    elapsed_t = time() - init_t
    print(f"read {num_seq} sequences in {elapsed_t} seconds.")

    print("⚙\tsimulating trypsin digest...")
    init_t = time()
    tryptic_peptides = pool.map(digest_trypsin, add_tqdm(fasta_sequences))
    tryptic_peptides = collapse_second_order_list(tryptic_peptides)
    tryptic_peptides = list(filter(lambda seq: 'X' not in seq, tryptic_peptides))
    shuffle(tryptic_peptides)
    num_pep = len(tryptic_peptides)
    elapsed_t = time() - init_t
    print(f"separated {num_pep} peptides in {elapsed_t} seconds.")

    print("⚙\tgenerating fragment spectra...")
    init_t = time()
    spectra = pool.map(generate_spectrum_and_list_mz, add_tqdm(tryptic_peptides))
    num_spec = len(spectra)
    elapsed_t = time() - init_t
    print(f"synthesized {num_spec} spectra in {elapsed_t} seconds.")

    print("⚙\tlocating pivot points...")
    init_t = time()
    spectra_tolerances_searchmode = list(zip(spectra, [tolerance] * num_spec, [searchmode] * num_spec))
    pivots = pool.starmap(locate_pivot_point, add_tqdm(spectra_tolerances_searchmode, description="processing"))
    pivots_peptides_spectra = list(zip(pivots, tryptic_peptides, spectra))
    validation = pool.starmap(check_pivot, add_tqdm(pivots_peptides_spectra, description="validating"))
    num_successes = sum(validation)
    elapsed_t = time() - init_t
    percent_success = round(100*num_successes/len(spectra), 3)
    print(f"solved {num_successes} / {num_spec} ({percent_success}%) of pivot point locations in {elapsed_t} seconds.")

    if num_successes != num_spec:
        print("⚙\trecording missed peptides...")
        init_t = time()
        # get missed peptides
        unsolved_peptides = [x[1] for x in zip(validation,tryptic_peptides) if not(x[0])]
        num_misses = len(unsolved_peptides)
        assert num_misses == num_spec - num_successes 
        # record
        input_fasta_suffix = os.path.basename(path_fasta)
        misses_dir = "./misses/"
        if not(os.path.exists(misses_dir)):
            os.mkdir(misses_dir)
        misses_fasta = f"{misses_dir}misses_{searchmode}_{input_fasta_suffix}"
        num_written = save_strings_to_fasta(misses_fasta, unsolved_peptides)
        assert num_written == num_misses
        elapsed_t = time() - init_t
        print(f"wrote {num_written} peptides to '{misses_fasta}' in {elapsed_t} seconds.")

def main(argv):
    mode = argv[1]
    if mode == "test":
        run_validation_test(argv, "gap-driven")
    elif mode == "test-agnostic":
        run_validation_test(argv, "gap-agnostic")
    else:
        # todo: load spectrum from .mzML and .mzMLb
        print(f"unsupported mode `{mode}`")

if __name__ == "__main__": from sys import argv; main(argv)