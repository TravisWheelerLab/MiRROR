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

def locate_pivot_point(spectra, tolerance = 2 * AVERAGE_MASSWISE_DIFFERENCE):
    pivots_overlap, pivots_disjoint = search(spectra, tolerance, AMINO_MASS_MONO)
    # todo: check overlap
    # if overlap fails with outputs, try to recover nested structure.
    # if nested structure is not present, or overlap has no outputs, 
    # use the mode of disjoint.
    return pivots_overlap

def check_pivot(pivot):
    # todo
    return False

def main(argv):
    mode = argv[1]
    if mode == "test":
        path_fasta = argv[2]
        pool = multiprocessing.Pool()

        print(f"⚙\tloading fasta records from `{path_fasta}`...")
        init_t = time()
        fasta_sequences = load_fasta_records_as_str(path_fasta)
        elapsed_t = time() - init_t
        print(f"read {len(fasta_sequences)} sequences in {elapsed_t} seconds.")

        print("⚙\tsimulating trypsin digest...")
        init_t = time()
        tryptic_peptides = pool.map(digest_trypsin, fasta_sequences)
        tryptic_peptides = collapse_second_order_list(tryptic_peptides)
        elapsed_t = time() - init_t
        print(f"separated {len(tryptic_peptides)} peptides in {elapsed_t} seconds.")

        print("⚙\tgenerating fragment spectra...")
        init_t = time()
        viable_peptides = filter(lambda seq: 'X' not in seq, tryptic_peptides)
        spectra = pool.map(generate_spectrum_and_list_mz, viable_peptides)
        elapsed_t = time() - init_t
        print(f"synthesized {len(spectra)} spectra in {elapsed_t} seconds.")

        print("⚙\tlocating pivot points...")
        init_t = time()
        pivots = pool.map(locate_pivot_point, spectra)
        num_successes = sum(map(check_pivot,pivots))
        elapsed_t = time() - init_t
        print(f"solved {num_successes} / {len(spectra)} ({100*num_successes/len(spectra)}%) of pivot point locations in {elapsed_t} seconds.")
        
    else:
        # todo: load spectrum
        print(f"unsupported mode `{mode}`")

if __name__ == "__main__": from sys import argv; main(argv)