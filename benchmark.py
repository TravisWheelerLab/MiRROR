import multiprocessing, glob, pathlib
import itertools as it
from random import shuffle

import pprint
import tqdm
import tabulate
import numpy as np
import matplotlib.pyplot as plt

from mirror.io import read_mgf
from mirror.util import load_config
from mirror.spectra.types import Peaks
from mirror.sequences.suffix_array import SuffixArray
from mirror.annotation import AnnotationParams, AnnotationResult, annotate
from mirror.alignment import AlignmentParams, AlignmentResult, align
from mirror.enumeration import EnumerationParams, EnumerationResult, enumerate_candidates

CONFIG_DIR = "/home/user/Projects/MiRROR/params"
CONFIG_NAME = "config"
CONFIG = load_config(CONFIG_DIR, CONFIG_NAME)

THREADS = 8

ANNOTATION_PARAMS = AnnotationParams.from_config(CONFIG.annotation)
def run_annotation(peaks: Peaks):
    return annotate(peaks, ANNOTATION_PARAMS)

ALIGNMENT_PARAMS = AlignmentParams.from_config(CONFIG.alignment)
def run_align(annotation: AnnotationResult):
    return align(annotation, ALIGNMENT_PARAMS)

ENUMERATION_PARAMS = EnumerationParams.from_config(CONFIG.enumeration)
def run_enumeration(
    annotation: AnnotationResult,
    alignment: AlignmentResult,
    suffix_arrays: tuple[SuffixArray,SuffixArray],
):
    return enumerate_candidates(annotation, alignment, suffix_arrays, ENUMERATION_PARAMS)

def sequence(peaks, N, pool):
    print("(annotation)")
    annotations = list(pool.imap(
        run_annotation,
        tqdm.tqdm(peaks,total=N),
    ))
    print("(alignment)")
    alignments = list(pool.imap(
        run_align,
        tqdm.tqdm(annotations,total=N),
    ))
    print("(enumeration)")
    candidates = list(pool.imap(
        run_enumeration,
        tqdm.tqdm(alignments,total=N),
    ))
    return (
        annotations,
        alignments,
        candidates,
    )

def visualize(input_dir: str, output_dir: str):
    input_pattern = str(pathlib.Path(input_dir) / "*.mgf")
    nine_species_mgf_paths = glob.glob(input_pattern)
    nine_species_names = [pathlib.Path(x).stem for x in nine_species_mgf_paths]
    nine_species_benchmark = {
        species_name: read_mgf(path) 
        for (species_name, path) in tqdm.tqdm(
            zip(nine_species_names, nine_species_mgf_paths),
            total=9,
        )
    }
    # load nine-species for data visualization
    
    print(tabulate.tabulate(
        [
            [species_name, len(spectra), str(type(spectra))]
            for (species_name, spectra) in sorted(
                nine_species_benchmark.items(),
                key=lambda x: len(x[1]),
            )
        ],
        headers=["Species", "# Spectra", "Type"],
    ))
    # print summary table
    
    spectrum_lengths = [
        len(Peaks.from_mgf(x)) for (name,_) in sorted(nine_species_benchmark.items(), key=lambda x: len(x[1]))
        for x in tqdm.tqdm(nine_species_benchmark[name])
    ]
    cumulative_offsets = np.cumsum([0,] + [len(spectra) for spectra in nine_species_benchmark.values()])
    per_species_spectrum_lengths = [spectrum_lengths[i:j] for (i,j) in it.pairwise(cumulative_offsets)]
    # collect spectrum length distributions
    
    fig, ax = plt.subplots()
    ax.set_title("Spectrum Size Distribution in Log Scale")
    ax.set_xlabel("Number of Peaks")
    ax.set_ylabel("Number of Spectra")
    ax.hist(spectrum_lengths,bins=200,log=True)
    fig.set_tight_layout(True)
    plt.savefig(str(pathlib.Path(output_dir) / "spectrum_size_hist_log.png"))
    # plot aggregate distribution
    
    fig, axes = plt.subplots(3,3)
    for (idx, name) in enumerate(nine_species_benchmark):
        i = idx % 3
        j = idx // 3
        ax = axes[i,j]
        abbreviated_name = name[0] + '. ' + name.split('-')[-1]
        ax.set_title(abbreviated_name)
        if i == 1 and j == 0:
            ax.set_ylabel("Number of Spectra")
        if i == 2 and j == 1:
            ax.set_xlabel("Number of Peaks")
        ax.hist(per_species_spectrum_lengths[idx], bins=200, log=True)
    fig.set_tight_layout(True)
    plt.savefig(str(pathlib.Path(output_dir) / "spectrum_size_hist_log_per_species.png"))
    # plot individual distributions

def run_benchmark(input_dir: str, output_dir: str, species: list[str], samples: list[int]):
    print("  reading benchmark.")
    input_pattern = str(pathlib.Path(input_dir) / "*.mgf")
    nine_species_mgf_paths = glob.glob(input_pattern)
    nine_species_names = [pathlib.Path(x).stem for x in nine_species_mgf_paths]
    nine_species_benchmark = {
        species_name: read_mgf(path) 
        for (species_name, path) in tqdm.tqdm(
            zip(nine_species_names, nine_species_mgf_paths),
            total=9,
        )
    }
    pool = multiprocessing.Pool(THREADS)
    for (name,n) in zip(species, samples):
        if name in nine_species_benchmark:
            print(f"  loading {name}.")
            mgf_spectra = list(nine_species_benchmark[name])
            peaks = [Peaks.from_mgf(spectrum) for spectrum in mgf_spectra[:n]]
            print(f"  sequencing {n} spcetra.")
            sequence(peaks, n, pool)
        else:
            raise ValueError(f"species {name} was not found in the nine-species benchmark: {nine_species_names}.")

if __name__ == "__main__":
    print("generating visualizations...")
    visualize(input_dir="./data/spectra",output_dir="./data/benchmark")
    print("processing spectra...")
    run_benchmark(
        input_dir="./data/spectra",
        output_dir="./data/benchmark",
        species=["Mus-musculus",],
        samples=[10,],
    )
