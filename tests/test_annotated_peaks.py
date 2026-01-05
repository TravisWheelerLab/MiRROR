# uv run ./tests/test_annotated_peaks.py ./data/spectra/Apis-mellifera.mzlib.txt 0.01 True 3
from sys import argv

import pytest

from mrror.io import read_mzlib
from mrror.evaluation.annotated_peaks import DEFAULT_PARAM, COMPLEX_PARAM, AnnotatedPeaks

import numpy as np
from tabulate import tabulate

TEST_PEPTIDES = [
    "PEPTIDE",
    "NREQSTK",
    "AEEHANR",
    "GNAGGLHHHR",
    "HHVLHHQTVDK",
    "HHSTIPQK",
    "FTHQHKPDER",
    "CEACPKPGTHAHK",
    "HHTIAHYK",
    "KPGVHQPQR",
    "AAHLAAHEAAK",
    "GHSCYRPR",
    "HHNIIR",
    "HLAEHEVK",
    "HGLTNTASHTR",
    "INPDNHNEK",
    "HGATVVNHVK",
    "HLNGHGSPPATNSSHR",
    "HASNIHVEK",
    "ELHVHPK",
]

def _assert_maxmin_tolerance(queries, targets, tolerance):
    max_min_err = -np.inf
    max_min_idx = -1
    max_min_tgt = -1
    queries = list(set(queries))
    target = list(set(targets))
    for (i,x) in enumerate(queries):
        min_err = np.inf
        min_tgt = -1
        for (j,y) in enumerate(targets):
            dif = abs(x - y)
            if dif < min_err:
                min_err = dif
                min_tgt = j
        if min_err > max_min_err:
            max_min_err = min_err
            max_min_idx = i
            max_min_tgt = min_tgt
    assert max_min_err < tolerance

def _assert_positive(arrs: list[list]):
    assert np.concat(arrs).min() > 0

def _test_decharged_fragment_masses(peptide, param, tolerance):
    peaks_simple = AnnotatedPeaks.from_simulation(peptide, param, 1)
    peaks_charged = AnnotatedPeaks.from_simulation(peptide, param, 3)
    pair_masses = peaks_simple.pair_masses()
    decharged_pair_masses = peaks_charged.decharged_pair_masses()
    _assert_maxmin_tolerance(decharged_pair_masses, pair_masses, tolerance)

    lower_boundary_masses = peaks_simple.lower_boundary_masses()
    decharged_lower_boundary_masses = peaks_simple.decharged_lower_boundary_masses()
    _assert_maxmin_tolerance(decharged_lower_boundary_masses, lower_boundary_masses, tolerance)

    upper_boundary_masses = peaks_simple.upper_boundary_masses()
    decharged_upper_boundary_masses = peaks_simple.decharged_upper_boundary_masses()
    _assert_maxmin_tolerance(decharged_upper_boundary_masses, upper_boundary_masses, tolerance)

    _assert_positive([pair_masses, decharged_pair_masses, lower_boundary_masses, decharged_lower_boundary_masses, upper_boundary_masses, decharged_upper_boundary_masses])

def test_decharged_fragment_masses():
    tolerance = 0.001
    for peptide in TEST_PEPTIDES:
        _test_decharged_fragment_masses(peptide, DEFAULT_PARAM, tolerance)
        # without losses
        _test_decharged_fragment_masses(peptide, COMPLEX_PARAM, tolerance)
        # with losses

def main(path_to_dataset, comp_tolerance, sim_losses, sim_charges, precision = 2):
    sim_param = COMPLEX_PARAM if sim_losses else DEFAULT_PARAM
    # construct params dependent on losses arg
    sim_param.setValue("add_first_prefix_ion", "true")
    # always add first b ion
    print(tabulate([
            ("benchmark",path_to_dataset,"Path, an mzSpecLib dataset."),
            ("tolerance",comp_tolerance,"Float, m/z comparison tolerance."),
            ("simulate losses",sim_losses,"Boolean, determines whether losses are simulated."),
            ("simulated charges",sim_charges,"Integer, max charge to simulate."),
        ], headers=("Argument","Value","Description")))
    print(f"\nreading dataset...")
    dataset = read_mzlib(path_to_dataset)
    n = len(dataset)
    record = -1
    c = 0
    sum_y_cov = 0.
    sum_b_cov = 0.
    sum_sym = 0
    while True:
        inp = input(f"\nenter an index between 0 and {n - 1}, or + to increment (next position = {record + 1}). enter x to exit.\n> ")
        if inp.isnumeric():
            old_rec = record
            record = int(inp)
            if record >= len(dataset):
                print(f"[!] input '{record}' is out of range.")
                record = old_rec
                continue
        elif inp == '+':
            record += 1
        elif inp == 'x':
            break
        else:
            print(f"[!] input '{inp}' is not a number.")
            continue
        # parse input.
    
        c += 1
        
        bench_peaks = AnnotatedPeaks.from_benchmark(dataset, record)
        # retrieve benchmark data.
    
        sim_peaks = AnnotatedPeaks.from_simulation(bench_peaks.peptide, sim_param, sim_charges)
        # simulate from peptide.
    
        comp_tolerance = 0.01
        comp = bench_peaks.compare(sim_peaks, comp_tolerance)
        # construct comparison.
    
        rows = []
        for ((i,j),(charge_i,charge_j),(loss_i,loss_j)) in zip(comp.alignment,comp.charge_aln,comp.loss_aln):
            rows.append((
                bench_peaks.mz[i],
                bench_peaks.series[i],
                bench_peaks.position[i],
                bench_peaks.loss[i][loss_i],
                bench_peaks.charge[i][charge_i],
                (int(i),int(j)),
                sim_peaks.mz[j],
                sim_peaks.series[j],
                sim_peaks.position[j],
                sim_peaks.loss[j][loss_j],
                sim_peaks.charge[j][charge_j],
            ))
        rows = sorted(rows, key=lambda x: (x[2],x[8]))
        headers = ["B m/z", "B series", "B pos", "B loss", "B charge", "Aln", "S m/z", "S series", "S pos", "S loss", "S charge"]
    
        def series_coverage(series):
            mask = sim_peaks.series == series
            # print(mask)
            sim_positions = sim_peaks.position[mask]
            # print(sim_positions)
            n_expected = len(set(sim_positions))
            # print(n_expected)
            observed_positions = [sim_peaks.position[j] for (_,j) in comp.alignment if sim_peaks.series[j] == series]
            # print(observed_positions)
            n_observed = len(set(observed_positions))
            return f"{round(n_observed / n_expected, precision)} ({n_observed} / {n_expected})"
        def count_symmetries():
            n_positions = sim_peaks.position.max()
            observed_y = set([sim_peaks.position[j] for (_,j) in comp.alignment if sim_peaks.series[j] == 'y'])
            refl_observed_b = set([n_positions - sim_peaks.position[j] + 1 for (_,j) in comp.alignment if sim_peaks.series[j] == 'b'])
            symmetric_y = observed_y.intersection(refl_observed_b)
            n_symmetric = len(symmetric_y)
            return f"{round(n_symmetric / n_positions, precision)} ({n_symmetric} / {n_positions}) y = {[int(x) for x in symmetric_y]}"
    
        y_coverage = series_coverage('y')
        b_coverage = series_coverage('b')
        symmetries = count_symmetries()
    
        sum_y_cov += float(y_coverage.split(' ')[0])
        sum_b_cov += float(b_coverage.split(' ')[0])
        sum_sym += float(symmetries.split(' ')[0])
    
        print(f"peptide: {bench_peaks.peptide}\npivot err: {bench_peaks.pivot} - {sim_peaks.pivot} = {comp.pivot_err}")
        print(tabulate(rows, headers=headers))
        print(f"y coverage: {y_coverage}\nb coverage: {b_coverage}\nsymmetric: {symmetries}")
        print(f"avg y cov: {sum_y_cov / c}\navg b cov: {sum_b_cov / c}\navg sym: {sum_sym / c}")
        # print comparison.

if __name__ == "__main__":
    main(
        path_to_dataset = argv[1],
        comp_tolerance = float(argv[2]),
        sim_losses = argv[3].lower() == 'true',
        sim_charges = int(argv[4]),
    )
