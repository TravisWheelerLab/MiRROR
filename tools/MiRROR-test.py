from _tool_init import mirror, timed_op, argparse
import numpy as np

PROG = "𝕄iℝℝ𝕆ℝ-test"
DESC = """the test system for MiRROR;
reads sequences from a .fasta.
enumerates the tryptic peptides of those sequences.
generates synthetic spectra for each peptide.
applies the integrated MiRROR pipeline to the spectra;
compares result(s) to the groundtruth tryptic peptide."""
ARG_NAMES = [
    "fasta_path",
    "alphabet_path",
    "suffix_array_path",
    "gap_tolerance",
    "intergap_tolerance",
    "symmetry_threshold",
    "occurrence_threshold",
    "alignment_threshold",
    "alignment_parameters",
    "terminal_residues",
    "gap_key"
]
ARG_TYPES = [
    str,
    str,
    str,
    float,
    float,
    float,
    int,
    float,
    str,
    list[str],
    str,
]
ARG_DEFAULTS = [
    None,
    None,
    None,
    mirror.util.GAP_TOLERANCE,
    mirror.util.INTERGAP_TOLERANCE,
    None,
    1,
    -1,
    "",
    mirror.util.TERMINAL_RESIDUES,
    mirror.spectrum_graphs.GAP_KEY,
]

def get_parser():
    parser = argparse.ArgumentParser(
        prog = PROG,
        description = DESC
    )
    for (argname, argtype, argdefault) in zip(ARG_NAMES, ARG_TYPES, ARG_DEFAULTS):
        parser.add_argument(f"--{argname}", type = argtype, default = argdefault)
    return parser

def main(args):
    # load sequences, gap alphabet
    sequences = mirror.io.load_fasta_as_strings(args.fasta_path)
    target_groups, residues = mirror.io.load_target_groups(args.alphabet_path)
    target_space = mirror.TargetSpace(target_groups, residues, args.gap_tolerance)

    # lazy iterators
    tryptic_peptides = mirror.util.enumerate_tryptic_peptides(sequences)
    peak_arrays = map(mirror.util.simulate_peaks, tryptic_peptides)

    # pipeline
    for (true_sequence, peaks) in zip(tryptic_peptides, peak_arrays):
        #print(f"peptide: {true_sequence}")
        # find gaps
        gap_results = target_space.find_gaps(peaks)
        #print(f"gaps: {sum(len(r) for r in gap_results)}")

        # find pivots
        symmetry_threshold = args.symmetry_threshold
        if symmetry_threshold == None:
            symmetry_threshold = mirror.util.expected_num_mirror_symmetries(peaks)
        
        pivots = mirror.collapse_second_order_list([
            mirror.find_pivots(
                peaks, 
                symmetry_threshold, 
                r.index_tuples(), 
                args.intergap_tolerance) 
            for r in gap_results])
        #print(f"pivots: {len(pivots)}")

        # enumerate boundary conditions
        boundaries = (mirror.find_boundary_peaks(peaks, p, args.terminal_residues) for p in pivots)

        for (pivot, (b_boundaries, y_boundaries)) in zip(pivots, boundaries):
            # todo - replace this with a TargetSpace method.
            pivot_residue = mirror.util.residue_lookup(pivot.gap())
            #print(f"boundaries: {len(b_boundaries), len(y_boundaries)}")
            for ((b_bound, b_res), (y_bound, y_res)) in zip(b_boundaries, y_boundaries):
                # create augmented peaks, pivots, gaps
                augmented_peaks, augmented_pivot = mirror.create_augmented_spectrum(
                    peaks, 
                    pivot,
                    b_bound,
                    y_bound,
                    args.gap_tolerance)
                augmented_gaps = [r.index_tuples() for r in target_space.find_gaps(augmented_peaks)]
                augmented_gaps = mirror.collapse_second_order_list(augmented_gaps)

                # create topologies
                graph_pair = mirror.create_spectrum_graph_pair(
                    augmented_peaks, 
                    augmented_gaps, 
                    augmented_pivot,
                    gap_key = args.gap_key)
                # find dual paths
                gap_comparator = lambda x, y: (abs(x - y) < args.intergap_tolerance) and (x != -1) and (y != -1)
                dual_paths = mirror.find_dual_paths(
                    *graph_pair,
                    args.gap_key,
                    gap_comparator)

                # find viable path pairings
                pair_indices = mirror.find_edge_disjoint_dual_path_pairs(dual_paths)

                # create affixes
                affixes = np.array([mirror.create_affix(dp, graph_pair) for dp in dual_paths])
                #print(f"affix candidates: {len(affixes)}")
                affixes = mirror.filter_affixes(
                    affixes, 
                    args.suffix_array_path, 
                    occurrence_threshold = args.occurrence_threshold)
                #print(f"affixes: {len(affixes)}")

                # create candidates
                for (i, j) in pair_indices:
                    candidates = np.array(mirror.create_candidates(
                        augmented_peaks, 
                        graph_pair, 
                        affixes[[i,j]],
                        (b_res, y_res),
                        pivot_residue))
                    candidates = mirror.filter_candidate_sequences(
                        peaks,
                        candidates,
                        args.alignment_threshold,
                        args.alignment_parameters)
                    for candidate in candidates:
                        print(candidate, candidate.edit_distance(true_sequence))

if __name__ == "__main__":
    args = get_parser().parse_args()
    for (k,v) in vars(args).items():
        print(f"{k} : {v}")
    main(args)