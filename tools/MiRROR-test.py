from _tool_init import mirror, timed_op, argparse
import numpy as np

PROG = "𝕄 iℝ ℝ 𝕆 ℝ - test\n"
DESC = """the test system for MiRROR;
reads sequences from a .fasta.
enumerates the tryptic peptides of those sequences.
generates synthetic spectra for each peptide.
applies the integrated MiRROR pipeline to the spectra;
compares result(s) to the groundtruth tryptic peptide."""

ARG_NAMES = [
    "sequences",
    "alphabet_path",
    "suffix_array_path",
    "gap_tolerance",
    "intergap_tolerance",
    "symmetry_threshold",
    "occurrence_threshold",
    "alignment_threshold",
    "alignment_parameters",
    "terminal_residues",
    "gap_key",
    "verbosity",
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
    int,
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
    0,
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
    if args.sequences.endswith(".fasta"):
        sequences = mirror.io.load_fasta_as_strings(args.sequences)
    if args.sequences.startswith("::random"):
        _, n_seqs, seq_len = args.sequences.split("_")
        n_seqs = int(n_seqs)
        seq_len = int(seq_len)
        sequences = [mirror.util.generate_random_tryptic_peptide(seq_len) for _ in range(n_seqs)]
    else:
        sequences = args.sequences.split(",")
    sequence_iterator = mirror.util.add_tqdm(sequences)

    if args.verbosity > 1:
        print(sequences)

    target_groups, residues = mirror.io.load_target_groups(args.alphabet_path)
    target_space = mirror.TargetSpace(target_groups, residues, args.gap_tolerance)

    if args.verbosity > 1:
        for (res, grp) in zip(residues, target_groups):
            print(f"{res}: {grp}")

    # lazy iterators
    tryptic_peptides = enumerate(mirror.util.enumerate_tryptic_peptides(sequence_iterator))

    # pipeline
    optimal_scores = []
    optimal_candidates = []
    for idx, true_sequence in tryptic_peptides:
        peaks = mirror.util.simulate_peaks(true_sequence)
        if args.verbosity > 1:
            print(peaks)
        
        # find gaps
        gap_results = target_space.find_gaps(peaks)
        if args.verbosity > 1:
            for (res, result) in zip(residues, gap_results):
                print(f"{res}: {result.index_tuples()}")

        # find pivots
        symmetry_threshold = args.symmetry_threshold
        if symmetry_threshold == None:
            symmetry_threshold = mirror.util.expected_num_mirror_symmetries(peaks)
        
        pivots = mirror.find_all_pivots(peaks, symmetry_threshold, gap_results, args.intergap_tolerance)

        best_score = np.inf
        best_cand = None
        for pivot in pivots:
            # todo - replace this with a TargetSpace method.
            pivot_residue = mirror.util.residue_lookup(pivot.gap())
            if args.verbosity > 1:
                print();print()
                print(f"\t{pivot_residue} {pivot}")

            # enumerate boundary conditions
            b_boundaries, y_boundaries = mirror.find_boundary_peaks(peaks, pivot, args.terminal_residues)
            for (b_bound, b_res) in b_boundaries:
                for (y_bound, y_res) in y_boundaries:
                    if args.verbosity > 1:
                        print(f"\tboundaries:\n\t\t{peaks[b_bound]}, {b_bound}, {b_res}\n\t\t{peaks[y_bound]}, {y_bound}, {y_res}")
                    
                    # create augmented peaks
                    augmented_peaks, offset = mirror.create_augmented_spectrum(
                        peaks, 
                        pivot,
                        b_bound,
                        y_bound,
                        args.gap_tolerance)
                    
                    # create augmented pivot
                    augmented_pivot = mirror.create_augmented_pivot(
                        augmented_peaks,
                        offset,
                        pivot)
                    
                    # create augmented gaps
                    original_gaps = mirror.util.collapse_second_order_list(r.index_tuples() for r in gap_results)
                    augmented_gaps = mirror.create_augmented_gaps(
                        augmented_peaks,
                        augmented_pivot,
                        offset,
                        original_gaps)

                    print(f"\t\taugmented peaks: {augmented_peaks}")
                    print(f"\t\taugmented pivot: {augmented_pivot}")
                    print(f"\t\taugmented gaps: {augmented_gaps}")
                    
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
                    
                    if args.verbosity > 1:
                        print("\t\tpaths",dual_paths)

                    # find viable path pairings
                    pair_indices = mirror.find_edge_disjoint_dual_path_pairs(dual_paths)

                    # create affixes
                    affixes = np.array([mirror.create_affix(dp, graph_pair) for dp in dual_paths])
                    affixes = mirror.filter_affixes(
                        affixes, 
                        args.suffix_array_path, 
                        occurrence_threshold = args.occurrence_threshold)

                    if args.verbosity > 1:
                        print("\t\taffixes",affixes)

                    # create candidates
                    for (i, j) in pair_indices:
                        candidates = np.array(mirror.create_candidates(
                            augmented_peaks, 
                            graph_pair, 
                            affixes[[i,j]],
                            (b_res, y_res),
                            pivot_residue))
                        #candidates = mirror.filter_candidate_sequences(
                        #    peaks,
                        #    candidates,
                        #    args.alignment_threshold,
                        #    args.alignment_parameters)
                        for candidate in candidates:
                            score, idx = candidate.edit_distance(true_sequence)
                            print("\t\tcandidate:", candidate.sequences()[idx])
                            if score < best_score:
                                best_score = score
                                best_cand = (candidate,idx)
        optimal_scores.append(best_score)
        optimal_candidates.append(best_cand)
        if best_score > 0:
            print(f"edit distance {best_score}")
            if best_score == np.inf:
                print("no candidates were found ):<")
            else:
                best_cand[0].edit_distance(true_sequence, verbose = True)
                print(best_cand[0]._pivot_res)
                print(best_cand[0].characterize_errors(true_sequence))
            #input()

    mirror.util.plot_hist(optimal_scores)

if __name__ == "__main__":
    args = get_parser().parse_args()
    for (k,v) in vars(args).items():
        print(f"{k} : {v}")
    main(args)