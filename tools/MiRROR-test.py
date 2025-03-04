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
    "boundary_padding",
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
    int,
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
    mirror.util.BOUNDARY_PADDING,
    mirror.spectrum_graphs.GAP_KEY,
    0,
]

def get_respectful_printer(args):
    def print_respectfully(msg, verbosity_level, arg_verbosity = args.verbosity):
        if arg_verbosity >= verbosity_level:
            print(msg)
    return print_respectfully

def get_parser():
    parser = argparse.ArgumentParser(
        prog = PROG,
        description = DESC
    )
    for (argname, argtype, argdefault) in zip(ARG_NAMES, ARG_TYPES, ARG_DEFAULTS):
        parser.add_argument(f"--{argname}", type = argtype, default = argdefault)
    return parser

def main(args):
    printer = get_respectful_printer(args)
    # load sequences, gap alphabet
    if args.sequences.endswith(".fasta"):
        sequences = mirror.io.load_fasta_as_strings(args.sequences)
    elif args.sequences.startswith("::random"):
        _, n_seqs, seq_len = args.sequences.split("_")
        n_seqs = int(n_seqs)
        seq_len = int(seq_len)
        sequences = [mirror.util.generate_random_tryptic_peptide(seq_len) for _ in range(n_seqs)]
    else:
        sequences = args.sequences.split(",")
    sequence_iterator = mirror.util.add_tqdm(sequences)
    printer(f"\nsequences:\n{sequences}\n", 2)

    gap_params = mirror.util.DEFAULT_GAP_SEARCH_PARAMETERS
    printer(f"alphabet:", 2)
    for (res, grp) in zip(gap_params.residues, gap_params.masses):
        printer(f"{res}: {grp}", 2)

    # lazy iterators
    tryptic_peptides = mirror.util.enumerate_tryptic_peptides(sequence_iterator)

    # pipeline
    optimal_scores = []
    optimal_candidates = []
    for peptide_idx, true_sequence in enumerate(tryptic_peptides):
        peaks = mirror.util.simulate_peaks(true_sequence)
        printer(f"\npeptide {peptide_idx}: {true_sequence}\n", 2)
        printer(f"peaks: {peaks}\n", 2)
        
        # find gaps
        gap_results = mirror.find_gaps(gap_params, peaks)
        printer("gaps:", 2)
        for (res, result) in zip(gap_params.residues, gap_results):
            printer(f"{res}: {result.get_index_pairs()}", 2)

        # find pivots
        symmetry_threshold = args.symmetry_threshold
        if symmetry_threshold == None:
            symmetry_threshold = mirror.util.expected_num_mirror_symmetries(peaks)
        
        pivots = mirror.find_all_pivots(peaks, symmetry_threshold, gap_results, args.intergap_tolerance)

        best_score = np.inf
        best_cand = None
        for pivot_idx, pivot in enumerate(pivots):
            # todo - replace this with a TargetSpace method.
            pivot_residue = mirror.util.residue_lookup(pivot.gap())

            printer(f"\npivot {pivot_idx}: {pivot_residue}\n\t{pivot}", 2)
            boundaries, b_ions, y_ions = mirror.find_and_create_boundaries(peaks, pivot, gap_params, args.terminal_residues, args.gap_tolerance, args.boundary_padding)
            if len(boundaries) == 0:
                printer("\tno boundaries.", 2)
                continue
            else:
                printer(f"\t{b_ions}\n\t{y_ions}", 2)
            for boundary in boundaries:
                b_res, y_res = boundary.get_residues()
                augmented_peaks = boundary.get_augmented_peaks()
                offset = boundary.get_offset()
                augmented_pivot = boundary.get_augmented_pivot()
                augmented_gaps = boundary.get_augmented_gaps()
                printer(f"\t\taugmented peaks: {augmented_peaks}, offset = {offset}", 2)
                printer(f"\t\taugmented pivot: {augmented_pivot}", 2)
                printer(f"{'-'*10}\n\t\taugmented gaps: {augmented_gaps}", 2)
                
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
                
                printer(f"\t\tpaths {dual_paths}", 2)

                # find viable path pairings
                pair_indices = mirror.find_edge_disjoint_dual_path_pairs(dual_paths)

                # create affixes
                affixes = np.array([mirror.create_affix(dp, graph_pair) for dp in dual_paths])
                affixes = mirror.filter_affixes(
                    affixes, 
                    args.suffix_array_path, 
                    occurrence_threshold = args.occurrence_threshold)

                printer(f"\t\taffixes {affixes}", 2)
                
                if len(pair_indices) == 0:
                    printer("\t\tNO CANDIDATES.", 2)
                    continue

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
                        printer(f"\t\tcandidate: {candidate.sequences()[idx]}\t{candidate._pivot_res}\t{score}", 2)
                        if score < best_score:
                            best_score = score
                            best_cand = (candidate,idx)
        if best_score > 0:
            printer(f"{'-'*10}\nedit distance {best_score}", 1)
            if best_score == np.inf:
                printer("no candidates were found ):<", 1)
            else:
                best_cand[0].edit_distance(true_sequence, verbose = args.verbosity >= 1)
                printer(best_cand[0]._pivot_res, 1)
                printer(best_cand[0].characterize_errors(true_sequence), 1)
            #input()
        if best_score == np.inf:
            best_score = len(true_sequence)
        optimal_scores.append(best_score)
        optimal_candidates.append(best_cand)

    mirror.util.plot_hist(optimal_scores)

if __name__ == "__main__":
    args = get_parser().parse_args()
    for (k,v) in vars(args).items():
        print(f"{k} : {v}")
    main(args)