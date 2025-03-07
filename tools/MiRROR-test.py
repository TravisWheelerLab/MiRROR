from _tool_init import mirror, timed_op, argparse
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import matplotlib.pyplot as plt
from copy import deepcopy

def draw_graph(graph, title, gap_key):
    graph.remove_node(-1)
    graph.graph['graph'] = {'rankdir':'LR'}
    graph.graph['node'] = {'shape':'circle'}
    graph.graph['edges'] = {'arrowsize':'4.0'}
    A = to_agraph(graph)
    for (i,j) in graph.edges:
        truncated_weight = round(graph[i][j][gap_key],4)
        res = mirror.util.residue_lookup(truncated_weight)
        A.get_edge(i,j).attr['label'] = f"{res} [ {str(truncated_weight)} ]" 
    A.layout('dot')
    A.draw(title)

PROG = "𝕄 iℝ ℝ 𝕆 ℝ - test\n"
DESC = """the test system for MiRROR;
reads sequences from a .fasta, or passed as a comma-separated string, or generated via ::random_<NUM SEQS>_<SEQ LEN>.
enumerates the tryptic peptides of those sequences.
generates synthetic spectra for each peptide.
applies the integrated MiRROR pipeline to the spectra;
compares result(s) to the groundtruth tryptic peptide."""

ARG_NAMES = [
    "sequences",
    "gap_params",
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
        tryptic_peptides = mirror.util.enumerate_tryptic_peptides(sequences)
    elif args.sequences.startswith("::random"):
        _, n_seqs, seq_len = args.sequences.split("_")
        n_seqs = int(n_seqs)
        seq_len = int(seq_len)
        tryptic_peptides = [mirror.util.generate_random_tryptic_peptide(seq_len) for _ in range(n_seqs)]
    else:
        sequences = args.sequences.split(",")
        tryptic_peptides = mirror.util.enumerate_tryptic_peptides(sequences)

    if args.gap_params == "simple":
        gap_params = mirror.gaps.SIMPLE_GAP_SEARCH_PARAMETERS
    elif args.gap_params == "uncharged":
        gap_params = mirror.gaps.UNCHARGED_GAP_SEARCH_PARAMETERS
    else: # elif args.gap_params == "default":
        gap_params = mirror.gaps.DEFAULT_GAP_SEARCH_PARAMETERS
    printer(f"alphabet:", 2)
    for (res, grp) in zip(gap_params.residues, gap_params.masses):
        printer(f"{res}: {grp}", 2)

    tryptic_peptides = mirror.util.add_tqdm(list(tryptic_peptides))
    printer(f"\nsequences:\n{tryptic_peptides}\n", 1)

    # pipeline
    optimal_scores = []
    optimal_candidates = []
    for peptide_idx, true_sequence in enumerate(tryptic_peptides):
        peaks = mirror.util.simulate_peaks(true_sequence)
        printer(f"\npeptide {peptide_idx}: {true_sequence}\n", 2)
        printer(f"peaks: {peaks}\n", 2)
        
        # find gaps
        annotated_peaks, gap_results = mirror.find_gaps(gap_params, peaks)
        printer(f"gaps:\n\t(annotated peaks)\n{annotated_peaks}", 2)
        for (res, result) in zip(gap_params.residues, gap_results):
            printer(f"{res}: {result.get_index_pairs()}", 2)

        # find pivots
        symmetry_threshold = args.symmetry_threshold
        if symmetry_threshold == None:
            symmetry_threshold = mirror.util.expected_num_mirror_symmetries(annotated_peaks)
        
        pivots = mirror.find_all_pivots(annotated_peaks, symmetry_threshold, gap_results, args.intergap_tolerance)

        best_score = np.inf
        best_cand = None
        for pivot_idx, pivot in enumerate(pivots):
            pivot_residue = mirror.util.residue_lookup(pivot.gap())

            printer(f"\npivot {pivot_idx}: {pivot_residue}\n\t{pivot}", 2)
            boundaries, b_ions, y_ions = mirror.find_and_create_boundaries(
                annotated_peaks, 
                pivot, 
                gap_params, 
                args.terminal_residues, 
                args.boundary_padding
            )
            
            if len(boundaries) == 0:
                printer("\tno boundaries.", 2)
                continue
            else:
                printer(f"\t{b_ions}\n\t{y_ions}", 2)
            for b_idx, boundary in enumerate(boundaries):
                printer(f"\tboundary {b_idx}", 2)
                b_res, y_res = boundary.get_residues()
                boundary_indices = boundary.get_boundary_indices()
                augmented_peaks = boundary.get_augmented_peaks()
                offset = boundary.get_offset()
                augmented_pivot = boundary.get_augmented_pivot()
                augmented_gaps = boundary.get_augmented_gaps()
                printer(f"\t\taugmented peaks: {augmented_peaks}, offset = {offset}", 2)
                printer(f"\t\taugmented pivot: {augmented_pivot}", 2)
                printer(f"\t\taugmented gaps: {augmented_gaps}", 2)
                
                # create topologies
                graph_pair = mirror.create_spectrum_graph_pair(
                    augmented_peaks, 
                    augmented_gaps, 
                    augmented_pivot,
                    boundary_indices,
                    gap_key = args.gap_key)
                #for (name,graph) in zip(["asc", "desc"],graph_pair):
                #    draw_graph(deepcopy(graph), f"{name}_graph.png", args.gap_key)
                
                # find dual paths
                gap_comparator = lambda x, y: (abs(x - y) < args.intergap_tolerance) and (x != -1) and (y != -1)
                dual_paths = mirror.graph_utils.find_dual_paths(
                    *graph_pair,
                    args.gap_key,
                    gap_comparator)
                
                printer(f"\t\tpaths {dual_paths}", 2)

                # find viable path pairings
                pair_indices = mirror.find_edge_disjoint_dual_path_pairs(dual_paths)

                # create affixes
                affixes = np.array([mirror.create_affix(dp, graph_pair) for dp in dual_paths])
                #affixes = mirror.filter_affixes(
                #    affixes, 
                #    args.suffix_array_path, 
                #    occurrence_threshold = args.occurrence_threshold)

                printer(f"\t\taffixes {affixes}\n\t\taffix translations {[affix.translate() for affix in affixes]}", 2)
                
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
        printer(best_score, 1)
        if best_score > 0:
            printer(f"{'-'*10}\nedit distance {best_score}", 1)
            if best_score == np.inf:
                printer("no candidates were found ):<", 1)
                best_score = 2 * len(true_sequence)
            else:
                best_cand[0].edit_distance(true_sequence, verbose = args.verbosity >= 1)
                printer(best_cand[0]._pivot_res, 1)
                printer(best_cand[0].characterize_errors(true_sequence), 1)
        #input()
        optimal_scores.append(best_score)
        optimal_candidates.append(best_cand)

    mirror.util.plot_hist(optimal_scores)

if __name__ == "__main__":
    args = get_parser().parse_args()
    for (k,v) in vars(args).items():
        print(f"{k} : {v}")
    main(args)