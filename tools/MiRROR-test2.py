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

def get_parser():
    parser = argparse.ArgumentParser(
        prog = PROG,
        description = DESC
    )
    for (argname, argtype, argdefault) in zip(ARG_NAMES, ARG_TYPES, ARG_DEFAULTS):
        parser.add_argument(f"--{argname}", type = argtype, default = argdefault)
    return parser

def main(args):
    printer = mirror.util.get_respectful_printer(args)
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
        printer(f"mz\n\t{peaks}", 2)
        
        residue_seq = np.array([r for r in true_sequence])
        printer(f"\nresidue sequence\n\t{residue_seq}", 2)

        mass_seq = np.array([mirror.util.RESIDUE_MONO_MASSES[r] for r in residue_seq])
        printer(f"mass sequence\n\t{mass_seq}", 2)

        test_spectrum = mirror.TestSpectrum(
            residue_seq,
            mass_seq,
            np.zeros_like(mass_seq),
            np.zeros_like(mass_seq),
            np.zeros_like(mass_seq),
            np.array([]),
            peaks,
            gaps = [],
            pivot = None,
            gap_search_parameters = gap_params,
            intergap_tolerance = args.intergap_tolerance,
            symmetry_factor = 1.0,
            terminal_residues = args.terminal_residues,
            boundary_padding = args.boundary_padding,
            gap_key = args.gap_key
        )

        best_score, best_cand = test_spectrum.optimize  ()
        optimal_scores.append(best_score)
        optimal_candidates.append(best_cand)

    mirror.util.plot_hist(optimal_scores)

if __name__ == "__main__":
    args = get_parser().parse_args()
    for (k,v) in vars(args).items():
        print(f"{k} : {v}")
    main(args)