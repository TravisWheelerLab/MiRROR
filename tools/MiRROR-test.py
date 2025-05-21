from _tool_init import mirror, timed_op, argparse
import numpy as np
from pathlib import Path
import uuid
from time import time

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
    "pivot_precision",
    "symmetry_factor",
    "occurrence_threshold",
    "alignment_threshold",
    "alignment_parameters",
    "terminal_residues",
    "boundary_padding",
    "gap_key",
    "verbosity",
    "output_dir",
    "session_id",
    "write_matches",
    "simulation_mode",
    "crash",
]
ARG_TYPES = [
    str,
    str,
    str,
    float,
    float,
    int,
    float,
    int,
    float,
    str,
    list[str],
    int,
    str,
    int,
    str,
    str,
    bool,
    str,
    bool,
]
ARG_DEFAULTS = [
    None,
    None,
    None,
    mirror.util.GAP_TOLERANCE,
    mirror.util.INTERGAP_TOLERANCE,
    4,
    5.0,
    0,
    -1,
    "",
    mirror.util.TERMINAL_RESIDUES,
    mirror.util.BOUNDARY_PADDING,
    mirror.graphs.spectrum_graphs.GAP_KEY,
    0,
    "./data/output/",
    str(uuid.uuid4())[:8],
    False,
    "simple",
    False,
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

    # construct tryptic peptides
    if args.sequences.endswith(".fasta"):
        # load from fasta
        sequences = mirror.io.load_fasta_as_strings(args.sequences)
        tryptic_peptides = mirror.util.enumerate_tryptic_peptides(sequences)
    elif args.sequences.startswith("::random"):
        # generate randomly
        _, n_seqs, seq_len = args.sequences.split("_")
        n_seqs = int(n_seqs)
        seq_len = int(seq_len)
        sequences = tryptic_peptides = [mirror.util.generate_random_tryptic_peptide(seq_len) for _ in range(n_seqs)]
        if n_seqs == 1:
            print("peptide", sequences[0])
    else:
        # parse from sequences arg
        sequences = args.sequences.split(",")
        tryptic_peptides = mirror.util.enumerate_tryptic_peptides(sequences)
    
    # prepare suffix array
    if args.suffix_array_path == None or args.suffix_array_path == "None":
        suffix_array = None
    elif args.suffix_array_path == "::auto":
        temp_fasta_file = "_temp.fa"
        mirror.save_strings_as_fasta(temp_fasta_file, sequences)
        suffix_array = mirror.SuffixArray.create(temp_fasta_file)
    else:
        suffix_array = mirror.SuffixArray.read(args.suffix_array_path)

    # create gap parameters
    if args.gap_params == "simple":
        gap_params = mirror.gaps.SIMPLE_GAP_SEARCH_PARAMETERS
    elif args.gap_params == "uncharged":
        gap_params = mirror.gaps.UNCHARGED_GAP_SEARCH_PARAMETERS
    else: # elif args.gap_params == "default":
        gap_params = mirror.gaps.DEFAULT_GAP_SEARCH_PARAMETERS
    printer(f"alphabet:", 2)
    for (res, grp) in zip(gap_params.residues, gap_params.masses):
        printer(f"{res}: {grp}", 2)
    
    # parametize peak simulator
    if args.simulation_mode == "simple":
        simulation_param = mirror.util.DEFAULT_PARAM
    elif args.simulation_mode == "complex":
        simulation_param = mirror.util.ADVANCED_PARAM

    printer(f"\ttryptic_peptides:\n\t{tryptic_peptides}\n", 1)
    tryptic_peptides = mirror.util.add_tqdm(list(tryptic_peptides))

    test_record = mirror.TestRecord(args.session_id)
    time_start = time()
    for peptide_idx, true_sequence in enumerate(tryptic_peptides):
        peaks = mirror.util.simulate_peaks(true_sequence, param = simulation_param)
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
            target_gaps = [],
            target_pivot = None,
            gap_search_parameters = gap_params,
            intergap_tolerance = args.intergap_tolerance,
            pivot_precision = args.pivot_precision,
            symmetry_factor = args.symmetry_factor,
            terminal_residues = args.terminal_residues,
            boundary_padding = args.boundary_padding,
            gap_key = args.gap_key,
            suffix_array = suffix_array,
            occurrence_threshold = args.occurrence_threshold,
            crash = args.crash,
        )

        test_record.add(test_spectrum)
    time_elapsed = time() - time_start
    test_record.finalize()
    
    test_record.print_summary()
    test_record.print_miss_distances()
    test_record.print_complexity_table()

    output_dir = Path(args.output_dir)
    test_record.save_misses(output_dir / "misses/")
    test_record.save_crashes(output_dir / "crashes/")
    test_record.save_temporal_outliers(output_dir / "temporal_outliers/")
    test_record.save_spatial_outliers(output_dir / "spatial_outliers/")
    if args.write_matches:
        test_record.save_matches(output_dir / "matches/")
    print("RUNTIME:", time_elapsed)

if __name__ == "__main__":
    args = get_parser().parse_args()
    for (k,v) in vars(args).items():
        print(f"{k} {v}")
    main(args)
