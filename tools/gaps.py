from _tool_init import mirror, pathlib, args, timed_op

def main(
    out,
    peaks_path: pathlib.Path,
    targets_path: pathlib.Path,
    target_modifiers: str,
    gap_tolerance: float,
    verbosity: int,
):
    path_str = str(peaks_path)
    if path_str.endswith(".csv"):
        intensities, peaks = timed_op("loading peaks...", mirror.io.load_peaks_from_csv, 
            peaks_path,
        )
    elif path_str.endswith(".fasta"):
        sequence = mirror.io.load_fasta_as_strings(peaks_path)[0]
        peaks = mirror.util.list_mz(mirror.util.generate_default_fragment_spectrum(sequence))

    target_groups, residues = timed_op("loading gap targets...", mirror.io.load_target_groups,
        targets_path,
    )

    target_space = timed_op("creating target space... (1/2)", mirror.TargetSpace,
        target_groups,
        residues,
        gap_tolerance,
    )

    gap_results = timed_op("finding gaps... (2/2)", target_space.find_gaps,
        peaks,
    )

    gaps_2 = timed_op("finding gaps... (old method)", mirror.find_all_gaps,
        peaks,
        target_groups,
        gap_tolerance
    )

    timed_op("saving gap results...", mirror.io.write_gap_results,
        out,
        gap_results)
    
    out.close()

    if verbosity == 2:
        for residue,group_values,result in zip(residues,target_groups,gap_results):
            print(residue,group_values,[(i,j) for (i,j) in result.indices()])
    elif verbosity == 1:
        for (res,G1,G2) in zip(residues,gap_results, gaps_2):
            print(res, len(G1), len(G2))
            if len(G1) == len(G2):        
                for (g1,(_,g2,__)) in zip(G1.indices(),G2):
                    if [g1[0],g1[1]] != [g2[0],g2[1]]:
                        print(g1,g2)

if __name__ == "__main__":
    main(args.output,
        args.peaks,
        args.target_alphabet,
        args.target_modifiers,
        args.gap_tolerance,
        args.verbosity)