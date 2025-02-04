from _tool_init import mirror, pathlib, args, timed_op

def main(
    out,
    peaks_path: pathlib.Path,
    targets_path: pathlib.Path,
    target_modifiers: str,
    gap_tolerance: float,
):
    peaks = timed_op("loading peaks...", mirror.io.load_peaks_from_csv, 
        peaks_path,
    )

    target_groups = timed_op("loading gap targets...", mirror.io.load_target_groups,
        targets_path,
    )

    target_space = timed_op("creating target space... (1/2)", mirror.TargetSpace,
        target_groups,
        gap_tolerance,
    )

    gaps = timed_op("finding gaps... (2/2)", target_space.find_gaps,
        peaks,
    )

    gaps_2 = timed_op("finding gaps... (old method)", mirror.find_all_gaps,
        peaks,
        target_groups,
        gap_tolerance
    )

    timed_op("saving gap results...", mirror.io.write_gap_results,
        out,
        gaps)
    
    out.close()

if __name__ == "__main__":
    main(args.output,
        args.peaks,
        args.target_alphabet,
        args.target_modifiers,
        args.gap_tolerance)