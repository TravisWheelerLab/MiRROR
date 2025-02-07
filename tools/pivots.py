from _tool_init import mirror, pathlib, _DEFAULT_PARSER, timed_op
args = default_parser.parse_args()

def main(
    out,
    peaks_path: str,
    gaps_path: str,
    search_modes: str,
    tolerance: float,
    num_processes: int,
    verbosity: int,
):
    intensities, peaks = timed_op("loading peaks...", mirror.io.load_peaks_from_csv, 
        peaks_path,
    )

    symmetry_threshold = mirror.util.expected_num_mirror_symmetries(peaks)
    
    gap_results = timed_op("loading gap results...", mirror.io.load_gap_results,
        gaps_path
    )

    if num_processes == None or num_processes == 1:
        for result in gap_results:
            indices = [(i,j) for (i,j) in result.indices()]
            print(indices)
            pivots = timed_op("finding pivots...", mirror.find_pivots,
                peaks,
                symmetry_threshold,
                indices,
                tolerance,
            )
            print(len(pivots))
    else:
        # parallel
        pass


if __name__ == "__main__":
    main(args.output,
        args.peaks,
        args.gaps,
        args.search_modes,
        args.intergap_tolerance,
        args.num_processes,
        args.verbosity)