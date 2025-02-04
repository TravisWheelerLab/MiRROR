from _tool_init import mirror, pathlib, args, timed_op

def main(
    out,
    mzML_path: str,
    max_mz: int,
    resolution: float,
    binned_frequency_threshold: int,
    num_processes: int,
):
    experiment = timed_op("loading spectra...", mirror.io.load_spectrum_from_mzML,
        mzML_path,
    )

    if num_processes != None:
        spectrum_bins = timed_op("binning spectra (parallel)...", mirror.create_spectrum_bins,
            experiment,
            max_mz,
            resolution,
            parallelize = True,
            n_processes = num_processes,
        )
    else:
        spectrum_bins = timed_op("binning spectra...", mirror.create_spectrum_bins,
            experiment,
            max_mz,
            resolution
        )

    intensities, peaks = timed_op("reducing spectrum bins...", mirror.filter_spectrum_bins,
        spectrum_bins,
        max_mz,
        resolution,
        binned_frequency_threshold,
    )

    timed_op("writing reduced peak list...", mirror.io.write_peaks_to_csv,
        out,
        intensities,
        peaks)
    
    out.close()


if __name__ == "__main__":
    main(args.output, 
        args.mzML, 
        args.max_mz, 
        args.resolution, 
        args.binned_frequency_threshold, 
        args.num_processes)