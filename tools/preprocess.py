from _tool_init import mirror, pathlib, args, timed_op

def main(
    out,
    mzML: str,
    max_mz: int,
    resolution: float,
    binned_frequency_threshold: int,
):
    experiment = timed_op("loading spectra...", mirror.load_spectrum_from_mzML,
        mzML,
    )

    spectrum_bins = timed_op("binning spectra...", mirror.create_spectrum_bins,
        experiment,
        max_mz,
        resolution,
        parallelize = True,
        n_processes = 8,
    )

    intensities, peaks = timed_op("reducing spectrum bins...", mirror.filter_spectrum_bins,
        spectrum_bins,
        max_mz,
        resolution,
        binned_frequency_threshold,
    )

    timed_op("writing reduced peak list...", mirror.write_peaks_to_csv,
        out,
        intensities,
        peaks)
    
    out.close()


if __name__ == "__main__":
    main(args.output, args.mzML, args.max_mz, args.resolution, args.binned_frequency_threshold)