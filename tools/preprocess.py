from _tool_init import mirror, pathlib, _DEFAULT_PARSER, timed_op
args = default_parser.parse_args()

def main(
    out,
    input_path: str,
    max_mz: int,
    resolution: float,
    binned_frequency_threshold: int,
    num_processes: int,
    spectrum_index: int,
    verbosity: int,
):
    # mzML branch
    if input_path.endswith(".mzML"):
        experiment = timed_op("loading spectra...", mirror.io.load_spectrum_from_mzML,
            input_path,
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
            peaks
        )
    elif input_path.endswith(".mzlib.txt"):
        mzspeclib = timed_op("loading spectra...", mirror.io.load_spectra_from_mzSpecLib,
            input_path,
        )

        def getdata():
            intensity, mz, metadata, etc = mzspeclib.get_peaks(spectrum_index)
            for (i,z,d,e) in zip(intensity,mz,metadata,etc):
                print(i,z,d,e)
            peptides = mzspeclib.get_peptides(spectrum_index)
            return intensity, mz, peptides
        
        intensities, peaks, peptides = timed_op("retrieving spectrum...", getdata)

        timed_op("writing peak list...", mirror.io.write_peaks_to_csv,
            out,
            intensities,
            peaks
        )

        out_path = pathlib.Path(out.name)
        fasta_path = out_path.parent / (out_path.stem + f".fasta") 
        timed_op(f"writing peptides to {fasta_path}...", mirror.io.save_strings_as_fasta,
            fasta_path,
            peptides
        )
    else:
        print(f"unrecognized input filetype {input_path}. please pass a file ending in .mzML or .mzlib.txt.")

    try:
        out.close()
    except Exception as e:
        print(f"failed to close output file:\n\t{repr(e)}")

if __name__ == "__main__":
    main(args.output, 
        args.input, 
        args.max_mz, 
        args.resolution, 
        args.binned_frequency_threshold, 
        args.num_processes,
        args.spectrum_index,
        args.verbosity)