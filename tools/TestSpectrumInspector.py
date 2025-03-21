from _tool_init import mirror
from pathlib import Path
from numpy import inf

def main(args):
    print(f"TestSpectrumInspector\n{args}")
    test_spectrum_file, *_ = args
    # read the test spectrum
    session_id, output_class, num = Path(test_spectrum_file).stem.split('_')
    test_spectrum = mirror.TestSpectrum.read(test_spectrum_file)
    print(f"target peptide:\n\t{test_spectrum.get_peptide()}")
    # summary stats
    test_record = mirror.TestRecord(f"{session_id}-inspector")
    test_record.add(test_spectrum)
    test_record.finalize()
    test_record.print_summary()
    test_record.print_complexity_table()
    
    # graph inspection
    mode = ""
    while mode != "exit":
        mode = input("modes:\n\t'iteration': prints a comprehensive  \n\t'outputs': print each candidate and its output stack.\n\t'topology': print ASCII representations for spectrum graph pairs.\n\t'exit': close the inspector.\n\n> ")
        if mode == "iteration":
            pass
        elif mode == "outputs":
            for (idx, i) in enumerate(test_spectrum):
                pivot, boundary, augmented_data, graph_pair, affixes, affix_pair, candidate = test_spectrum.get_output_stack(i)
                paths = [afx.path() for afx in affixes[affix_pair]]
                translations = [afx.translate() for afx in affixes[affix_pair]]
                augmented_peaks, augmented_pivot, augmented_gaps, _ = augmented_data
                augmented_peaks = list(augmented_peaks)
                print(f"""Output[{idx}]:
                
-   Pivot[{i.pivot_index}]
    {pivot}

    -   Boundary[{i.pivot_index}, {i.boundary_index}]
        {boundary}
        Augmented {augmented_pivot}
        Augmented Gaps {augmented_gaps}
        Augmented Peaks {augmented_peaks}
        Ascending, Descending Graphs
{mirror.spectrum_graphs.draw_graph_pair_ascii(graph_pair)}

        -   AffixPair[{i.pivot_index}, {i.boundary_index}, {i.affixes_index}]
            {paths[0]} → {translations[0]}
            {paths[1]} → {translations[1]}

[{i.pivot_index}, {i.boundary_index}, {i.affixes_index}, {i.candidate_index}]
{candidate.sequences()}""")
                input()
        elif mode == "topology":
            for p in range(test_spectrum.n_pivots):
                for b in range(test_spectrum.n_boundaries[p]):
                    graph_pair = test_spectrum.get_spectrum_graph_pair(p, b)
                    print(mirror.spectrum_graphs.draw_graph_pair_ascii(graph_pair))
                    input()

if __name__ == "__main__":
    from sys import argv
    main(argv[1:])