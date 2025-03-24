from _tool_init import mirror
from pathlib import Path
from numpy import inf

OUTPUT_STACK = """Output[{i}]:
- Pivot[{pivot_index}]
{pivot_repr}
    - Boundary[{pivot_index}, {boundary_index}]
{boundary_repr}
        Augmented Pivot
{augmented_pivot_repr}
        Augmented Gaps {augmented_gaps}
        Augmented Peaks {augmented_peaks}
        Ascending, Descending Graphs
{graph_repr}
        - AffixPair[{pivot_index}, {boundary_index}, {affixes_index}]
            {paths0} → {translations0}
            {paths1} → {translations1}
        - Candidate[{pivot_index}, {boundary_index}, {affixes_index}, {candidate_index}]
{cand_repr}
- Called Sequence
    {best_seq}
- Target
    {target_peptide}
- Edit Distance
    {distance}
"""

def trim_repr(obj):
    return '\n'.join(repr(obj).split('\n')[1:-1])

def main(args):
    print(f"TestSpectrumInspector\n{args}")
    test_spectrum_file, rerun, *_ = args
    rerun = bool(rerun)

    # read the test spectrum
    session_id, output_class, num = Path(test_spectrum_file).stem.split('_')
    test_spectrum = mirror.TestSpectrum.read(test_spectrum_file)
    if rerun:
        test_spectrum.run()
    target_peptide = test_spectrum.get_peptide()
    print(f"target peptide:\n\t{target_peptide}")

    # summary stats
    test_record = mirror.TestRecord(f"{session_id}-inspector")
    test_record.add(test_spectrum)
    test_record.finalize()
    test_record.print_summary()
    test_record.print_complexity_table()
    
    mode = ""
    while mode != "exit":
        mode = input("modes:\n\t'outputs' (default): print each candidate and its output stack.\n\t'iteration': prints each data structure in run-order.\n\t'topology': print ASCII representations for spectrum graph pairs.\n\t'exit': close the inspector.\n\n> ")
        if mode == "iteration":
            for p in range(test_spectrum.n_pivots):
                print(p)
                print(test_spectrum.get_pivot(p))
                if input("press enter to descend, enter anything else to skip to the next item at this level.") != "":
                    continue
                for b in range(test_spectrum.n_boundaries[p]):
                    print(p, b)
                    print(test_spectrum.get_boundary(p, b))
                    aug_spectrum, aug_pivot, aug_gaps, boundary_peaks = test_spectrum.get_augmented_data(p, b)
                    print(aug_spectrum)
                    print(aug_pivot)
                    print(aug_gaps)
                    print(boundary_peaks)
                    print(mirror.spectrum_graphs.draw_graph_pair(test_spectrum.get_spectrum_graph_pair(p, b), "simple"))
                    if input("press enter to descend, enter anything else to skip to the next item at this level.") != "":
                        continue
                    affixes = test_spectrum.get_affixes(p, b)
                    for a in range(test_spectrum.n_affix_pairs[p][b]):
                        afx1, afx2 = affixes[test_spectrum.get_affix_pair(p, b, a)]
                        print(p, b, a)
                        print(afx1)
                        print(afx2)
                        if input("press enter to descend, enter anything else to skip to the next item at this level.") != "":
                            continue
                        for c in range(test_spectrum.n_candidates[p][b][a]):
                            print(p, b, a, c)
                            cand = test_spectrum.get_candidate((p, b, a, c)) 
                            distance, optimizer = cand.edit_distance(target_peptide)
                            best_seq = ''.join(cand.sequences()[optimizer].split(' '))
                            print(cand)
                            print("Optimal Call:")
                            print(distance, best_seq)
                            if input("press enter to descend, enter anything else to skip to the next item at this level.") != "":
                                continue
                        
        elif mode == "" or mode == "outputs":
            outputs = list(enumerate(test_spectrum))
            outputs.sort(key = lambda x: test_spectrum._edit_distances[x[0]])
            for (i, idx) in outputs:
                pivot, boundary, augmented_data, graph_pair, affixes, affix_pair, candidate = test_spectrum.get_output_stack(idx)
                graph_repr = mirror.spectrum_graphs.draw_graph_pair(graph_pair, mode = "simple")
                paths = [afx.path() for afx in affixes[affix_pair]]
                translations = [afx.translate() for afx in affixes[affix_pair]]
                augmented_peaks, augmented_pivot, augmented_gaps, _ = augmented_data
                augmented_peaks = list(augmented_peaks)
                pivot_repr = trim_repr(pivot)
                augmented_pivot_repr = trim_repr(augmented_pivot)
                boundary_repr = trim_repr(boundary)
                cand_repr = trim_repr(candidate)
                distance, optimizer = candidate.edit_distance(target_peptide)
                best_seq = ''.join(candidate.sequences()[optimizer].split(' '))
                print(OUTPUT_STACK.format(
                    i = i,
                    pivot_index = idx.pivot_index,
                    boundary_index = idx.boundary_index,
                    affixes_index = idx.affixes_index,
                    candidate_index = idx.candidate_index,
                    pivot_repr = pivot_repr,
                    boundary_repr = boundary_repr,
                    augmented_pivot_repr = augmented_pivot_repr,
                    augmented_gaps = augmented_gaps,
                    augmented_peaks = augmented_peaks,
                    graph_repr = graph_repr,
                    paths0 = paths[0],
                    paths1 = paths[1],
                    translations0 = translations[0],
                    translations1 = translations[1],
                    cand_repr = cand_repr,
                    best_seq = best_seq,
                    target_peptide = target_peptide,
                    distance = distance
                ))
                input()
        elif mode == "topology":
            for p in range(test_spectrum.n_pivots):
                for b in range(test_spectrum.n_boundaries[p]):
                    graph_pair = test_spectrum.get_spectrum_graph_pair(p, b)
                    graph_repr = mirror.spectrum_graphs.draw_graph_pair(graph_pair, mode = "ascii")
                    input(graph_repr)

if __name__ == "__main__":
    from sys import argv
    main(argv[1:])