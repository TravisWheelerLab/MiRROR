from _tool_init import mirror
from pathlib import Path

def main(args):
    print(f"TestSpectrumInspector\n{args}")
    test_spectrum_file, *_ = args
    # read the test spectrum
    session_id, output_class, num = Path(test_spectrum_file).stem.split('_')
    test_spectrum = mirror.TestSpectrum.read(test_spectrum_file)
    # summary stats
    test_record = mirror.TestRecord(f"{session_id}-inspector")
    test_record.add(test_spectrum)
    test_record.finalize()
    test_record.print_summary()
    test_record.print_complexity_table()
    
    gaps, peaks, y_terminii, pivots, boudnaries, augments, graphs, affixes, affix_pairs = test_spectrum.get_state()
    # graph inspection
    graphs = mirror.util.collapse_second_order_list(graphs)
    for graph_pair in graphs:
        print(mirror.spectrum_graphs.draw_graph_pair_ascii(graph_pair))
        input()


if __name__ == "__main__":
    from sys import argv
    main(argv[1:])