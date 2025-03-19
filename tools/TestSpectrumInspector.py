from _tool_init import mirror
from pathlib import Path

def main(args):
    print(f"TestSpectrumInspector\n{args}")
    test_spectrum_file, *_ = args
    session_id, output_class, num = Path(test_spectrum_file).stem.split('_')
    test_spectrum = mirror.TestSpectrum.read(test_spectrum_file)
    test_record = mirror.TestRecord(f"{session_id}-inspector")
    test_record.add(test_spectrum)
    test_record.finalize()
    test_record.print_summary()
    test_record.print_complexity_table()

if __name__ == "__main__":
    from sys import argv
    main(argv[1:])