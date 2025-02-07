import sys
import argparse
import pathlib
from time import time

sys.path.insert(1, '../')
sys.path.insert(1, '.')
import mirror

_KEYS = [
# primary input/output types
    ('-o', '--output', argparse.FileType('w')),
    ('-i', '--input', str),
    ('-p', '--peaks', pathlib.Path),
    ('-g', '--gaps', pathlib.Path),
    ('-v', '--pivots', pathlib.Path),
    ('-a', '--augmented_triplet', str),
    ('-s', '--spectrum_graph_pair', pathlib.Path),
    ('-x', '--affixes', pathlib.Path),
    ('-c', '--candidates', pathlib.Path),
# parameters
    ('-N', '--num_processes', int),
    ('-V', '--verbosity', int),
#   preprocessing
    ('-M', '--max_mz', int),
    ('-R', '--resolution', float),
    ('-F', '--binned_frequency_threshold', int),
    ('-I', '--spectrum_index', int),
#   gap search
    ('-A', '--target_alphabet', pathlib.Path),
    ('-O', '--target_modifiers', str),
    ('-G', '--gap_tolerance', float),
#   pivot search
    ('-S', '--search_modes', str),
    ('-P', '--intergap_tolerance', float),
#   boundary search
    ('-T', '--valid_terminal_residues', str),
#   spectrum graph creation
    ('-K', '--gap_key', str),
#   affix search and filter
    ('-X', '--suffix_array', pathlib.Path),
    ('-C', '--occurrence_threshold', int),
#   candidate filter
    ('-L', '--aligner', str),
    ('-W', '--alignment_threshold', float),
]

def _build_parser(prog: str, description: str, epilog: str, keys: list):
    parser = argparse.ArgumentParser(
        prog = prog,
        description = description,
        epilog = epilog)
    for x in keys:
        parser.add_argument(
            x[0], x[1],
            type = x[2]
        )
    return parser

_DEFAULT_PARSER = _build_parser("MiRROR", "prototype", "", _KEYS)

#args = _DEFAULT_PARSER.parse_args()
#print(args)

def timed_op(msg, op, *args, **kwargs):
    print(msg)
    time_start = time()
    out = op(*args, **kwargs)
    print(f"\tduration: {time() - time_start}")
    return out