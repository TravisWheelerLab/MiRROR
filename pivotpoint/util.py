from copy import deepcopy
from pprint import pprint

AMINO_ACID_CHAR_REPRESENTATION = [
    'A',
    'R',
    'N',
    'D',
    'C',
    'E',
    'Q',
    'G',
    'H',
    'I',
    'L',
    'K',
    'M',
    'F',
    'P',
    'S',
    'T',
    'W',
    'Y',
    'V',
]

AMINO_ACID_MASS_MONOISOTOPIC = [
    71.037,
    156.101,
    114.042,
    115.027,
    103.009,
    129.043,
    128.059,
    57.021,
    137.059,
    113.084,
    113.084,
    128.095,
    131.040,
    147.068,
    97.053,
    87.032,
    101.048,
    186.079,
    163.063,
    99.068]

AMINO_ACID_MASS_AVERAGE = [
    71.08,
    156.2,
    114.1,
    115.1,
    103.1,
    129.1,
    128.1,
    57.05,
    137.1,
    113.2,
    113.2,
    128.2,
    131.2,
    147.2,
    97.12,
    87.08,
    101.1,
    186.2,
    163.2,
    99.13]

def collect_data(pivotmap):
    return [pivot.data() for pivot in deepcopy(pivotmap)]

def unique(a):
    return list(set(a))

def subset(A,B):
    return all([B.count(a) > 0 for a in A])

def dump_object_data(obj):
    pprint(vars(obj))

def make_index_table(arr: list):
    return dict([x[::-1] for x in enumerate(arr)])
