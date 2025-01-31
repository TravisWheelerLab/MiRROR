import itertools

import pyopenms as oms
import numpy as np
from tqdm import tqdm

#=============================================================================#
# residue constants and functions

AMINO_MASS = [
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

AMINO_MASS_MONO = [
    71.037,
    156.10,
    114.04,
    115.03,
    103.01,
    129.04,
    128.06,
    57.021,
    137.06,
    113.08,
    113.08,
    128.10,
    131.04,
    147.07,
    97.053,
    87.032,
    101.05,
    186.08,
    163.06,
    99.068]

AVERAGE_MASS_DIFFERENCE = np.mean(np.abs(np.array(AMINO_MASS) - np.array(AMINO_MASS_MONO)))

LOOKUP_TOLERANCE = 0.1
GAP_TOLERANCE = 0.01 # min(abs(m1 - m2) for m1 in AMINO_MASS_MONO for m2 in AMINO_MASS_MONO if abs(m1 - m2) > 0)
INTERGAP_TOLERANCE = GAP_TOLERANCE * 2 

AMINOS = [
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
    'V']

UNKNOWN_AMINO = 'X'

RESIDUES = AMINOS

TERMINAL_RESIDUES = ['R', 'K']

NONTERMINAL_RESIDUES = [r for r in RESIDUES if r not in TERMINAL_RESIDUES]

AMINO_MASS_LOOKUP = dict(zip(AMINOS,AMINO_MASS))

AMINO_MASS_MONO_LOOKUP = dict(zip(AMINOS,AMINO_MASS_MONO))

ION_SERIES_OFFSETS = [
    -27,
    1,
    18,
    45,
    19,
    2]

ION_SERIES = [
    'a',
    'b',
    'c',
    'x',
    'y',
    'z']

ION_OFFSET_LOOKUP = dict(zip(ION_SERIES,ION_SERIES_OFFSETS))

def generate_random_residues(length: int, alphabet = RESIDUES):
    return np.random.choice(alphabet, length, replace=True)

def generate_random_peptide(length: int, alphabet = RESIDUES):
    return ''.join(generate_random_residues(length, alphabet))

def generate_random_tryptic_peptide(length: int):
    terminus = np.random.choice(TERMINAL_RESIDUES)
    prefix = generate_random_peptide(length - 1, alphabet = NONTERMINAL_RESIDUES)
    return prefix + terminus

def mask_ambiguous_residues(res: chr):
    if res == "L" or res == "I":
        return "I/L"
    else:
        return res

def mass_error(
    mass: float,
    mass_table: list[float] = AMINO_MASS_MONO,
):
    return min([abs(m - mass) for m in mass_table])

def residue_lookup(
    gap: float,
    letters: list[str] = AMINOS,
    mass_table: list[float] = AMINO_MASS_MONO,
    tolerance: float = LOOKUP_TOLERANCE,
    unknown = UNKNOWN_AMINO,
    nan = '.',
):
    if gap == -1:
        return unknown
    elif gap == -2:
        return nan
    else:
        dif = [abs(m - gap) for m in mass_table]
        i = np.argmin(dif)
        optimum = dif[i]
        if optimum > tolerance: # no match
            return unknown
        else:
            return mask_ambiguous_residues(letters[i])
    
#=============================================================================#
# misc utilities

def comma_separated(items):
    return ','.join(map(str, items))

def split_commas(commastr, target_type):
    return list(map(target_type, commastr.split(',')))

def collapse_second_order_list(llist: list[list]):
    return list(itertools.chain.from_iterable(llist))

def log(message, prefix=""):
    print(prefix + f"⚙\t{message}")

def add_tqdm(inputs, total=None, description=None):
    if total == None:
        total = len(inputs)
    return tqdm(inputs, total=total, leave=False, desc=description)

#=============================================================================#
# simulation via pyOpenMS

def digest_trypsin(seq: str, minimum_length: int = 7, maximum_length: int = 40):
    dig = oms.ProteaseDigestion()
    result = []
    oms_seq = oms.AASequence.fromString(seq)
    dig.digest(oms_seq, result, minimum_length, maximum_length)
    return [r.toString() for r in result]

def generate_fragment_spectrum(seq: str, param: oms.Param):
    tsg = oms.TheoreticalSpectrumGenerator()
    spec = oms.MSSpectrum()
    peptide = oms.AASequence.fromString(seq)
    tsg.setParameters(param)
    tsg.getSpectrum(spec, peptide, 1, 1)
    return spec

def generate_default_fragment_spectrum(seq: str):
    param = oms.Param()
    param.setValue("add_metainfo", "true")
    return generate_fragment_spectrum(seq, param)

def list_mz(spec: oms.MSSpectrum):
    return np.array([peak.getMZ() for peak in spec])

def get_b_ion_series(seq: str):
    param = oms.Param()
    param.setValue("add_b_ions", "true")
    param.setValue("add_y_ions", "false")
    return list_mz(generate_fragment_spectrum(seq,param))
    
def get_y_ion_series(seq: str):
    param = oms.Param()
    param.setValue("add_b_ions", "false")
    param.setValue("add_y_ions", "true")
    return list_mz(generate_fragment_spectrum(seq,param))

#=============================================================================#
# mirror symmetry

def reflect(x, center: float):
    """Reflect about a given point.
    :param x: the number or numeric array subject to reflection.
    :param center: the center of reflection."""
    return 2 * center - x

def count_mirror_symmetries(arr: np.array, center: float, tolerance = 0.01):
    """Count the elements of `arr` that are symmetric under reflection around `center`.
    
    :param arr: a sorted one-dimensional numeric array.
    :param center: the center of reflection.
    :param tolerance: the proximity threshold for detecting mirror symmetry. defaults to 0.01.
    :return: the number of elements that are symmetric under the reflection."""
    reflected_arr = reflect(arr, center)
    n_symmetric = 0
    for reflected_val in reflected_arr:
        if np.min(np.abs(arr - reflected_val)) < tolerance:
            n_symmetric += 1
    return n_symmetric

def expected_num_mirror_symmetries(arr: np.array, tolerance = 0.01):
    """Estimate the number of mirror-symmetric elements under a random reflection.

        len(arr) * (len(arr) - 1) * tolerance / (max(arr) - min(arr))
    
    :param arr: a sorted one-dimensional numeric array.
    :param tolerance: the proximity threshold for detecting mirror symmetry. defaults to 0.01.
    :return: the number of elements expected to be symmetric under a random reflection."""
    n = len(arr)
    minv = arr[0]
    maxv = arr[-1]
    return n * (n - 1) * tolerance / (maxv - minv)

def find_initial_b_ion(
    spectrum, 
    lo,
    hi,
    center: float,
):
    # starting at the pivot, scan the upper half of the spectrum
    for i in range(lo, hi):
        corrected_mass = reflect(spectrum[i], center) - ION_OFFSET_LOOKUP['b']
        residue = residue_lookup(corrected_mass)
        if residue != 'X':
            yield i, residue

def find_terminal_y_ion(
    spectrum, 
    hi,
):
    # starting at the pivot, scan the lower half of the spectrum
    for i in range(hi - 1, -1, -1):
        corrected_mass = spectrum[i] - ION_OFFSET_LOOKUP['y']
        residue = residue_lookup(corrected_mass)
        if residue != 'X':
            yield i, residue

#=============================================================================#
# disjoint pairs

def _construct_membership_table(
    X: list[set]
):
    S = set(itertools.chain.from_iterable(X))
    table = {
        s: []
        for s in S
    }
    n = len(X)
    for i in range(n):
        for elt in X[i]:
            table[elt].append(i)
    return table

def _find_disjoint(
    x: set,
    n: int,
    membership_table: dict
):
    disjoint = [
        True 
        for _ in range(n)
    ]
    for elt in x:
        for set_idx in membership_table[elt]:
            disjoint[set_idx] = False
    return [
        i 
        for i in range(n) 
        if disjoint[i]
    ]

def _table_disjoint_pairs(
    X: list[set]
):
    membership_table = _construct_membership_table(X)
    n = len(X)
    for i in range(n):
        for j in _find_disjoint(X[i], n, membership_table):
            if j > i:
                yield (i,j)

def _naiive_disjoint_pairs(
    X: list[set]
):
    n = len(X)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if X[i].isdisjoint(X[j])]

def disjoint_pairs(
    X: list[set],
    mode = "table"
):
    "associates sets that do not share elements."
    if mode == "table":
        return list(_table_disjoint_pairs(X))
    elif mode == "naiive":
        return _naiive_disjoint_pairs(X)
    else:
        raise ValueError(f"unknown pairing mode {mode}. supported modes are [\"table\", \"naiive\"]")