import itertools
from itertools import product

import pyopenms as oms
import numpy as np
from tqdm import tqdm

# the util module is an initial object; it cannot have any local dependencies.

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

BOUNDARY_PADDING = 3

def generate_random_residues(length: int, alphabet = RESIDUES):
    """
        np.random.choice(alphabet, length, replace=True)"""
    return np.random.choice(alphabet, length, replace=True)

def generate_random_peptide(length: int, alphabet = RESIDUES):
    "Generates a random string of given length from the alphabet, which defaults to RESIDUES."
    return ''.join(generate_random_residues(length, alphabet))

def generate_random_tryptic_peptide(length: int):
    "Creates a random string of given length from NONTERMINAL_RESIDUES, then appends a TERMINAL_RESIDUE suffix."
    terminus = np.random.choice(TERMINAL_RESIDUES)
    prefix = generate_random_peptide(length - 1, alphabet = NONTERMINAL_RESIDUES)
    return prefix + terminus

def mask_ambiguous_residues(res: chr):
    "Maps residues \'L\' and \'I\' to \"I/L\"."
    if res == "L" or res == "I":
        return "I/L"
    else:
        return res

def mass_error(
    mass: float,
    mass_table: list[float] = AMINO_MASS_MONO,
):
    "The least difference between `mass: float` and `mass table: list[float]`. Mass table defaults to AMINO_MASS_MONO."
    return min([abs(m - mass) for m in mass_table])

def residue_lookup(
    gap: float,
    letters: list[str] = AMINOS,
    mass_table: list[float] = AMINO_MASS_MONO,
    tolerance: float = LOOKUP_TOLERANCE,
    unknown = UNKNOWN_AMINO,
    nan = '.',
):
    """Associates a gap value to the best matching amino acid residue.
    
    :gap: a float.
    :letters: the list of letters that may be output. Defaults to AMINOS.
    :mass_table: the list of masses. Indexed by the same set as letters. Defaults to AMINO_MASS_MONO.
    :tolerance: a float; the maximum permissible difference between the gap and the best match.
    :unknown: the return value for gaps whose best match differs by more than the tolerance, or gaps == -1.
    :nan: the value returned for gaps == -2."""
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

def plot_hist(x: list[int], width = 60):
    """Renders a histogram for a list of integers into the standard output.
    
    :x: a list of integers.
    :width: the maximum width of the histogram, in characters. Defaults to 60."""
    left = min(x)
    right = max(x)
    vals = list(range(left, right + 1))
    counts = [0 for _ in range(left, right + 1)]
    for pt in x:
        counts[pt - left] += 1
    
    lo = min(counts)
    hi = max(counts)
    rng = hi - lo + 1
    print('-' * (width + 20))
    for i in range(len(counts)):
        frac = (counts[i] - lo) / rng
        bars = int(width * frac)
        if counts[i] > 0:
            bars = max(bars, 1)
        print(f"{vals[i]}\t|" + "o" * bars + f"  ({counts[i]})")
    print('-' * (width + 20))

def comma_separated(items):
    """
        ','.join(map(str, items))"""
    return ','.join(map(str, items))

def split_commas(commastr, target_type):
    """Splits a comma-separated string and converts the parts to a given type.
    
    :commastr: a comma-separated string.
    :target_type: a callable; the type into which the split string components will be converted."""
    parts = commastr.split(',')
    try:
        return list(map(target_type, parts))
    except ValueError:
        return []
    except Exception as e:
        raise e

def collapse_second_order_list(llist: list[list]):
    """Associates a list of lists of elements to a flat list of elements.
    
        list(itertools.chain.from_iterable(llist))

    :llist: a second-order iterable."""
    return list(itertools.chain.from_iterable(llist))

def log(message, prefix=""):
    print(prefix + f"⚙\t{message}")

def add_tqdm(inputs, total=None, description=None):
    """Wraps an iterator with a tqdm progress bar.
    
    :inputs: an iterable.
    :total: the number of elements in the iterable. This argument is non-optional if `inputs` does not implement __len__!
    :description: passed to the `desc` field of the tqdm object."""
    if total == None:
        total = len(inputs)
    return tqdm(inputs, total=total, leave=False, desc=description)

#=============================================================================#
# simulation via pyOpenMS

def default_param():
    """Creates a pyopenms.Param() object with add_b_ions, add_y_ions, and add_metadata set to true."""
    param = oms.Param()
    param.setValue("add_b_ions", "true")
    param.setValue("add_y_ions", "true")
    param.setValue("add_metainfo", "true")
    return param

def advanced_param():
    """Creates a pyopenms.Param() object with add_b_ions, add_y_ions, add_metadata,
    add_all_precursor_charges, and add_losses set to true."""
    param = oms.Param()
    param.setValue("add_b_ions", "true")
    param.setValue("add_y_ions", "true")
    param.setValue("add_all_precursor_charges", "true")
    param.setValue("add_losses", "true")
    param.setValue("add_metainfo", "true")
    return param

def generate_fragment_spectrum(seq: str, param: oms.Param):
    """From a string and a pyopenms.Param() object, uses a pyopenms.TheoreticalSpectrumGenerator
    object to create a simulated fragment spectrum as a pyopenms.MSSpectrum object."""
    tsg = oms.TheoreticalSpectrumGenerator()
    spec = oms.MSSpectrum()
    peptide = oms.AASequence.fromString(seq)
    tsg.setParameters(param)
    tsg.getSpectrum(spec, peptide, 1, 1)
    return spec

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

def generate_default_fragment_spectrum(seq: str):
    """Creates a pyopenms.MSSpectrum object from a string 
    with the default parametization from default_param."""
    return generate_fragment_spectrum(
        seq, param = default_param())

def list_mz(spec: oms.MSSpectrum):
    """Creates a numpy array of the peaks of a pyopenms.MSSpectrum object.
    
        np.array([peak.getMZ() for peak in spec])
    
    :spec: a pyopenms.MSSpectrum object."""
    return np.array([peak.getMZ() for peak in spec])

def list_intensity(spec: oms.MSSpectrum):
    """Creates a numpy array of the intensities of a pyopenms.MSSpectrum object.
    
        NOT IMPLEMENTED!
    
    :spec: a pyopenms.MSSpectrum object."""
    pass

def simulate_peaks(seq: str, param = default_param()):
    """Associate a sequence to a numpy array of its b and y ions.
    
    :seq: a str.
    :param: a pyopenms.Param() object, to be passed to a pyopenms.TheoreticalSpectrumGenerator."""
    return list_mz(generate_fragment_spectrum(seq, param))

def digest_trypsin(seq: str, minimum_length: int = 7, maximum_length: int = 50):
    """Uses OpenMS ProteaseDigestion interface to generate tryptic peptides from a sequence.
    
    :seq: a str.
    :minimum length: the smallest viable tryptic peptide. Defaults to 7.
    :maximum length: the largest viable tryptic peptide. Defaults to 50."""
    dig = oms.ProteaseDigestion()
    result = []
    oms_seq = oms.AASequence.fromString(seq)
    dig.digest(oms_seq, result, minimum_length, maximum_length)
    return [r.toString() for r in result]

def enumerate_tryptic_peptides(sequences, minimum_length: int = 7, maximum_length: int = 50):
    """Lazily iterates the tryptic peptides of a list of sequences.
    
    :sequences: an iterable of str types.
    :minimum length: the smallest viable tryptic peptide. Defaults to 7.
    :maximum length: the largest viable tryptic peptide. Defaults to 50."""
    dig = oms.ProteaseDigestion()
    for seq in sequences:
        for pep in digest_trypsin(seq, minimum_length, maximum_length):
            yield pep

def enumerate_tryptic_spectra(sequences, minimum_length: int = 7, maximum_length: int = 50):
    """Return a lazy iterable of lazy iterators of the tryptic peptides of each sequence.
    Can be collected by collapse_second_order_list, but this is not advised as the size
    of the fully enumerated set may be intractible.
    
    :sequences: an iterable of str types.
    :minimum length: the smallest viable tryptic peptide. Defaults to 7.
    :maximum length: the largest viable tryptic peptide. Defaults to 50."""
    return map(simulate_peaks, enumerate_tryptic_peptides(sequences, minimum_length, maximum_length))

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
    """Generates the list of (index, residue) pairs. Each index corresponds to an m/z value 
    (peak) in the spectrum, which, upon reflection around a center, and translation by the
    typical b ion offset, matches the residue mass.
    
    :param spectrum: a sorted one-dimensional numeric array.
    :lo: the smallest index to consider.
    :hi: the largest index to consider.
    :center: the point around which putative peaks are reflected."""
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
    """Generates the list of (index, residue) pairs. Each index corresponds to an m/z value 
    (peak) in the spectrum which, upon translation by the typical y ion offset, matches the 
    residue mass.
    
    :param spectrum: a sorted one-dimensional numeric array.
    :hi: the largest index to consider. Iteration descends from this value."""
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
    """Creates a dictionary that associates each element in each set in X to the 
    index of the sets in which it is contained."""
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
    """Given a set x, the number of sets in the superset, and the membership table representing
    the superset, list the indices of the sets in the superset that do not intersect x.
    
    TODO: i think the last couple lines can be replaced by "yield set_idx"?"""
    disjoint = [True for _ in range(n)]
    for elt in x:
        for set_idx in membership_table[elt]:
            disjoint[set_idx] = False
    return [i for i in range(n) if disjoint[i]]

def _table_disjoint_pairs(
    X: list[set]
):
    "Implements the \"table\" mode of `disjoint_pairs`."
    membership_table = _construct_membership_table(X)
    n = len(X)
    for i in range(n):
        for j in _find_disjoint(X[i], n, membership_table):
            if j > i:
                yield (i,j)

def _naiive_disjoint_pairs(
    X: list[set]
):
    "Implements the \"naiive\" mode of `disjoint_pairs`."
    n = len(X)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if X[i].isdisjoint(X[j])]

def disjoint_pairs(
    X: list[set],
    mode = "table"
):
    """Returns the list of pairs (i,j) of indexes into X such that X[i] and X[j] do not share any
    elements.
    
    :X: a list of sets, elsewhere referred to as the superset.
    :mode: a string, specifies the internal routine by which pairs are identified. Supported modes 
    are \"table\" and \"naiive\". Table is generally faster."""
    if mode == "table":
        return list(_table_disjoint_pairs(X))
    elif mode == "naiive":
        return _naiive_disjoint_pairs(X)
    else:
        raise ValueError(f"unknown pairing mode {mode}. supported modes are [\"table\", \"naiive\"]")