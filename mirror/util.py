import itertools
from itertools import product

import pyopenms as oms
import numpy as np
from tqdm import tqdm

from .types import Iterable, UNKNOWN_RESIDUE, TERMINAL_RESIDUES, NONTERMINAL_RESIDUES, ION_SERIES, ION_SERIES_OFFSETS, ION_OFFSET_LOOKUP, AVERAGE_MASS_DIFFERENCE, LOOKUP_TOLERANCE, GAP_TOLERANCE, INTERGAP_TOLERANCE, BOUNDARY_PADDING

from .gaps.gap_types import DEFAULT_GAP_SEARCH_PARAMETERS, RESIDUES, MONO_MASSES, MASSES, RESIDUE_MONO_MASSES, RESIDUE_MASSES, LOSS_WATER, LOSS_AMMONIA, LOSSES, RESIDUE_LOSSES, MOD_Methionine_Sulfone, MOD_Methionine_Sulfoxide, MOD_PhosphoSerine, MODIFICATIONS, RESIDUE_MODIFICATIONS, CHARGES

#=============================================================================#
# residue constants and functions

MAX_MASS = max(MONO_MASSES.max(), MASSES.max()) + LOSSES.max() + MODIFICATIONS.max() + LOOKUP_TOLERANCE

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
        return "L"
    else:
        return res

def mass_error(
    mass: float,
    mass_table: np.ndarray = MONO_MASSES,
):
    "The least difference between `mass: float` and `mass table: list[float]`. Mass table defaults to AMINO_MASS_MONO."
    return min([abs(m - mass) for m in mass_table])

def residue_lookup(
    gap: float,
    letters: np.ndarray = RESIDUES,
    mass_table: list[float] = MONO_MASSES,
    tolerance: float = LOOKUP_TOLERANCE,
    unknown = UNKNOWN_RESIDUE,
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

def horizontal_panes(a_repr: str, b_repr: str):
    a_lines = a_repr.split('\n')
    b_lines = b_repr.split('\n')
    
    # vertically pad line lists
    max_num_lines = max(len(a_lines), len(b_lines))
    for lines in (a_lines, b_lines):
        for i in range(max_num_lines - len(lines)):
            lines.append("")

    # horizontally pad (ljust) each line
    max_line_len = lambda L: max([len(l) for l in L])
    a_max = max_line_len(a_lines)
    b_max = max_line_len(b_lines)
    graph_pair_aii = '\n'.join([f"[ {a_line.ljust(a_max)}  ][ {b_line.ljust(b_max)}  ]" for (a_line, b_line) in zip(a_lines, b_lines)])
    
    # render borders
    make_border = lambda n: '[' + ('=' * (n + 3)) + ']'
    border = make_border(a_max) + make_border(b_max)
    
    return '\n'.join([border, graph_pair_aii, border])

def get_respectful_printer(args):
    def print_respectfully(msg, verbosity_level, arg_verbosity = args.verbosity):
        if arg_verbosity >= verbosity_level:
            print(msg)
    return print_respectfully

def plot_hist(x: list[int], header: str = None, width = 60):
    """Renders a histogram for a list of integers into the standard output.
    
    :x: a list of integers.
    :width: the maximum width of the histogram, in characters. Defaults to 60."""
    n_inf = len([v for v in x if v == np.inf])
    x = [int(v) for v in x if v != np.inf]

    print('=' * (width + 20))
    if header != None:
        print(header)
        print('-' * (width + 20))
    if len(x) > 0:
        left = min(x)
        right = max(x)
        vals = list(range(left, right + 1))
        counts = [0 for _ in range(left, right + 1)]
        for pt in x:
            counts[pt - left] += 1
        
        lo = min(counts)
        hi = max(n_inf, max(counts))
        rng = hi - lo + 1
        for i in range(len(counts)):
            frac = (counts[i] - lo) / rng
            bars = int(width * frac)
            if counts[i] > 0:
                bars = max(bars, 1)
            print(f"{vals[i]}\t|" + "o" * bars + f"  ({counts[i]})")
        if n_inf > 0:
            inf_frac = (n_inf - lo) / rng
            inf_bars = max(int(width * inf_frac), 1)
            print(f"\t|\n∞\t|" + "o" * inf_bars + f"  ({n_inf})")
    else:
        print(f"∞\t|" + "o" * (width // 2) + f"  ({n_inf})")
    print('=' * (width + 20))

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

def _recursive_collapse(items, index, count):
    if isinstance(items, Iterable):
        new_items = []
        new_index = []
        for x in items:
            new_x, new_i, count = _recursive_collapse(x, index, count)
            new_items.append(new_x)
            new_index.append(new_i)
        return collapse_second_order_list(new_items), new_index, count
    else:
        count += 1
        return [items], count, count

def recursive_collapse(items):
    return _recursive_collapse(items, [], -1)[:2]

def recursive_uncollapse(flat_items, index):
    if isinstance(index, Iterable):
        items = []
        for subindex in index:
            new_item = recursive_uncollapse(flat_items, subindex)
            items.append(new_item)
        return items
    else:
        return flat_items[index]

def test_collapse():
    x1 = 0
    x2 = [0,1,2,3,4,5]
    x3 = [[0,1],[2,3,4],[5]]
    x4 = [[[0,1],[2]],[[3,4],[5]]]
    X = [x1,x2,x3,x4]
    for x in X:
        flat_x, index = recursive_collapse(x)
        new_x = recursive_uncollapse(flat_x, index)
        print(f"{x} ↦ {flat_x} ↦ {new_x}\n")
        assert x == new_x

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

def unique(arr):
    return list(set(arr))

#=============================================================================#
# simulation via pyOpenMS

def default_param():
    """Creates a pyopenms.Param() object with add_b_ions, add_y_ions, and add_metadata set to true."""
    param = oms.Param()
    param.setValue("add_b_ions", "true")
    param.setValue("add_y_ions", "true")
    param.setValue("add_metainfo", "true")
    return param

DEFAULT_PARAM = default_param()

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

ADVANCED_PARAM = advanced_param()

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
    return generate_fragment_spectrum(seq, param = DEFAULT_PARAM)

def list_mz(spec: oms.MSSpectrum):
    """Creates a numpy array of the peaks of a pyopenms.MSSpectrum object.
    
        np.array([peak.getMZ() for peak in spec])
    
    :spec: a pyopenms.MSSpectrum object."""
    return np.array([peak.getMZ() for peak in spec])

def list_intensity(spec: oms.MSSpectrum):
    """Creates a numpy array of the intensities of a pyopenms.MSSpectrum object.
    
        np.array([peak.getIntensity() for peak in spec])
    
    :spec: a pyopenms.MSSpectrum object."""
    return np.array([peak.getIntensity() for peak in spec])

def simulate_peaks(seq: str, param = DEFAULT_PARAM):
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
    try:
        oms_seq = oms.AASequence.fromString(seq)
    except Exception as e:
        print(seq)
        raise e
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

def expected_pivot_center(seq: str):
    """Estimate the center of mirror symmetry of the spectrum of a sequence, using the simulated
    b- and y-ion series generated by `get_b_ion_series`, `get_y_ion_series`.

        np.mean([*get_b_ion_series(seq)[0:2], *get_y_ion_series(seq)[-3:-1]])
    
    :seq: a str."""
    b_series = get_b_ion_series(seq)
    y_series = get_y_ion_series(seq)
    symmetric_peaks = [*b_series[0:2],*y_series[-3:-1]]
    return np.mean(symmetric_peaks)

def reflect(x, center: float):
    """Reflect a numeric type about a given point. Reflection is performed in-place. 
    If passed a one-dimensional array, the array will not be reversed!

        2 * center - x
    
    :param x: the number or numeric numpy array subject to reflection.
    :param center: the center of reflection."""
    return 2 * center - x

def reflect_spectrum(spectrum: oms.MSSpectrum, center: float):
    """Reflect the peaks of a pyopenms.MSSpectrum type about a given point.
    
    :param spectrum: oms.MSSpectrum object.
    :param center: the center of reflection."""
    reflected_spectrum = oms.MSSpectrum()
    reflected_spectrum.set_peaks([
        reflect(list_mz(spectrum), center),
        list_intensity(spectrum)
    ])
    reflected_spectrum.sortByPosition()
    return reflected_spectrum

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

def _generate_extremal_ions(
    spectrum: np.ndarray,
    index_range: Iterable,
    mass_transform,
    max_query_mass = MAX_MASS
):
    for i in index_range:
        corrected_mass = mass_transform(spectrum[i])
        residue = residue_lookup(corrected_mass)
        #print(f"index {i} query {spectrum[i]} corrected {corrected_mass} residue {residue} MAX MASS {max_query_mass}")
        if residue != 'X':
            #print("\tmatch!")
            yield i, residue
        #elif corrected_mass > MAX_MASS:
        #    #print("\tbreak!")
        #    break

def _get_b_ion_transform(center):
    def _b_ion_transform(mz):
        return reflect(mz, center) - ION_OFFSET_LOOKUP['b']
    return _b_ion_transform

def _y_ion_transform(mz):
    return mz - ION_OFFSET_LOOKUP['y']

def find_initial_b_ion(
    spectrum, 
    lo,
    hi,
    center: float,
):
    """Returns a generator of (index, residue) pairs. Each index corresponds to an m/z value 
    (peak) in the spectrum, which, upon reflection around a center, and translation by the
    typical b ion offset, matches the residue mass.
    
    :param spectrum: a sorted one-dimensional numeric array.
    :lo: the smallest index to consider.
    :hi: the largest index to consider.
    :center: the point around which putative peaks are reflected."""
    # starting at the pivot, scan the upper half of the spectrum
    return _generate_extremal_ions(
        spectrum, 
        range(hi - 1, lo, -1), 
        _get_b_ion_transform(center),
    )

def find_terminal_y_ion(
    spectrum, 
    hi,
):
    """Returns a generator of (index, residue) pairs. Each index corresponds to an m/z value 
    (peak) in the spectrum which, upon translation by the typical y ion offset, matches the 
    residue mass.
    
    :param spectrum: a sorted one-dimensional numeric array.
    :hi: the largest index to consider. Iteration descends from this value."""
    # starting at the pivot, scan the lower half of the spectrum
    return _generate_extremal_ions(
        spectrum,
        range(hi),
        _y_ion_transform,
    )

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