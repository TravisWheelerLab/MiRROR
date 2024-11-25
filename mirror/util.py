import itertools

import pyopenms as oms
import numpy as np
from Bio import Seq, SeqRecord, SeqIO
from tqdm import tqdm

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

GAP_TOLERANCE = min(abs(m1 - m2) for m1 in AMINO_MASS_MONO for m2 in AMINO_MASS_MONO if abs(m1 - m2) > 0)
INTERGAP_TOLERANCE = GAP_TOLERANCE / 2

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
    tolerance: float = GAP_TOLERANCE,
    unknown = UNKNOWN_AMINO,
):
    if gap == -1:
        return unknown
    dif = [abs(m - gap) for m in mass_table]
    i = np.argmin(dif)
    optimum = dif[i]
    if optimum > tolerance: # no match
        return unknown
    else:
        return mask_ambiguous_residues(letters[i])

def collapse_second_order_list(llist: list[list]):
    return list(itertools.chain.from_iterable(llist))

def log(message, prefix=""):
    print(prefix + f"⚙\t{message}")

def add_tqdm(inputs, total=None, description=None):
    if total == None:
        total = len(inputs)
    return tqdm(inputs, total=total, leave=False, desc=description)

def load_fasta_records(path_to_fasta: str):
    records = []
    with open(path_to_fasta) as handle:
        records = list(SeqIO.parse(handle, "fasta"))
    return records

def save_strings_to_fasta(path_to_fasta: str, seqs: list[str]):
    n = len(seqs)
    records = [SeqRecord.SeqRecord(Seq.Seq(seqs[i]), id=f"miss_{i}", name="", description="") for i in range(n)]
    with open(path_to_fasta, "w") as handle:
        return SeqIO.write(records, handle, "fasta")

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

def reflect(x, center: float):
    return 2 * center - x

def count_mirror_symmetries(arr: np.array, center: float, tolerance = 0.01):
    reflected_arr = reflect(arr, center)
    n_symmetric = 0
    for reflected_val in reflected_arr:
        if np.min(np.abs(arr - reflected_val)) < tolerance:
            n_symmetric += 1
    return n_symmetric