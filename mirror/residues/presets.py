from .types import ResidueParams, BisectMassTransformationSolver

RESIDUES = [
    'A',
    'R',
    'N',
    'D',
    'C',
    'E',
    'Q',
    'G',
    'H',
    'L',
    'I',
    'K',
    'M',
    'F',
    'P',
    'S',
    'T',
    'W',
    'Y',
    'V']

MONO_MASSES = [
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

RESIDUE_MONO_MASSES = dict(zip(RESIDUES, MONO_MASSES))

MASSES = [
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

RESIDUE_MASSES = dict(zip(RESIDUES, MASSES))

LOSS_WATER = -18
LOSS_AMMONIA = -17
LOSSES = [LOSS_WATER, LOSS_AMMONIA]
LOSS_SYMBOLS = ["H2O", "NH3"]

RESIDUE_LOSSES = {
    'A' : [],
    'R' : [0],
    'N' : [0],
    'D' : [],
    'C' : [],
    'E' : [1],
    'Q' : [0],
    'G' : [],
    'H' : [],
    'L' : [],
    'I' : [],
    'K' : [0],
    'M' : [],
    'F' : [],
    'P' : [],
    'S' : [1],
    'T' : [1],
    'W' : [],
    'Y' : [],
    'V' : []}

MOD_PhosphoSerine = 97.9769
MOD_Methionine_Sulfoxide = 15.9949
MOD_Methionine_Sulfone = 31.9898

MODIFICATIONS = [
    MOD_PhosphoSerine,
    MOD_Methionine_Sulfoxide,
    MOD_Methionine_Sulfone]
MODIFICATION_SYMBOLS = [
    "SEP",
    "MetO",
    "MetO2"]

RESIDUE_MODIFICATIONS = {
    'A' : [],
    'R' : [],
    'N' : [],
    'D' : [],
    'C' : [],
    'E' : [],
    'Q' : [],
    'G' : [],
    'H' : [],
    'L' : [],
    'I' : [],
    'K' : [],
    'M' : [2, 1],
    'F' : [],
    'P' : [],
    'S' : [0],
    'T' : [],
    'W' : [],
    'Y' : [],
    'V' : []}

CHARGES = [
    1.,
    2.,
    3.]
CHARGE_SYMBOLS = [
    "+1",
    "+2",
    "+3"]

DEFAULT_RESIDUE_PARAMS = ResidueParams(
    tolerance = 0.01,
    comparative_tolerance = 0.02,
    strategy = BisectMassTransformationSolver,
    charge_symbols = CHARGE_SYMBOLS,
    charge_states = CHARGES,
    loss_symbols = LOSS_SYMBOLS,
    loss_masses = LOSSES,
    modification_symbols = MODIFICATION_SYMBOLS,
    modification_masses = MODIFICATIONS,
    residue_symbols = RESIDUES,
    residue_masses = MONO_MASSES,
    residue_losses = RESIDUE_LOSSES,
    residue_modifications = RESIDUE_MODIFICATIONS)