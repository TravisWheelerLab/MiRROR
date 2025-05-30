RESIDUES = np.array([
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
    'V',
])

MONO_MASSES = np.array([
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
    99.068,
])

RESIDUE_MONO_MASSES = dict(zip(RESIDUES, MONO_MASSES))

MASSES = np.array([
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
    99.13,
])

RESIDUE_MASSES = dict(zip(RESIDUES, MASSES))

LOSS_WATER = -18
LOSS_AMMONIA = -17
LOSSES = np.array([LOSS_WATER, LOSS_AMMONIA])

RESIDUE_LOSSES = {
    'A' : [],
    'R' : [LOSS_WATER],
    'N' : [LOSS_WATER],
    'D' : [],
    'C' : [],
    'E' : [LOSS_AMMONIA],
    'Q' : [LOSS_WATER],
    'G' : [],
    'H' : [],
    'L' : [],
    'I' : [],
    'K' : [LOSS_WATER],
    'M' : [],
    'F' : [],
    'P' : [],
    'S' : [LOSS_AMMONIA],
    'T' : [LOSS_AMMONIA],
    'W' : [],
    'Y' : [],
    'V' : [],
}

MOD_PhosphoSerine = 97.9769
MOD_Methionine_Sulfoxide = 15.9949
MOD_Methionine_Sulfone = 31.9898

MODIFICATIONS = np.array([
    MOD_PhosphoSerine,
    MOD_Methionine_Sulfoxide,
    MOD_Methionine_Sulfone,
])

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
    'M' : [MOD_Methionine_Sulfone, MOD_Methionine_Sulfoxide],
    'F' : [],
    'P' : [],
    'S' : [MOD_PhosphoSerine],
    'T' : [],
    'W' : [],
    'Y' : [],
    'V' : [],
}

CHARGES = np.array([
    2.,
    3.,
])