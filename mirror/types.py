from numpy import ndarray
from pyopenms import MSExperiment
from networkx import DiGraph

from collections.abc import Iterator
from typing import Any, Iterable

from dataclasses import dataclass

from .gaps.gap_types import *

#=============================================================================#

UNKNOWN_RESIDUE = 'X'

TERMINAL_RESIDUES = np.array(['R', 'K'])

NONTERMINAL_RESIDUES = np.array([r for r in RESIDUES if r not in TERMINAL_RESIDUES])

ION_SERIES = [
    'a',
    'b',
    'c',
    'x',
    'y',
    'z',
]

ION_SERIES_OFFSETS = [
    -27,
    1,
    18,
    45,
    19,
    2,
]

ION_OFFSET_LOOKUP = dict(zip(ION_SERIES,ION_SERIES_OFFSETS))

AVERAGE_MASS_DIFFERENCE = np.mean(np.abs(MASSES - MONO_MASSES))

LOOKUP_TOLERANCE = 0.1
GAP_TOLERANCE = 0.01
INTERGAP_TOLERANCE = GAP_TOLERANCE * 2 

BOUNDARY_PADDING = 3

Edge = tuple[int,int]
SingularPath = list[int]
DualPath = list[tuple[int,int]]
GraphPair = tuple[DiGraph, DiGraph]

@dataclass
class OutputIndex:
    pivot_index: int
    boundary_index: int
    affixes_index: int
    candidate_index: int

    def __iter__(self):
        return iter((self.pivot_index, self.boundary_index, self.affixes_index, self.candidate_index))

@dataclass
class RunParameters:
    gap_search_parameters: GapSearchParameters
    pivot_tolerance: float
    symmetry_factor: float
    terminal_residues: list[str]
    boundary_padding: int
    gap_key: str = "gap"

DEFAULT_RUN_PARAMETERS = RunParameters(
    DEFAULT_GAP_SEARCH_PARAMETERS,
    INTERGAP_TOLERANCE,
    1.0,
    TERMINAL_RESIDUES,
    BOUNDARY_PADDING
)

@dataclass
class TestParameters:
    run_params: RunParameters
    num_residues: int
    num_losses: int
    num_charged: int
    num_modifications: int
    num_noise: int
    masses = MASSES
    residues = RESIDUES
    losses = LOSSES
    modifications = MODIFICATIONS
    charges = CHARGES