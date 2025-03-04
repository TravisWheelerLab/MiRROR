import numpy as np
from typing import Type
from dataclasses import dataclass
from collections.abc import Iterator
from abc import ABC, abstractmethod

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

@dataclass
class GapAbstractTransformationSolver(ABC):
    mz: np.ndarray
    mass_tensor: np.ndarray
    left_loss_tensor: np.ndarray
    right_loss_tensor: np.ndarray
    modification_tensor: np.ndarray
    extremal_delta: float
    tolerance: float

    @abstractmethod
    def __post_init__(self) -> None:
        """stuff that needs doing before other stuff"""

    @abstractmethod
    def set_outer_index(self, i: int) -> None:
        """Sets the outer index for gap search, and performs any work needed for subsequent operations."""
    
    @abstractmethod
    def set_inner_index(self, j: int) -> None:
        """Sets the inner index for gap search, and performs any work needed to retrieve the optimal result for (left_index, right_index)."""

    @abstractmethod
    def get_solutions(self) -> tuple[int, int, int, int]:
        """Having set outer and inner indices as i, and j, respectively, 
        determine the 4-tuple (mass_idx, left_loss_idx, right_loss_idx, modification_idx)
        which optimally decomposes the gap value self.mz[j] - self.mz[self._i] as
        self.mass_tensor[mass_idx] - self.left_loss_tensor[left_loss_idx] + self.right_loss_tensor[right_loss_indx] + self.modification_tensor[modification_idx]."""

    def _filter_solution(self, optimizers: Iterator[tuple[int, int, int, int]]) -> Iterator[tuple[int, int, int, int]]:
        return filter(lambda x: ((x[1] == 0) or (x[1] != x[2])), optimizers)

@dataclass
class GapMatch:
    index_pair: tuple[int,int]
    outer_index: tuple[int, int]
    inner_index: tuple[int,int]
    charge_state: tuple[float,float]
    match_residue: str
    match_mass: float
    match_idx: int
    query_mass: float
    left_loss: float
    left_loss_idx: int
    right_loss: float
    right_loss_idx: int
    modification: float
    modification_idx: int

    def indices(self):
        return np.array([
            *self.index_pair,       # 0,1
            *self.outer_index,      # 2,3
            *self.inner_index,      # 4,5
            self.match_idx,         # 6
            self.left_loss_idx,     # 7
            self.right_loss_idx,    # 8
            self.modification_idx,  # 9
        ])
    
    def values(self):
        return np.array([
            *self.charge_state,     # 0,1
            self.match_mass,        # 2
            self.query_mass,        # 3
            self.left_loss,         # 4
            self.right_loss,        # 5
            self.modification,      # 6
        ])

    def _state(self):
        return [self.left_loss_idx != -1,
            self.right_loss_idx != -1,
            self.modification_idx != -1,
            self.charge_state[0] != 1,
            self.charge_state[1] != 1,
            self.match_idx == -1,
            self.index_pair[0] == self.index_pair[1],
        ]

    def cost(self):
        *features, legibility, distinctness = self._state()
        n = len(features)
        score = sum(features) + (n * legibility) + (2 * n * distinctness)
        max_score = 2 * n
        return score / max_score

    def __repr__(self):
        return f"GapMatch(cost\t = {self.cost()}\nindex_pair\t = {self.index_pair}\ninner_index\t = {self.inner_index}\ncharge_state\t = {self.charge_state}\nmatch_residue\t = {self.match_residue}\nmatch_mass\t = {self.match_mass} ({self.match_idx})\nquery_mass\t = {self.query_mass}\nleft_loss\t = {self.left_loss} ({self.left_loss_idx})\nright_loss\t = {self.right_loss} ({self.right_loss_idx})\nmodification\t = {self.modification} ({self.modification_idx})\n)"

class GapResult:
    """Array representation of a collection of GapMatch objects. Can be constructed either from an iterable, or using the `gap_data` kwarg, from a tuple of arrays of shapes (n,10), (n,7), and (n,)."""
    def __init__(self, matches: Iterator[GapMatch], gap_data = None):
        if matches != None:
            inds = [m.indices() for m in matches]
            vals = [m.values() for m in matches]
            res = [m.match_residue for m in matches]
            self.empty = len(res) == 0
            if self.empty:
                inds = [[]]
                vals = [[]]
                res = []
            self._gap_inds = np.vstack(inds)
            self._gap_vals = np.vstack(vals)
            self._gap_res = np.array(res)
        else:
            self._gap_inds, self._gap_vals, self._gap_res = gap_data
        self._n = len(self._gap_res)
        if not self.empty:
            assert self._gap_inds.shape == (self._n,10)
            assert self._gap_vals.shape == (self._n,7)
            assert self._gap_res.shape == (self._n,)
    
    def __len__(self):
        return self._n
    
    def residue(self, i: int) -> float:
        "The matched residue of the gap at index `i`."
        return self._gap_res[i]
    
    def topological_index(self, i: int) -> tuple[int, int]:
        """Topological index pair (v, w) which guarantees: 
        1) v < w.
        2) the graph over all such pairs is a DAG.
        (Conditions 1 and 2 are more-or-less equivalent in this context.)"""
        return tuple(self._gap_inds[i, 0:2])
    
    def outer_index_pair(self, i: int) -> tuple[int,int]:
        "Indices into the original array for the gap at index `i`."
        return tuple(self._gap_inds[i, 2:4])
    
    def get_index_pairs(self) -> list[tuple[int, int]]:
        "All pairs of topological indices."
        if self.empty:
            return []
        else:
            return [self.topological_index(i) for i in range(len(self))]
    
    def inner_index_pair(self, i: int) -> tuple[int,int]:
        "Indices into the charge-duplicated array for the gap at index `i`."
        return tuple(self._gap_inds[i, 4:6])
    
    def match_index(self, i: int) -> int:
        "The match ID for the gap at index `i`."
        return self._gap_inds[i, 6]
    
    def loss_index_pair(self, i: int) -> tuple[int,int]:
        "The IDs of the loss types affecting the peaks supporting the gap at index `i`."
        return tuple(self._gap_inds[i, 7:9])
    
    def modification_index(self, i: int) -> int:
        "The ID of the modification affecting the right peak of the gap at index `i`."
        return self._gap_inds[i, 9]
    
    def charge_state(self, i: int) -> tuple[float,float]:
        "The charge states of the peaks supporting the gap at index `i`."
        return tuple(self._gap_vals[i, 0:2])

    def match_mass(self, i: int) -> float:
        "The match mass of the gap at index `i`."
        return self._gap_vals[i, 2]

    def query_mass(self, i: int) -> float:
        "The query mass of the gap at index `i`."
        return self._gap_vals[i, 3]
    
    def loss_mass_pair(self, i: int) -> tuple[float,float]:
        "The masses of the loss types affecting the peaks supporting the gap at index `i`."
        return tuple(self._gap_vals[4:6])
    
    def modification_mass(self, i: int) -> float:
        "The mass of the modification affecting the right peak of the gap at index `i`."
        return self._gap_vals[6]
    
    def cost(self, i: int) -> float:
        """The cost of the gap at index `i`, representing the gap's complexity as a weighted sum of
        the number of detected transformations and the legibility of the query mass."""
        # measure complexity of inferred transformations
        left_charge, right_charge = self.charge_state(i)
        left_loss_id, right_loss_id = self.loss_index_pair(i)
        mod_id = self.modification_index(i)
        unweighted_bools = np.array([
            left_charge != 1.0, 
            right_charge != 1.0, 
            left_loss_id != -1, 
            right_loss_id != -1, 
            mod_id != -1
        ])
        complexity = sum(unweighted_bools)
        # a gap is legigible if it's not an X
        res = self.residue(i)
        illegibility = res == 'X'
        # a gap is degenerate if its topological indices are equal, meaning that 
        # one charge duplicate has been paired to another charge duplicate of the same original peak.
        topo_inds = self.index_pair(i)
        degeneracy = topo_inds[0] == topo_inds[1]
        # compose weighted score
        n = len(unweighted_bools)
        return complexity + (n * illegibility) + (2 * n * degeneracy)
    
    def sort(self, order = None, key = None):
        if order == None and key == None:
            #raise Warning("Neither `order` nor `key` kwargs were passed; using default order via `self.cost`.")
            order = np.argsort([self.cost(i) for i in range(len(self))])
        elif key != None:
            if order != None:
                raise Warning("Both `order` and `key` kwargs were passed; `order` will be overwritten by the order created with `key`.")
            order = np.argsort([key(self[i]) for i in range(len(self))])
        self._gap_data = self._gap_data[order]

@dataclass
class GapSearchParameters:
    """TODO: store residue -> losses,modifications relational data"""
    mode: str
    residues: np.ndarray
    masses: np.ndarray
    losses: np.ndarray
    modifications: np.ndarray
    charges: np.ndarray
    tolerance: float

    def collect(self):
        return [
            self.mode, 
            self.residues, 
            self.masses, 
            self.losses, 
            self.modifications, 
            self.charges, 
            self.tolerance
        ]

DEFAULT_GAP_SEARCH_PARAMETERS = GapSearchParameters(
    "tensor",
    RESIDUES,
    MONO_MASSES,
    LOSSES,
    MODIFICATIONS,
    CHARGES,
    0.01,
)

SIMPLE_GAP_SEARCH_PARAMETERS = GapSearchParameters(
    "tensor",
    RESIDUES,
    MONO_MASSES,
    np.array([]),
    np.array([]),
    np.array([]),
    0.01,
)

UNCHARGED_GAP_SEARCH_PARAMETERS = GapSearchParameters(
    "tensor",
    RESIDUES,
    MONO_MASSES,
    LOSSES,
    MODIFICATIONS,
    np.array([]),
    0.01,
)