from dataclasses import dataclass

@dataclass
class MassTransformation:
    """A collection of indices describing the loss, charge, and modification
    states of a pair of peaks whose difference corresponds to a transformation
    of a residue mass. Other than the residue index, values of 0 imply the null
    state, e.g., charge +1, no neutral losses, or no modification."""
    residue: int
    modification: int
    inner_index: tuple[int, int]
    peaks: tuple[int, int]
    losses: tuple[int, int]
    charges: tuple[int, int]