from typing import Iterator

import numpy as np

from .types import MzArray, PeakList, AnnotatedPeakList
from ..residues.types import MassTransformation

def prepare_spectrum():
    """not yet implemented"""
    pass

def detect_peaks():
    """not yet implemented"""
    pass

def annotate_peaks(
    peaks: PeakList,
    transformations: Iterator[MassTransformation],
) -> AnnotatedPeakList:
    """Apply a collection of MassTransformation objects to annotate a PeakList,
    assigning to each peak a list of potential charge and loss states, and determining 
    whether any of a peak's annotations are compatible with one another."""
    n = len(peaks)
    # construct the annotations
    charges = [list() for _ in range(n)]
    losses = [list() for _ in range(n)]
    outgoing_states = [set() for _ in range(n)]
    incoming_states = [set() for _ in range(n)]
    for mt in transformations:
        left_idx, right_idx = mt.peaks_index
        outgoing_states[left_idx].add(mt.left_state())
        incoming_states[right_idx].add(mt.right_state())
        charges[left_idx].append(mt.charges_symbol[0])
        charges[right_idx].append(mt.charges_symbol[1])
        losses[left_idx].append(mt.losses_symbol[0])
        losses[right_idx].append(mt.losses_symbol[1])
    # determine peak-wise consistency
    consistency = [0 for _ in range(n)]
    for i in range(n):
        consistency[i] = len(incoming_states[i].intersection(outgoing_states[i]))
    # done
    return AnnotatedPeakList(
        mz = peaks.mz,
        intensity = peaks.intensity,
        charge = charges,
        losses = losses,
        metadata = {"consistency": consistency})