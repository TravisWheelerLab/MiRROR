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
    for mt in transformations:
        l, r = mt.peaks
        l_charge, r_charge = mt.charges
        l_loss, r_loss = mt.losses
        charges[l].append(l_charge)
        losses[l].append(l_loss)
        charges[r].append(r_charge)
        losses[r].append(r_loss)
    # determine peak-wise consistency
    consistency = [False for _ in range(n)]
    for i in range(n):
        for data in (charges[i], losses[i]):
            consistency[i] = len(set(data)) < len(data)
    # done
    return AnnotatedPeakList(
        mz = peaks.mz,
        intensity = peaks.intensity,
        charge = charges,
        losses = losses,
        metadata = {"consistency": consistency})