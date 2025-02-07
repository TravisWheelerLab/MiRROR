import numpy as np
import pyopenms as oms

#=============================================================================#
# OpenMS implements methods for aligning spectra. taken with the synthetic 
# spectra generators, we can use these tools to measure the alignment of a 
# candidate sequence to a reference spectrum.

def score_sequence_to_spectrum(
    sequence,
    peaks,
    *params
):
    return 0