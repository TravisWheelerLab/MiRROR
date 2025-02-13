import numpy as np
import pyopenms as oms

from .util import reflect_spectrum, expected_pivot_center, generate_fragment_spectrum, DEFAULT_PARAM, ADVANCED_PARAM

#=============================================================================#
# OpenMS implements methods for aligning spectra. taken with the synthetic 
# spectra generators, we can use these tools to measure the alignment of a 
# candidate sequence to a reference spectrum.

def default_aligner(tolerance = 0.1):
    "The default parametization of an oms.SpectrumAlignment object."
    aln = oms.SpectrumAlignment()
    p = aln.getParameters()
    p.setValue("tolerance", tolerance)
    p.setValue("is_relative_tolerance", "false")
    aln.setParameters(p)
    return aln

DEFAULT_ALIGNER = default_aligner()

def _align(
    aligner: oms.SpectrumAlignment,
    target: oms.MSSpectrum,
    query: oms.MSSpectrum
):
    peak_pairs = []
    aligner.getSpectrumAlignment(peak_pairs, target, query)
    return peak_pairs

def align_sequence_to_spectrum(
    aligner: oms.SpectrumAlignment,
    target_spectrum: oms.MSSpectrum,
    query_seq: str,
    param: oms.Param = DEFAULT_PARAM,
):
    """Align a query sequence's generated fragment spectrum to a given target spectrum, using the oms.SpectrumAlignment interface.

    :aligner: an oms.SpectrumAlignment object, which performs alignment.
    :target_spectrum: an oms.MSSpectrum object, which is the alignment target.
    :query_seq: a str, used to create the spectrum alignment query.
    :param: the oms.Param object used to parametize the fragment spectrum generator. Defaults to mirror.util.DEFAULT_PARAM."""
    query_spectrum = generate_fragment_spectrum(query_str, param)
    return _align(aligner, target_spectrum, query_spectrum)

def align_to_reflection(
    aligner: oms.SpectrumAlignment,
    target_spectrum: oms.MSSpectrum,
    center: float,
):
    """Align a spectrum to its reflection using the oms.SpectrumAlignment interface.
    
    :aligner: an oms.SpectrumAlignment object, which performs alignment.
    :target_spectrum: an oms.MSSpectrum object, which is the alignment target.
    :center: the given center of mirror symmetry, about which the peaks of `target_spectrum` will be reflected to create the query spectrum."""
    query_spectrum = reflect_spectrum(target_spectrum, center)
    return _align(aligner, target_spectrum, query_spectrum)

def score_sequence_to_spectrum(
    sequence,
    peaks,
    *params
):
    return 0