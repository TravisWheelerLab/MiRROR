import pathlib, os, uuid
import pytest
import numpy as np
from mirror.util import load_config
from mirror.session import setup
from mirror.sequences.queries import _query_peaks, query_by_mass, ord_to_idx

def test_query_peaks():
    peaks = np.array([0.1,0.2,0.3,0.4,0.5,0.51,0.52,0.6,0.7])
    assert _query_peaks(0.2,0.0,peaks) == 1 # exact match
    assert _query_peaks(0.11,0.01,peaks) == 0 # match w/in threshold
    assert _query_peaks(0.51,0.01,peaks) == 5 # best of 3 matches
    assert _query_peaks(0.25,0.01,peaks) is None # no matches, between vals
    assert _query_peaks(0,0.01,peaks) is None # no matches, less than all vals
    assert _query_peaks(1,0.01,peaks) is None # no matches, more than all vals

def test_query_mass():
    # TODO, make exhaustive.
    print("load")
    cfg = load_config("/home/user/Projects/MiRROR/params", config_name="setup_multiresidue.yaml")
    cfg.session.name = str(uuid.uuid4()).split('-')[0]
    print("setup")
    session = setup(cfg)
    
    suffix_array = session.forward_suffix_array
    residue_space = session.pair_targets[0].residue_space
    fragment_space = session.pair_targets[0].left_fragment_space

    amino_indexer = ord_to_idx(residue_space.amino_symbols)
    amino_masses = residue_space.amino_masses
    peptide = "LKTYF"
    peptide_idx = amino_indexer[np.frombuffer(peptide.encode(),dtype=np.uint8)]
    peptide_masses = amino_masses[peptide_idx]
    peptide_mass = peptide_masses.sum()
    loss_mass = fragment_space.loss_masses[1] # H2O1
    mod_mass = residue_space.modification_masses[2] # MetOx
    q = peptide_mass + mod_mass - loss_mass

    query_mass = peptide_mass + mod_mass - loss_mass
    print("querying...")
    result, _ = query_by_mass(
        query_mass = query_mass,
        tolerance = 0.001,
        suffix_array = suffix_array,
        residue_space = residue_space,
        fragment_space = fragment_space,
        max_peptide_length = 30,
    )
    # print(result)
    # print(list(result))
    assert any(peptide == result.get_peptide(i) for i in range(len(result)))
    # print("query", query_mass, '\n= sum {', peptide_masses, '-', loss_mass, '+', mod_mass)
