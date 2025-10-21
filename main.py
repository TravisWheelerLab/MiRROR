import os, dataclasses, shutil, glob, enum, pathlib
from typing import Self
from time import time
import itertools as it
# standard

import hydra
from omegaconf import DictConfig, OmegaConf
# hydra

from mrror.io import read_mzlib, read_mgf
from mrror.util import in_alphabet
from mrror.fragments.types import TargetMassStateSpace
from mrror.spectra.types import SpectraParams, PeaksDataset, Peaks, SimulatedPeaks

from mrror.annotation import AnnotationParams, AnnotationResult, annotate
from mrror.alignment import AlignmentParams, AlignmentResult, align
from mrror.enumeration import EnumerationParams, EnumerationResult, enumerate_candidates
# library

import numpy as np

def _sum_profiles(results, precision=4):
    return round(sum(t for res in results for t in res._profile.values()), precision)

@dataclasses.dataclass(slots=True)
class AppParams:
    name: str
    version: str
    descr: str
    verbosity: int
    time_precision: int
    mode: str
    output_dir: str
    output_shape: str
    session_name: str

    @classmethod
    def from_dict(
        cls,
        cfg: DictConfig,
    ) -> Self:
        return cls(
            **cfg,
        )

    def __repr__(self) -> str:
        symbol = '-'
        repeats = shutil.get_terminal_size().columns // len(symbol)
        header = symbol * repeats
        if self.verbosity == -1:
            return f"{self.name} v{self.version}\n{header}\n{self.descr}"
        else:
            return f"{self.name} v{self.version}"

class InputType(enum.Enum):
    MZLIB = "mzSpecLib"
    MGF = "MGF"
    FASTA = "Fasta"
    PEPTIDE = "Peptide"
    UNKNOWN = "???"

@hydra.main(version_base=None, config_path="params", config_name="config")
def main(cfg: DictConfig) -> None:
    mode = cfg['app']['mode']
    if mode == 'evaluate':
        return evaluate(cfg)
    elif mode == 'test':
        return test(cfg)
    else:
        print(f"Unrecognized mode {mode}")

def evaluate(cfg: DictConfig) -> None:
    t = time()
    app_cfg = AppParams.from_dict(cfg['app'])
    print(app_cfg)
    # configures the application.

    spec_cfg = SpectraParams.from_dict(cfg['spectra'])
    print('.', end='')
    # configures spectra simulation.
    
    anno_params = AnnotationParams.from_config(cfg['annotation'])
    alphabet = anno_params.residue_space.amino_symbols
    if app_cfg.verbosity > 1:
        print(anno_params)
    print('.', end='')
    # configures spectrum annotation, describes the state spaces of fragments and residues, as well as search and filtration parameters.
    
    algn_params = AlignmentParams.from_config(cfg['alignment'])
    if app_cfg.verbosity > 1:
        print(algn_params)
    print('.', end='')
    # configures graph alignment. describes a cost model, alignment mode, and cost threshold.
    
    enmr_params = EnumerationParams.from_config(cfg['enumeration'])
    if app_cfg.verbosity > 1:
        print(enmr_params)
    print('.', end='')
    # configures candidate enumeration.
    
    targets = TargetMassStateSpace.from_state_spaces(
        [anno_params.residue_space, anno_params.dimer_space, anno_params.trimer_space],
        anno_params.fragment_space,
        anno_params.boundary_fragment_space,
    )
    print('.', end='')
    # admissible target masse for pair deltas and boundaries.

    _anno = annotate(SimulatedPeaks.from_peptide("PEPTIDE"), targets, anno_params, verbose=False)
    enumerate_candidates(_anno, align(_anno, targets, algn_params, verbose=False), targets, enmr_params, verbose=False)
    precompile_runtime = time() - t
    print('.', end='')
    # precompile numba.jit functions

    output_dir = pathlib.Path(app_cfg.output_dir)
    if not(output_dir.exists()):
        output_dir.mkdir(parents=True, exist_ok=True)
    print('.')
    # construct the output path

    input_str = cfg['input']
    input_type = (
        InputType.MZLIB if input_str.endswith(".mzlib.txt") else 
        InputType.MGF if input_str.endswith(".mgf") else 
        InputType.FASTA if (input_str.endswith(".fasta") or input_str.endswith(".fa")) else
        InputType.PEPTIDE if (len(input_str) > 0 and in_alphabet(input_str.split('::')[0],alphabet)) else 
        InputType.UNKNOWN
    )
    peaks_data = None
    if input_type == InputType.MZLIB:
        peaks_data = PeaksDataset.from_mzml(input_str)
    elif input_type == InputType.MGF:
        peaks_data = PeaksDataset.from_mgf(input_str)
    elif input_type == InputType.FASTA:
        peaks_data = PeaksDataset.from_fasta(input_str)
    elif input_type == InputType.PEPTIDE:
        peaks_data = PeaksDataset.from_peptide(input_str, initial_b=spec_cfg.initial_b)
    else:
        raise ValueError(f"Unrecognized input type. The input string {input_str} cannot be parsed!")
    print(f"\n* {input_str} [{input_type}]")
    if app_cfg.verbosity > 2:
        print(peaks_data)
    # parses the input field and from it constructs a spectrum.
    
    anno_results = [
        annotate(peaks, targets, anno_params, verbose=app_cfg.verbosity > 1)
        for peaks in peaks_data
    ]
    anno_runtime = _sum_profiles(anno_results, precision=app_cfg.time_precision)
    for res in anno_results:
        output_path = output_dir / f"{app_cfg.session_name}.anno"
        res.save(output_path)
    if app_cfg.verbosity > 0:
        print(f"| annotation\t{anno_runtime}s")
    # runs and records spectrum annotation. identifies initial and terminal fragments, and pairs of fragments whose difference encodes the mass of a residue.
    
    algn_results = [
       align(anno_res, targets, algn_params, verbose=app_cfg.verbosity > 1)
       for anno_res in anno_results
    ]
    algn_runtime = _sum_profiles(algn_results, precision=app_cfg.time_precision)
    for res in algn_results:
        output_path = output_dir / f"{app_cfg.session_name}.algn"
        res.save(output_path)
    if app_cfg.verbosity > 0:
        print(f"| alignment\t{algn_runtime}s")
    # runs and records graph alignment. constructs spectrum graphs and their (sparse) product via cost propagation.
    
    enmr_results = [
       enumerate_candidates(anno_res, algn_res, targets, enmr_params, verbose=app_cfg.verbosity > 1)
       for (anno_res, algn_res) in zip(anno_results, algn_results)
    ]
    enmr_runtime = _sum_profiles(enmr_results, precision=app_cfg.time_precision)
    for res in enmr_results:
        output_path = output_dir / f"{app_cfg.session_name}.algn"
        res.save(output_path)
    if app_cfg.verbosity > 0:
        print(f"| enumeration\t{enmr_runtime}s")
    # runs and records candidate enumeration. candidate substrings are generated by depth-first search on the product graph, then stitched together using annotated data and an optional suffix array.

    eval_total_runtime = round(anno_runtime + algn_runtime + enmr_runtime, app_cfg.time_precision)
    total_runtime = round(time() - t, app_cfg.time_precision)
    if app_cfg.verbosity > 0:
        print(f"\nevaluate: {eval_total_runtime}s, runtime: {total_runtime}s")

def test(cfg: DictConfig):
    print("MRROR test!")
    app_cfg = AppParams.from_dict(cfg['app'])

    spec_cfg = SpectraParams.from_dict(cfg['spectra'])
    
    anno_params = AnnotationParams.from_config(cfg['annotation'])
    alphabet = anno_params.residue_space.amino_symbols
    
    algn_params = AlignmentParams.from_config(cfg['alignment'])
    
    enmr_params = EnumerationParams.from_config(cfg['enumeration'])
    
    targets = TargetMassStateSpace.from_state_spaces(
        [anno_params.residue_space, anno_params.dimer_space, anno_params.trimer_space],
        anno_params.fragment_space,
        anno_params.boundary_fragment_space,
    )

    _anno = annotate(SimulatedPeaks.from_peptide("PEPTIDE"), targets, anno_params, verbose=False)
    enumerate_candidates(_anno, align(_anno, targets, algn_params, verbose=False), targets, enmr_params, verbose=False)

    peaks = SimulatedPeaks.from_peptide(cfg['input'],initial_b=spec_cfg.initial_b)
    print("peaks:")
    print("peptide\n- ", peaks.peptide)
    print("peaks\n- ", peaks.mz)
    print("pivot\n- ", peaks.pivot)
    print("left boundaries\n- ", peaks.left_boundaries())
    print("right boundaries\n- ", peaks.right_boundaries())
    print("pairs\n- ", peaks.pairs())
    print("paths\n- ", peaks.paths())
    true_alignments = peaks.alignments()
    print("alignments\n- ", true_alignments)

    anno_res = annotate(peaks, targets, anno_params, verbose=app_cfg.verbosity > 1)
    observed_pivots = anno_res.pivots.cluster_points
    observed_pairs = anno_res.pairs.indices.T.tolist()
    observed_lb = anno_res.left_boundaries.index.tolist()
    observed_rb = [rb.index.tolist() for rb in anno_res.right_boundaries]
    print("annotation")
    print(observed_pivots)
    print(observed_pairs)
    print(observed_lb)
    print(observed_rb)

    print("anno symmetries: ", [sym.tolist() for sym in anno_res.pivots.symmetries], '\n', [np.round(peaks.mz[sym], 4).tolist() for sym in anno_res.pivots.symmetries])
    
    algn_res = align(anno_res, targets, algn_params, verbose=app_cfg.verbosity > 1)
    print("algn symmetries: ", [sym[:-1].tolist() for sym in algn_res.symmetries], '\n', [np.round(algn_res.fragment_masses[sym[:-1,:]], 4).tolist() for sym in algn_res.symmetries])
    enmr_res = enumerate_candidates(anno_res, algn_res, targets, enmr_params, verbose=app_cfg.verbosity > 1)
    enmr_paths = [[[g.unravel(v) for v in x[::-1][1:]] for (_,x) in pathspace] for (g,pathspace) in zip(algn_res.sparse_prod,enmr_res.aligned_paths)]
    enmr_paths = [[[(int(x),int(y)) for (x,y) in path] for path in pathspace] for pathspace in enmr_paths]
    print("enmr\n- ",len(enmr_paths[0]))
    print("matches")
    import editdistance as ed
    fragment_masses = algn_res.fragment_masses
    mz = peaks.mz
    for pathspace in enmr_paths:
        for alignment in pathspace:
            alignment = [(mz.searchsorted(fragment_masses[i]).tolist(), mz.searchsorted(fragment_masses[j]).tolist()) for (i, j) in alignment]
            rev_alignment = alignment[::-1]
            min_aln = (9999,None)
            min_rev = (9999,None)
            for other_algn in true_alignments:
                d_aln = ed.distance(alignment,other_algn)
                if d_aln < min_aln[0]:
                    min_aln = (d_aln, other_algn)
                d_rev = ed.distance(rev_alignment, other_algn)
                if d_rev < min_rev[0]:
                    min_rev = (d_rev, other_algn)
            if min_aln[0] < 2:
                print('  '*min_aln[0],*min_aln, alignment)
            if min_rev[0] < 2:
                print('  '*min_rev[0],*min_rev, rev_alignment)
    
if __name__ == "__main__":
    main()
