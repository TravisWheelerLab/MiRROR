import os, dataclasses, shutil, glob, enum, pathlib
from typing import Self
from time import time
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

    enumerate_candidates(align(annotate(SimulatedPeaks.from_peptide("PEPTIDE"), targets, anno_params, verbose=False), targets, algn_params, verbose=False), targets, enmr_params, verbose=False)
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
    print(f"| annotated in {anno_runtime}s")
    # runs and records spectrum annotation. identifies initial and terminal fragments, and pairs of fragments whose difference encodes the mass of a residue.
    
    algn_results = [
       align(anno_res, targets, algn_params, verbose=app_cfg.verbosity > 1)
       for anno_res in anno_results
    ]
    algn_runtime = _sum_profiles(algn_results, precision=app_cfg.time_precision)
    for res in algn_results:
        output_path = output_dir / f"{app_cfg.session_name}.algn"
        res.save(output_path)
    print(f"| aligned in {algn_runtime}s")
    # runs and records graph alignment. constructs spectrum graphs and their (sparse) product via cost propagation.
    
    enmr_results = [
       enumerate_candidates(algn_res, targets, enmr_params, verbose=app_cfg.verbosity > 1)
       for algn_res in algn_results
    ]
    enmr_runtime = _sum_profiles(enmr_results, precision=app_cfg.time_precision)
    for res in enmr_results:
        output_path = output_dir / f"{app_cfg.session_name}.algn"
        res.save(output_path)
    print(f"| enumerated in {enmr_runtime}s")
    # runs and records candidate enumeration. candidate substrings are generated by depth-first search on the product graph, then stitched together using annotated data and an optional suffix array.

    eval_total_runtime = round(anno_runtime + algn_runtime + enmr_runtime, app_cfg.time_precision)
    total_runtime = round(time() - t, app_cfg.time_precision)
    print(f"\nevaluate: {eval_total_runtime}s, runtime: {total_runtime}s")

def test(cfg: DictConfig):
    print("MRROR test!")
    x = SimulatedPeaks.from_peptide("PEPTIDE",initial_b=True)
    print("peaks\n\t",x.zip())
    print("\ntrunc 1\n\t",x.truncate(1).zip())
    print("\ntrunc 2\n\t",x.truncate(2).zip())
    print("\ntrunc 1 b\n\t",x.truncate(1,series='b').zip())
    print("\ntrunc 2 y terminal\n\t",x.truncate(2,series='y',terminal=True).zip())
    print("\noccl 3\n\t",x.occlude(3).zip())
    print("\noccl 4 b\n\t",x.occlude(4,series='b').zip())
    print("\noccl [3,4] y\n\t",x.occlude([3,4],series='y').zip())
    print(f"\ncorrupt\n\t",x.corrupt().zip())
    print(f"\ncorrupt snr=0.5\n\t",x.corrupt(snr=0.5).zip())
    print(f"\ncorrupt num=3\n\t",x.corrupt(num=3).zip())
    
    anno_params = AnnotationParams.from_config(cfg['annotation'])
    peaks_data = PeaksDataset.from_peptide("TEST", initial_b=True)
    targets = TargetMassStateSpace.from_state_spaces(
        [anno_params.residue_space, anno_params.dimer_space, anno_params.trimer_space],
        anno_params.fragment_space,
        anno_params.boundary_fragment_space,
    )
    anno_results = [
        annotate(peaks, targets, anno_params)
        for peaks in peaks_data
    ]
    print("\nanno_results")
    for x in anno_results:
        input(x)
        x.save("test.anno")
        y = AnnotationResult.load("test.anno")
        input(y)
        y.save("test2.anno")
        z = AnnotationResult.load("test2.anno")
        input(z)
        input(str(dataclasses.asdict(z)).replace('\n','').replace('\t','').replace('  ',''))

if __name__ == "__main__":
    main()
