from typing import Iterator
from .minimal_nodes import propagate
from .minimal_paths import backtrace
from .align_types import LocalCostModel
from .ensemble_types import EnsembleAlignment
from .concatenation_types import DualIntervalDAG, ConcatenationAlignment
from networkx import is_bipartite, connected_components

# this file is basically a clone of ensemble.py
# TODO: refactor with a shared interface
# e.g.: solve_alignment_order iterates the components of an AlignmentOrderGraph as AlignmentDAG objects

def _construct_ensemble_intervals(
    alignments: list[EnsembleAlignment],
    first_node_weights: list[float],
    second_node_weights: list[float],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    first_intervals = []
    second_intervals = []
    for aln in alignments:
        fs = aln.first_source()
        ft = aln.first_target()
        ss = aln.second_source()
        st = aln.second_target()
        print(f"aln: {aln}\nfirst source/target: {fs, ft}\nsecond source/target: {ss, st}")
        first_intervals.append((first_node_weights[aln.first_source()], first_node_weights[aln.first_target()]))
        second_intervals.append((second_node_weights[aln.second_target()], second_node_weights[aln.second_source()]))
    return (first_intervals, second_intervals)

def _solve_ensemble_concatenations(
    ensemble_dag: DualIntervalDAG,
    cost_model: LocalCostModel,
    threshold: float,
):
    ensemble_sequences = []
    sources = ensemble_dag.sources()
    sinks = ensemble_dag.sinks()
    # the weights of the ensemble dag are their own cost
    identity_cost = lambda _, x: x
    for src in sources:
        nc = propagate(
            topology = ensemble_dag,
            cost = identity_cost,
            threshold = threshold,
            source = src,
        )
        for snk in sinks:
            minimal_paths = backtrace(
                topology = ensemble_dag,
                cost = identity_cost,
                node_cost = nc,
                threshold = threshold,
                source = src,
                sink = snk,
            )
            ensemble_sequences.extend(minimal_paths)
    return ensemble_sequences

def concatenate_ensembles(
    alignments: list[EnsembleAlignment],
    first_node_weights: list[float],
    second_node_weights: list[float],
    cost_model: LocalCostModel,
    threshold: float,
):
    # construct the ensemble sequentiality
    first_ensemble_intervals, second_ensemble_intervals = _construct_ensemble_intervals(alignments, first_node_weights, second_node_weights)
    ensemble_dag = DualIntervalDAG(
        first_intervals = first_ensemble_intervals,
        second_intervals = second_ensemble_intervals)
    # find optimal sequences of ensembles
    ensemble_concatenations = _solve_ensemble_concatenations(
        ensemble_dag = ensemble_dag,
        cost_model = cost_model,
        threshold = threshold)
    # concatenate the ensemble sequences
    return list(map(
        lambda x: ConcatenationAlignment(
            score = x[0],
            ensemble_sequence = [alignments[i] for i in x[1]]),
        ensemble_concatenations))
