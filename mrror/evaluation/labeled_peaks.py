import dataclasses, collections
import itertools as it
from typing import Self

from ..util import listsum, fuzzy_unique
from ..spectra.types import Peaks, AbstractLabeledPeaks, LabelType
from ..fragments.types import PairResult, BoundaryResult, PivotResult, TargetMasses

import numpy as np

_LabelData = tuple[
	np.ndarray,			# indices
	np.ndarray,			# charges
	list[np.ndarray],	# per-state costs
	list[np.ndarray],	# per-state target indices
	list[np.ndarray],	# per-state losses
	list[np.ndarray],	# per-state mods
]

@dataclasses.dataclass(slots=True)
class FragmentLabels:
	peaks: Peaks
	index: np.ndarray							# [int; n]
	charge: np.ndarray							# [int; n]
	loss: list[np.ndarray]						# [[int; _]; n]
	modifications: list[np.ndarray]				# [[int; _]; n]
	target_index: list[np.ndarray]				# [[int; _]; n]
	costs: list[np.ndarray]						# [[float; _]; n]
	mass: np.ndarray							# [float; n]
	unique_mass_index: np.ndarray				# [int; n]
	pair_segment: int							
	lower_boundary_segment: int					
	pivot_interchunk_segments: list[int]		# [int; p]
	pivot_intrachunk_segments: list[list[int]]	# [[int; 3]; p]
	
	def reindex_pairs(
	    self,
	) -> np.ndarray: # [[int; 2]; n]
		return self.unique_mass_index[:self.pair_segment].reshape((-1,2))

	def reindex_lower_boundaries(
	    self,
	) -> np.ndarray: # [int; l]
		return self.unique_mass_index[self.pair_segment:self.lower_boundary_segment]

	def reindex_upper_boundaries(
		self,
	) -> list[np.ndarray]: # [[int; _]; p]
		return [
			self.unique_mass_index[i:j][segments[0]:segments[1]]
			for ((i,j), segments) in zip(
			    it.pairwise(self.pivot_interchunk_segments),
			    self.pivot_intrachunk_segments
			)
		]

	def reindex_pivots(
	    self,
	) -> list[np.ndarray]: # [[[int; 4]; _]; p]
		return [
			self.unique_mass_index[i:j][segments[1]:segments[2]].reshape((-1,4))
			for ((i,j), segments) in zip(
			    it.pairwise(self.pivot_interchunk_segments),
			    self.pivot_intrachunk_segments
			)
		]

	def reindex_symmetries(
	    self,
	) -> list[np.ndarray]: # [[[int; 2]; _]; p]
		return [
			self.unique_mass_index[i:j][segments[2]:segments[3]].reshape((-1,2))
			for ((i,j), segments) in zip(
			    it.pairwise(self.pivot_interchunk_segments),
			    self.pivot_intrachunk_segments
			)
		]

	@classmethod
	def _extract_pairs_labels(
		cls,
		pairs: list[PairResult],
	) -> _LabelData:
		pair_segments = [list(it.pairwise(x.segments)) for x in pairs]
		return (
			np.concat([x.indices.flatten() for x in pairs]),
			# indices

			np.concat([x.charges.flatten() for x in pairs]),
			# charges

			[
				x.costs[i:j]
				for (x,segments) in zip(pairs,pair_segments)
				for (i,j) in segments
				for k in range(2)
			],
			# per-state costs

			[
				x.features[i:j,4] 
			    for (x,segments) in zip(pairs,pair_segments)
				for (i,j) in segments
				for k in range(2)
			],
			# per-state target indices

			[
				x.features[i:j,k]
			    for (x,segments) in zip(pairs,pair_segments)
				for (i,j) in segments
				for k in range(1,3)
			],
			# per-state losses

			[
				np.empty((0,),dtype=int) if k==0 else x.features[i:j,3] 
				for (x,segments) in zip(pairs,pair_segments)
				for (i,j) in segments
				for k in range(2)
			],
			# per-state mods
		)

	@classmethod
	def _extract_boundaries_labels(
		cls,
		boundaries: list[BoundaryResult],
	) -> _LabelData:
		boundary_segments = [list(it.pairwise(x.segments)) for x in boundaries]
		return (
			np.concat([x.index for x in boundaries]),
			# indices

			np.concat([x.charge for x in boundaries]),
			# charges

			[
				x.costs[i:j]
				for (x,segments) in zip(boundaries,boundary_segments)
				for (i,j) in segments
			],
			# per-state costs

			[
				x.features[i:j,4]
				for (x,segments) in zip(boundaries,boundary_segments)
				for (i,j) in segments
			],
			# per-state target indices

			[
				x.features[i:j,2]
				for (x,segments) in zip(boundaries,boundary_segments)
				for (i,j) in segments
			],
			# per-state losses
			
			[
				x.features[i:j,3]
				for (x,segments) in zip(boundaries,boundary_segments)
				for (i,j) in segments
			],
			# per-state mods
		)

	@staticmethod
	def _null_charge(n: int) -> np.ndarray:
		return np.ones(n, dtype=int)

	@staticmethod
	def _null_cost(n: int) -> np.ndarray:
		return [[np.inf,] for _ in range(n)]#np.full(n, np.inf, dtype=float)
	
	@staticmethod
	def _null_target_indices(n: int) -> list[list[int]]:
		return [[-1,] for _ in range(n)]
	
	@staticmethod
	def _null_losses(n: int) -> list[list[int]]:
		return [[0,] for _ in range(n)]
	
	@staticmethod
	def _null_mods(n: int) -> list[list[int]]:
		return [[0,] for _ in range(n)]

	@classmethod
	def _extract_pivots_labels(
		cls,
		pivots: PivotResult,
	) -> tuple[
		list[_LabelData], 	# pivots
		list[_LabelData],	# per-pivot symmetries
	]:
		pivot_labels = []
		sym_labels = []
		
		for i in range(len(pivots)):
			pivot_index = pivots.pivot_indices[pivots.clusters[i]].flatten()
			n = len(pivot_index)
			pivot_labels.append((
				pivot_index,
				# indices
				cls._null_charge(n),
				# charges
				cls._null_cost(n),
				# per-state costs
				cls._null_target_indices(n),
				# per-state target indices
				cls._null_losses(n),
				# per-state losses
				cls._null_mods(n),
				# per-state mods
			))
			# label pivot indices with null data and inf cost.

			sym_index = pivots.symmetries[i].flatten()
			m = len(sym_index)
			sym_labels.append((
			    sym_index,
				# indices
				cls._null_charge(m),
				# charges
				cls._null_cost(m),
				# per-state costs
				cls._null_target_indices(m),
				# per-state target indices
				cls._null_losses(m),
				# per-state losses
				cls._null_mods(m),
				# per-state mods
			))
			# likewise, symmetric indices get null labels.
		
		return (
			pivot_labels,
			sym_labels
		)

	@classmethod
	def _concat_labels(
		cls,
		labels: list[_LabelData],
	) -> tuple[
		_LabelData,	# result of concatenation
		list[int],	# offsets to recover each component label data.
	]:
		offsets = []
		indices = []
		charges = []
		costs = []
		tgt_indices = []
		losses = []
		mods = []
		for (_indices, _charges, _costs, _tgt_indices, _losses, _mods) in labels:
			offsets.append(len(_indices))
			indices.append(_indices)
			charges.append(_charges)
			costs.append(_costs)
			tgt_indices.append(_tgt_indices)
			losses.append(_losses)
			mods.append(_mods)
		return (
			(
				np.concat(indices),		# indices
				np.concat(charges),		# charges
				listsum(costs),			# per-state costs
				listsum(tgt_indices),	# per-state target indices
				listsum(losses),		# per-state losses
				listsum(mods),			# per-state mods
			),
			np.cumsum([0] + offsets), 	# segments
		)

	@classmethod
	def from_results(
		cls,
		peaks: Peaks,
		pairs: list[PairResult],
		lower_boundaries: list[BoundaryResult],
		upper_boundaries: list[list[BoundaryResult]],
		pivots: PivotResult,
		tolerance: float = 0.01,
	) -> Self:
		pair_labels = cls._extract_pairs_labels(pairs)
		lb_labels = cls._extract_boundaries_labels(lower_boundaries)
		ub_labels = [cls._extract_boundaries_labels(x) for x in upper_boundaries]
		pivot_labels, symmetry_labels = cls._extract_pivots_labels(pivots)
		# extract label data

		pivot_chunks, pivot_intrachunk_segments = zip(*[
			cls._concat_labels([ub,pvt,sym])
			for (ub,pvt,sym) in zip(ub_labels,pivot_labels,symmetry_labels)
		])
		pivot_chunks = list(pivot_chunks)
		pivot_intrachunk_segments = list(pivot_intrachunk_segments)
		# concat pivot-dependent triplets
		
		labels = [pair_labels, lb_labels] + pivot_chunks
		concat_labels, segments = cls._concat_labels(labels)
		idx, charge, costs, tgt_idx, loss, mods = concat_labels
		pair_segment, lb_segment, *pivot_chunk_segments  = segments
		# concatenate label data.

		fragment_mass = peaks.mz[idx] * charge
		unique_fragment_mass, unique_mass_index = fuzzy_unique(fragment_mass, tolerance)
		# use fragment masses to cluster indices.

		return cls(
			peaks = peaks,
			index = idx,
			charge = charge,
			loss = loss,
			modifications = mods,
			target_index = tgt_idx,
			costs = costs,
			mass = unique_fragment_mass,
			unique_mass_index = unique_mass_index,
			pair_segment = pair_segment,
			lower_boundary_segment = lb_segment,
			pivot_interchunk_segments = pivot_chunk_segments,
			pivot_intrachunk_segments = pivot_intrachunk_segments,
		)

@dataclasses.dataclass(slots=True)
class AnnotationLabeledPeaks(AbstractLabeledPeaks):
	@classmethod
	def from_fragment_labels(
		cls,
		labels: FragmentLabels,
		targets: list[TargetMasses],
	) -> Self:
		mz = labels.peaks.mz
		n = len(mz)
		losses = [[]  for _ in range(n)]
		costs = [[] for _ in range(n)]
		target_indices = [[] for _ in range(n)]
		charges = [[] for _ in range(n)]
		original_indices = labels.index
		m = len(original_indices)
		for i in range(m):
			j = original_indices[i]
			costs[j].append(labels.costs[i])
			losses[j].append(labels.loss[i])
			target_indices[j].append(labels.target_index[i])
			charges[j].append(np.array([labels.charge[i],] * len(labels.costs[i])))
		# cluster fragment labels by original (peak) index.

		losses = [np.concat(x) for x in losses]
		costs = [np.concat(x) for x in costs]
		target_indices = [np.concat(x) for x in target_indices]
		charges = [np.concat(x) for x in charges]
		for i in range(n):
			order = np.argsort(costs[i])
			charges[i] = charges[i][order]
			losses[i] = np.array([
				targets[t].right_fragment_space.loss_symbols[l] if t != -1 else '' # TODO
				for (t,l) in zip(target_indices[i],losses[i])
			])[order]
		# sort labels and convert to symbolic representation.
		
		return cls(
			label_type = LabelType.FRAGMENTS,
			peptide = '',
			pivot = 0.,
			mz = mz,
			intensity = labels.peaks.intensity,
			series = np.full(n, '', dtype=str),
			position = np.zeros(n, dtype=int),
			charge = charges,
			loss = losses,
			mods = [],
		)

@dataclasses.dataclass(slots=True)
class CandidateLabeledPeaks(AbstractLabeledPeaks):
	@classmethod
	def from_candidate_cluster(
		peaks: Peaks,
		cluster: int,
		annotation,#: tuple[AnnotationResult,AnnotationParams],
		alignment,#: tuple[AlignmentResult,AlignmentParams],
		enumeration,#: tuple[EnumerationResult,EnumerationParams],
		candidate: int = None,
	) -> Self:
		anno_res, anno_cfg = annotation
		algn_res, algn_cfg = alignment
		enmr_res, enmr_cfg = enumeration
		candidate_cluster = enmr_res.candidates[cluster]
		peptide = candidate_cluster.get_sequence(candidate)
		mods = [None for _ in peptide]
		# TODO mods
		mz, intensity, charge = candidate_cluster.get_peaks(
			candidate,
			peaks,
			anno_res.pairs,
			anno_res.lower_boundaries,
			anno_res.upper_boundaries[cluster],
			algn_res.prod_topology[cluster],
			algn_res.lower_topology[cluster],
			algn_res.upper_topology[cluster],
		)
		loss = [np.empty((0,),dtype=str) for _ in mz]
		# TODO loss
		mz_ord = np.argsort(mz)
		mz = mz[mz_ord]
		intensity = intensity[mz_ord]
		series = candidate_cluster.get_series(candidate)
		position = candidate_cluster.get_position(candidate)
		return cls(
			label_type = LabelType.CANDIDATE,
			peptide = peptide,
			pivot = anno_res.pivots.cluster_points[cluster],
			mz = mz,
			intensity = intensity,
			series = series,
			position = position,
			charge = charge,
			loss = loss,
			mods = mods,
		)
