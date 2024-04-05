from scipy.signal import find_peaks as scipy_find_peaks, peak_prominences as scipy_peak_prominences
from numpy import histogram
from .pivot import PivotingIntervalPair
from .util import make_index_table

class PivotPointsDistribution:
    def __init__(self,
        pivots: list[PivotingIntervalPair],
        resolution: int):
        # construct the distribution itself.
        self._n_pivot_points = len(pivots)
        self.pivot_points = [pivot.center() for pivot in pivots]
        self.resolution = resolution
        # fit the histogram 
        hist,bin_edges = histogram(self.pivot_points,bins = self.resolution)
        self.pivot_points_histogram = hist
        self.pivot_points_bin_edges = bin_edges
        # find the index to every peak in the histogram.
        peak_idc = scipy_find_peaks(self.pivot_points_histogram)[0]
        self._n_peaks = len(peak_idc)
        self.peak_indices = peak_idc
        # calculate prominence, bases of each peak.
        peak_proms, peak_lefts, peak_rights = scipy_peak_prominences(self.pivot_points_histogram,self.peak_indices)
        self.peak_prominences = peak_proms
        self.peak_bases = list(zip(peak_lefts, peak_rights))
        # find clusters
        self.clusters = [[] for _ in range(self._n_peaks)]
        for pivot_id in range(self._n_pivot_points):
            val = self.pivot_points[pivot_id]
            for peak_id in range(self._n_peaks):
                left_index, right_index = self.peak_bases[peak_id]
                left_val = self.pivot_points_bin_edges[left_index]
                right_val = self.pivot_points_bin_edges[right_index]
                if left_val <= val <= right_val:
                    self.clusters[peak_id].append(pivot_id)
        
    def n_pivots(self):
        return self._n_pivot_points

    def n_clusters(self):
        return self._n_peaks

def pivot_point_histogram(
    pivots: list[PivotingIntervalPair],
    bins_arg):
    # list the pivot points of each pivoting interval pair.
    pivot_centers = [pivot.center() for pivot in pivots]
    # invoke numpy's histogram function
    return histogram(pivot_centers,bins = bins_arg)

def find_pivot_point_peaks(
    pivot_point_hist_tuple: tuple,
    peak_finding_method: str = "scipy"):
    # unpack, we'll need the bin edges later
    pivot_point_hist,hist_bin_edges = pivot_point_hist_tuple
    # compute the peaks of the histogram
    if peak_finding_method == "scipy":
        find_peaks = scipy_find_peaks
    else:
        raise NotImplementedError
    peak_idcs,_ = find_peaks(pivot_point_hist)
    peak_proms, prom_lefts, prom_rights = scipy_peak_prominences(pivot_point_hist,peak_idcs)
    peak_intervals = [(hist_bin_edges[l],hist_bin_edges[r]) for (l,r) in zip(prom_lefts,prom_rights)]
    return list(zip(peak_idcs,peak_proms,peak_intervals))

def naiive_pivot_clusters(
    pivots: list[PivotingIntervalPair],
    peaks: list[tuple[int,float,tuple[int,int]]]):
    return [peak_interval for (_,__,peak_interval) in peaks]

def cluster_pivots_by_peaks(
    pivots: list[PivotingIntervalPair],
    peaks: list[tuple[int,float,tuple[int,int]]],
    pivot_clustering_method: str = "naiive"):
    if pivot_clustering_method == "naiive":
        pivot_clusters = naiive_pivot_clusters
    else:
        raise NotImplementedError
    return pivot_clusters(pivots,peaks)

def interpolate_pivot_cluster(
    bounds: tuple[float,float],
    pivots: list[PivotingIntervalPair]):
    left_bound, right_bound = bounds
    cluster = list[PivotingIntervalPair]()
    n = len(pivots)
    for i in range(n):
        p = pivots[i]
        c = p.center()
        if left_bound < c < right_bound:
            cluster.append(p)
    return cluster

def find_pivot_clusters(
    pivots: list[PivotingIntervalPair],
    resolution: int,
    peak_finding_method: str = "scipy",
    pivot_clustering_method: str = "naiive"):
    # find peaks of histogram of pivot centers
    pivot_point_hist_tuple = pivot_point_histogram(pivots,resolution)
    peaks = find_pivot_point_peaks(pivot_point_hist_tuple,peak_finding_method)
    # group pivots by their proximity and relatedness to peak structures
    cluster_bounds = cluster_pivots_by_peaks(pivots,peaks,pivot_clustering_method)
    return [interpolate_pivot_cluster(cb,pivots) for cb in cluster_bounds]