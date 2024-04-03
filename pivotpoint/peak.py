from scipy.signal import find_peaks as scipy_find_peaks, peak_prominences as scipy_peak_prominences
from numpy import histogram
from .pivot import PivotingIntervalPair

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
    # compute the prominence of each peak and bundle that data into the dict.
    peak_proms, prom_lefts, prom_rights = scipy_peak_prominences(pivot_point_hist,peak_idcs)
    peak_intervals = [(hist_bin_edges[l],hist_bin_edges[r]) for (l,r) in zip(prom_lefts,prom_rights)]
    return list(zip(peak_idcs,peak_proms,peak_intervals))

def naiive_cluster_pivots(
    pivots: list[PivotingIntervalPair],
    peaks: list[tuple[int,float,tuple[int,int]]]):
    return [peak_interval for (_,__,peak_interval) in peaks]

def cluster_pivots_by_peaks(
    pivots: list[PivotingIntervalPair],
    peaks: list[tuple[int,float,tuple[int,int]]],
    pivot_clustering_method: str = "naiive"):
    if pivot_clustering_method == "naiive":
        cluster_pivots = naiive_cluster_pivots
    else:
        raise NotImplementedError
    return cluster_pivots(pivots,peaks)

def find_pivot_clusters(
    pivots: list[PivotingIntervalPair],
    resolution: int,
    peak_finding_method: str = "scipy",
    pivot_clustering_method: str = "naiive"):
    # find peaks of histogram of pivot centers
    pivot_point_hist_tuple = pivot_point_histogram(pivots,resolution)
    peaks = find_pivot_point_peaks(pivot_point_hist_tuple,peak_finding_method)
    # group pivots by their proximity and relatedness to peak structures
    return cluster_pivots_by_peaks(pivots,peaks,pivot_clustering_method)