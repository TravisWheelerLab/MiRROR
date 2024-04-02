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

'''
the iterative algorithm will basically do the same thing as the noniterative version,
but rather than waiting for the clustering stage to partition the signal we immediately
take the highest prominence peak out of the dataset and then run again on either side.
'''

def find_pivot_point_peaks(
    pivots: list[PivotingIntervalPair],
    resolution: int,
    peak_finding_method: str = "scipy"): 
    # construct a histogram from the pivots list
    pivot_point_hist,_ = pivot_point_histogram(pivots,resolution)
    # compute the peaks of the histogram
    if peak_finding_method == "scipy":
        find_peaks = scipy_find_peaks
    else:
        raise NotImplementedError
    peak_idcs,_ = find_peaks(pivot_point_hist)
    # compute the prominence of each peak and bundle that data into the dict.
    peak_proms, prom_lefts, prom_rights = scipy_peak_prominences(pivot_point_hist,peak_idcs)
    peak_intervals = list(zip(prom_lefts,prom_rights))
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
    peaks = find_pivot_point_peaks(pivots,resolution,peak_finding_method)
    # group pivots by their proximity and relatedness to peak structures
    return cluster_pivots_by_peaks(pivots,peaks,pivot_clustering_method)