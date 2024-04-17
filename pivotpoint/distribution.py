from scipy.signal import find_peaks as scipy_find_peaks, peak_prominences as scipy_peak_prominences
from numpy import histogram
from .pivot import PivotingIntervalPair, pivot_sort_key
from .util import unique

class PivotPointsDistribution:
    def __init__(self,
        pivots: list[PivotingIntervalPair],
        resolution: int):
        # construct the distribution itself.
        self.pivots = pivots
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
        self.clusters.sort(key = lambda x: -len(x))
        
    def n_pivots(self):
        return self._n_pivot_points

    def n_clusters(self):
        return self._n_peaks

    def get_pivot_cluster(self,i):
        return PivotPointsCluster([self.pivots[pivot_id] for pivot_id in self.clusters[i]])


class PivotPointsCluster:
    def __init__(self,
        pivots: list[PivotingIntervalPair]):
        self.pivots = sorted(pivots,key = pivot_sort_key)

    def get_data(self):
        sorted(unique(collect_pivot_data([x.data for x in self.pivots])))

    def get_pivots(self):
        return self.pivots