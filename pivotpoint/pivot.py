from .interval_pair import find_disjoint_quasiisometric_interval_pairs

class PivotingIntervalPair:
    def __init__(self, p):
        self._data = list(p)

    def data(self):
        return self._data

    def center(self):
        return (self._data[1] + self._data[2]) / 2
    
    def gap(self):
        return (self._data[1] - self._data[0], self._data[3] - self._data[2])
    
    def error(self):
        gap_left, gap_right = self.gap()
        return abs(gap_right - gap_left)

    def outer(self):
        return self.data()[[0,3]]

    def outer_diameter(self):
        return self.outer()[1] - self.outer()[0]

    def inner(self):
        return self.data()[[1,2]]

    def inner_diameter(self):
        return self.inner()[1] - self.inner()[0]

    def succeeds(self, other: PivotingIntervalPair):
        return self.outer() == other.inner()

    def contains(self, other: PivotingIntervalPair):
        return self.inner()[1] < other.outer()[1] < other.outer()[2] < self.inner()[2]
    
def pivot_from_data(indices: tuple[int,int,int,int], dataset: list[float]):
    data = (dataset[i] for i in indices)
    return PivotingIntervalPair(data)

def pivot_from_params(middle: float, gap: float, radius: float):
    hgap = gap / 2
    p1 = middle - radius - hgap
    p2 = middle - radius + hgap
    p3 = middle + radius - hgap
    p4 = middle + radius + hgap
    data = (p1,p2,p3,p4)
    return PivotingIntervalPair(data)

def construct_pivoting_interval_pairs(
    data: list[float],
    pivot_indices: list[tuple[int,int,int,int]]):
    # curry `pivot_from_data` with `data` to get the constructor
    def _pivot(indices):
        return pivot_from_data(indices,data)
    # eagerly construct pivots parametized by `pivot_indices`
    return list(map(_pivot,pivot_indices))

def find_pivoting_interval_pairs(
    data: list[float], 
    gapset: list[float], 
    precision: float,
    search_mode: str = "simple"):
    # parametize the pivots over `data`
    pivot_indices = find_disjoint_quasiisometric_interval_pairs(
        data,gapset,precision,search_mode = search_mode)
    return construct_pivoting_interval_pairs(data,pivot_indices)
