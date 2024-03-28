def gap_search_simple(arr: list[float], key: float, precision: float):
    n = len(arr)
    candidates = list[tuple[float,float]]()
    for i in range(n):
        x = arr[i]
        for j in range(i + 1,n):
            y = arr[j]
            gap = y - x
            ε = abs(key - gap)
            if ε <= precision:
                # `gap` is close enough to `key`; this is a match.
                candidates.append((i,j))
            elif gap > ε:
                # `gap` is out of bounds and larger than `key`; 
                # we assume that `arr` is in ascending order, so 
                # there's no point in looking past this point.
                break
            else:
                # otherwise, `gap` is too small; keep looking.
                continue
    return candidates

"floor divide"
def fld(num,dom):
    return num // dom

def gap_search_binary(arr: list[float], key: float, precision: float):
    n = len(arr)
    candidates = list[tuple[int,int]]()
    for i in range(n):
        lo = i
        hi = n - 1
        x = arr[i]
        while l <= r:
            j = fld(l + r,2)
            y = arr[j]
            gap = y - x
            ε = abs(key - gap)
            if ε < precision:
                candidates.append((i,j))
                break
            elif gap > key:
                r = j - 1
            else: # gap < key
                l = j + 1
    return candidates

def unique_ascending_pairs(arr: list):
    return [(arr[i],arr[j]) for i in range(len(arr)) for j in range(i + 1,len(arr))]

def find_disjoint_quasiisometric_interval_pairs(
    data: list[float], 
    gapset: list[float], 
    precision: float,
    search_mode = "simple"):
    dqiips = list[tuple[int,int,int,int]]()
    # choose a search algorithm
    if search_mode == "binary":
        gap_search = gap_search_binary
    else:
        gap_search = gap_search_simple
    # sort data in ascending order
    data.sort()
    for key in gapset:
        # for each key, find the pairs (x,y) from data
        # such that `y - x` is within `precision` of `key`
        candidate_intervals = gap_search(data,key,precision)
        # filter for disjoint (nonintersecting) pairs of intervals. 
        for ((b1,b2),(y1,y2)) in unique_ascending_pairs(candidate_intervals):
            if data[b1] < data[2] < data[y1] < data[y2]:
                dqiips.append((b1,b2,y1,y2))
    return dqiips

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

def pivoting_interval_pairs(
    data: list[float], 
    gapset: list[float], 
    precision: float,
    search_mode = "simple"):
    # parametize the pivots over `data`
    pivot_indices = find_disjoint_quasiisometric_interval_pairs(
        data,gapset,precision,search_mode = search_mode)
    # curry of `pivot_from_data` with `data`
    def _pivot(indices):
        return pivot_from_data(indices,data)
    return map(_pivot,pivot_indices)
