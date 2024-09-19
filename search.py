class Pivot:
    def __init__(self, pair_a, pair_b, target_gap):
        self.data = sorted([*pair_a, *pair_b])
        self.target_gap = target_gap

        self.pair_a = pair_a
        self.gap_a = pair_a[1] - pair_a[0]
        
        self.pair_b = pair_b
        self.gap_b = pair_b[1] - pair_b[0]

def _search_overlap_alt(spectrum: list[float], target_gap: float, tolerance: float):
    pivots = []
    n = len(spectrum)
    for i in range(n):
        p1 = spectrum[i]
        for j in range(i + 1, n):
            p2 = spectrum[j]
            p_gap = p2 - p1
            if p_gap > target_gap + tolerance:
                break
            elif p_gap < target_gap - tolerance:
                continue
            else:
                p_delta = abs(target_gap - p_gap)
                if p_delta < tolerance:
                    for k in range(i + 1, j):
                        q1 = spectrum[k]
                        for l in range(j + 1, n):
                            q2 = spectrum[l]
                            q_gap = q2 - q1
                            if q_gap > target_gap + tolerance:
                                break
                            elif q_gap < target_gap - tolerance:
                                continue
                            else:
                                pivots.append(Pivot((p1,p2),(q1,q2),target_gap))
    return pivots

def _find_gapped_pairs(spectrum: list[float], target_gap: float, tolerance: float):
    pairs = []
    n = len(spectrum)
    for i in range(n):
        p1 = spectrum[i]
        for j in range(i + 1, n):
            p2 = spectrum[j]
            pair_gap = p2 - p1
            if pair_gap > target_gap + tolerance:
                break
            elif pair_gap < target_gap - tolerance:
                continue
            else:
                pairs.append((p1,p2))
    return pairs

def _search_overlap(spectrum: list[float], target_gap: float, tolerance: float):
    candidate_pairs = _find_gapped_pairs(spectrum, target_gap, tolerance)
    pivots = []
    n = len(candidate_pairs)
    for i in range(n):
        p = candidate_pairs[i]
        for j in range(i + 1, n):
            q = candidate_pairs[j]
            if p[0] < q[0] < p[1] < q[1]:
                pivots.append(Pivot(p,q,target_gap))
    return pivots

def _search_disjoint(spectrum: list[float], target_gap: float, tolerance: float):
    candidate_pairs = _find_gapped_pairs(spectrum, target_gap, tolerance)
    pivots = []
    n = len(candidate_pairs)
    for i in range(n):
        p = candidate_pairs[i]
        for j in range(i + 1, n):
            q = candidate_pairs[j]
            if p[0] < p[1] < q[0] < q[1]:
                pivots.append(Pivot(p,q,target_gap))
    return pivots

def search(
    spectrum: list[float], 
    strategy: str, 
    target_gaps: list[float], 
    tolerance: float
):
    pivots = []
    if strategy == 'overlap':
        _search_strategy = _search_overlap
    elif strategy == 'overlap_alt':
        _search_strategy = _search_overlap_alt
    elif strategy == 'disjoint':
        _search_strategy = _search_disjoint
    else:
        raise ValueError(f"Unrecognized search strategy: `{strategy}`.") 
    for target_gap in target_gaps:
        pivots.extend(_search_strategy(spectrum, target_gap, tolerance))
    return pivots