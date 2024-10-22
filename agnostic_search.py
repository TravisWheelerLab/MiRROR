def _find_gapped_pairs(spectrum: list[float], min_gap: float, max_gap: float, tolerance: float):
    pairs = []
    n = len(spectrum)
    for i in range(n):
        p1 = spectrum[i]
        for j in range(i + 1, n):
            p2 = spectrum[j]
            pair_gap = p2 - p1
            if pair_gap > max_gap + tolerance:
                break
            elif pair_gap < min_gap - tolerance:
                continue
            else:
                pairs.append(((p1,p2),(i,j)))
    return sorted(pairs, key = lambda x: x[1] - x[0])

def _search_overlap(spectrum: list[float], min_gap: float, max_gap: float, tolerance: float, intergap_tolerance = 0.01):
    candidate_pairs = _find_gapped_pairs(spectrum, min_gap, max_gap, tolerance)
    pivots = []
    n = len(candidate_pairs)
    for i in range(n):
        p, indices_p = candidate_pairs[i]
        p_gap = p[1] - p[0]
        for j in range(i + 1, n):
            q, indices_q = candidate_pairs[j]
            q_gap = q[1] - q[0]
            if q_gap - p_gap > intergap_tolerance:
                break
            elif (abs(p_gap - q_gap) < intergap_tolerance):
                if (p[0] < q[0] < p[1] < q[1]):
                    pivots.append(Pivot(p,q,indices_p,indices_q,(p_gap + q_gap) / 2))
                elif (q[0] < p[0] < q[1] < p[1]):
                    pivots.append(Pivot(q,p,indices_q,indices_p,(p_gap + q_gap) / 2))
    return pivots

def _search_disjoint(spectrum: list[float], min_gap: float, max_gap: float, tolerance: float, intergap_tolerance = 0.01):
    candidate_pairs = _find_gapped_pairs(spectrum, min_gap, max_gap, tolerance)
    pivots = []
    n = len(candidate_pairs)
    for i in range(n):
        p, indices_p = candidate_pairs[i]
        p_gap = p[1] - p[0]
        for j in range(i + 1, n):
            q, indices_q = candidate_pairs[j]
            q_gap = q[1] - q[0]
            if q_gap - p_gap > intergap_tolerance:
                break
            elif (abs(p_gap - q_gap) < intergap_tolerance):
                if (p[0] < p[1] < q[0] < q[1]):
                    pivots.append(Pivot(p,q,indices_p,indices_q,(p_gap + q_gap) / 2))
                elif (q[0] < q[1] < p[0] < p[1]):
                    pivots.append(Pivot(q,p,indices_q,indices_p,(p_gap + q_gap) / 2))
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
    elif strategy == 'disjoint':
        _search_strategy = _search_disjoint
    else:
        raise ValueError(f"Unrecognized search strategy: `{strategy}`.") 
    return _search_strategy(spectrum, min(target_gaps), max(target_gaps), tolerance)