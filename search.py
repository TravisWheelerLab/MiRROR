def _search_overlap2(spectrum: list[float], tolerance: float, key: float):
    # identify candidate pairs with gap ~= a mass in the alphabet
    n = len(spectrum)
    candidate_pairs = []
    for i in range(n):
        p1 = spectrum[i]
        for j in range(i + 1, n):
            p2 = spectrum[j]
            p_gap = p2 - p1
            if p_gap - tolerance > key:
                break
            elif p_gap + tolerance < key:
                continue
            else:
                p_delta = abs(key - p_gap)
                if p_delta < tolerance:
                    candidate_pairs.append((p1,p2))
    # among candidate pairs, identify overlapping and disjoint pairs
    m = len(candidate_pairs)
    pivots_overlap = []
    for i in range(m):
        p1, p2 = candidate_pairs[i]
        for j in range(i + 1, m):
            q1, q2 = candidate_pairs[j]            
            if q1 > p2:
                break
            elif p1 < q1 < p2 < q2:
                pivots_overlap.append((p1,q1,p2,q2))

    return pivots_overlap

def search_overlap2(spectrum: list[float], tolerance: float, alphabet: list[float]):
    pivots_overlap = []
    for key in alphabet:
        p_o = _search_overlap2(spectrum, tolerance, key)
        pivots_overlap.extend(p_o)
    return pivots_overlap
    
def _search_overlap(spectrum: list[float], tolerance: float, key: float):
    pivots = []
    n = len(spectrum)
    for i in range(n):
        p1 = spectrum[i]
        for j in range(i + 1, n):
            p2 = spectrum[j]
            p_gap = p2 - p1
            if p_gap - tolerance > key:
                break
            elif p_gap + tolerance < key:
                continue
            else:
                p_delta = abs(key - p_gap)
                if p_delta < tolerance:
                    for k in range(i + 1, j):
                        q1 = spectrum[k]
                        for l in range(j + 1, n):
                            q2 = spectrum[l]
                            q_gap = q2 - q1
                            if q_gap - tolerance > key:
                                break
                            elif q_gap + tolerance < key:
                                continue
                            else:
                                pivots.append((p1,q1,p2,q2))
    return pivots

def search_overlap(spectrum: list[float], tolerance: float, alphabet: list[float]):
    pivots = []
    for key in alphabet:
        p_o = _search_overlap(spectrum, tolerance, key)
        pivots.extend(p_o)
    return pivots

def _search(spectrum: list[float], tolerance: float, key: float):
    # identify candidate pairs with gap ~= a mass in the alphabet
    n = len(spectrum)
    candidate_pairs = []
    for i in range(n):
        p1 = spectrum[i]
        for j in range(i + 1, n):
            p2 = spectrum[j]
            p_gap = p2 - p1
            delta = abs(key - p_gap)
            if delta < tolerance:
                candidate_pairs.append((p1,p2))
    # among candidate pairs, identify overlapping and disjoint pairs
    m = len(candidate_pairs)
    pivots_overlap = []
    pivots_disjoint = []
    for i in range(m):
        p1, p2 = candidate_pairs[i]
        for j in range(i + 1, m):
            q1, q2 = candidate_pairs[j]
            if p1 < q1 < p2 < q2:
                pivots_overlap.append((p1,q1,p2,q2))
            if p1 < p2 < q1 < q2:
                pivots_disjoint.append((p1,p2,q1,q2))
    return pivots_overlap, pivots_disjoint

def search(spectrum: list[float], tolerance: float, alphabet: list[float]):
    pivots_overlap = []
    pivots_disjoint = []
    for key in alphabet:
        p_o, p_d = _search(spectrum, tolerance, key)
        pivots_overlap.extend(p_o)
        pivots_disjoint.extend(p_d)
    return pivots_overlap, pivots_disjoint

# NOTE: both search modes are being performed simultaneously for convenience of testing
# in practice, overlap is performed first and identifies a handful of candidates.
# disjoint will identify many, many more, and should only be used when necessary.