import numpy as np

def merge_in_order(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    merged_array = np.hstack(arrays)
    order = np.argsort(merged_array)
    indices = np.hstack([np.arange(len(arr)) for arr in arrays])
    source = np.hstack([np.full_like(arrays[i], i, dtype=int) for i in range(len(arrays))])
    return merged_array[order], indices[order], source[order]
    #N = [len(A) for A in arrays]
    #I = [0 for _ in arrays]
    #m = len(arrays)
    #minimum = min(min(A) for A in arrays)
    #for _ in range(sum(N)):
    #    for k in range(m):
    #        i = I[k]
    #        n = N[k]
    #        arr = arrays[k]
