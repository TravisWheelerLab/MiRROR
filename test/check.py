import sys
from pivotpoint.pivotpoint import *
import random
lo = 0
hi = 100
nsample = 75
gapset = [2,4,6,8,10]
data = [random.randint(lo,hi) for _ in range(nsample)]
precision = 0.5
dqiips = find_disjoint_quasiisometric_interval_pairs(data,gapset,precision)
pivots = [pivot_from_data(indices,data) for indices in dqiips]
print([p.data for p in pivots])