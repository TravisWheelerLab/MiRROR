import sys
from pivotpoint.pivotpoint import *
import random
lo = 0
hi = 100
nsample = 75
gapset = [2,4,6,8,10]
data = [random.randint(lo,hi) for _ in range(nsample)]
precision = 0.5
pivots = pivoting_interval_pairs(data,gapset,precision)
print([p.data() for p in pivots])