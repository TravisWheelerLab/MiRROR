import sys
from pivotpoint.pivotpoint import *
from pivotpoint.synth import *

"   subset(A,B) = A ⊆ B"
def subset(A,B):
    return all([B.count(a) > 0 for a in A])

seq = "KVILKETPFYAESGGQVADKGIIRGANGFAVVSDVQKAPNGQHLHTVIVKEGTLQVNDQVQAIVEETERS"
yb_spec = generate_spectrum_from_sequence(seq,parametize_yb_spectrum())
yba_spec = generate_spectrum_from_sequence(seq,parametize_yba_spectrum())
#full_spec = generate_spectrum_from_sequence(seq,parametize_full_spectrum())
print("yb spec:",yb_spec)
print("yba spec:", yba_spec)
#print("full spec:",full_spec)
assert subset(yb_spec,yba_spec)
#assert subset(yba_spec,full_spec)

max_gap = 250
yb_gapset = generate_gapset_from_sequence(seq,parametize_yb_spectrum(),max_gap=max_gap)
yba_gapset = generate_gapset_from_sequence(seq,parametize_yba_spectrum(),max_gap=max_gap)
#full_gapset = generate_gapset_from_sequence(seq,parametize_full_spectrum(),max_gap=max_gap)
#print("n full gaps <", max_gap, ":\t",len(full_gapset))
assert subset(yb_gapset,yba_gapset)
#assert subset(yba_gapset,full_gapset)
print("\nyb gaps")
yb_gapset.sort()
for gap in yb_gapset:
    print(gap,end = ', ')
print()
print("\nyba gaps")
yba_gapset.sort()
for gap in yba_gapset:
    print(gap,end = ', ')
print()

precision = 0.1
yb_yb = pivoting_interval_pairs(yb_spec,yb_gapset,precision)
yb_yba = pivoting_interval_pairs(yb_spec,yba_gapset,precision)
yba_yb = pivoting_interval_pairs(yba_spec,yb_gapset,precision)
yba_yba = pivoting_interval_pairs(yba_spec,yba_gapset,precision)

def collect_data(pivotmap):
    return [pivot.data() for pivot in pivotmap]

yb_yb = collect_data(yb_yb)
yb_yba = collect_data(yb_yba)
yba_yb = collect_data(yba_yb)
yba_yba = collect_data(yba_yba)
print()
print("seq:", seq)
print("n yb gaps <", max_gap, "\t",len(yb_gapset))
print("n yba gaps <", max_gap, "\t",len(yba_gapset))
print("yb_yb  ⊆ yb_yba  =", subset(yb_yb, yb_yba))
print("yba_yb ⊆ yba_yba =", subset(yb_yb, yb_yba))
print("yb_yb  ⊆ yba_yb  =", subset(yb_yb, yb_yba))
print("yba_yb ⊆ yba_yba =", subset(yba_yb, yba_yba))

'''
yb_pivots = [pivot.data() for pivot in pivoting_interval_pairs(yb_spec,yb_gapset,precision)]
yb_pivots.sort()
yba_pivots = [pivot.data() for pivot in pivoting_interval_pairs(yba_spec,yba_gapset,precision)]
yba_pivots.sort()
#full_pivots = [pivot.data() for pivot in pivoting_interval_pairs(full_spec,full_gapset,precision)]
#full_pivots.sort()
print("\nn pivots expected minimum:\t\t", fld(len(seq),2))
print("n pivots | yb, with precision", precision, ":\t", len(yb_pivots))
print("n pivots | yba, with precision", precision, ":\t", len(yba_pivots))
#print("n pivots | full, with precision", precision, ":\t", len(full_pivots))

print("\npivots | yb")
for pivot in yb_pivots:
    print(pivot)
print("pivots | yba)")
for pivot in yba_pivots:
    print(pivot)

assert subset(yb_pivots,yba_pivots)
#assert subset(yba_pivots,full_pivots)

# lo = 0
# hi = 100
# nsample = 75
# gapset = [2,4,6,8,10]
# data = [random.randint(lo,hi) for _ in range(nsample)]
# precision = 0.5
# pivots = pivoting_interval_pairs(data,gapset,precision)
# print([p.data() for p in pivots])
'''