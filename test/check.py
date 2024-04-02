import sys
from pivotpoint import *

"   subset(A,B) = A ⊆ B"
def subset(A,B):
    return all([B.count(a) > 0 for a in A])

seq = "KVILKETPFYAESGGQVADKGIIRGANGFAVVSDVQKAPNGQHLHTVIVKEGTLQVNDQVQAIVEETERS"
yb_spec = generate_spectrum_from_sequence(seq,parametize_yb_spectrum())
yba_spec = generate_spectrum_from_sequence(seq,parametize_yba_spectrum())
full_spec = generate_spectrum_from_sequence(seq,parametize_full_spectrum())
print("yb spec:",yb_spec)
print("yba spec:", yba_spec)
print("full spec:",full_spec)
assert subset(yb_spec,yba_spec)
assert subset(yba_spec,full_spec)

max_gap = 250
yb_gapset = generate_gapset_from_sequence(seq,parametize_yb_spectrum(),max_gap=max_gap)
yba_gapset = generate_gapset_from_sequence(seq,parametize_yba_spectrum(),max_gap=max_gap)
full_gapset = generate_gapset_from_sequence(seq,parametize_full_spectrum(),max_gap=max_gap)
assert subset(yb_gapset,yba_gapset)
assert subset(yba_gapset,full_gapset)

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
yb_yb = find_pivoting_interval_pairs(yb_spec,yb_gapset,precision)
yb_yba = find_pivoting_interval_pairs(yb_spec,yba_gapset,precision)
yba_yb = find_pivoting_interval_pairs(yba_spec,yb_gapset,precision)
yba_yba = find_pivoting_interval_pairs(yba_spec,yba_gapset,precision)

def collect_data(pivotmap):
    return [pivot.data() for pivot in pivotmap]

yb_yb_data = collect_data(yb_yb)
yb_yba_data = collect_data(yb_yba)
yba_yb_data = collect_data(yba_yb)
yba_yba_data = collect_data(yba_yba)
print()
print("seq:", seq)
print("n yb gaps <", max_gap, "\t",len(yb_gapset))
print("n yba gaps <", max_gap, "\t",len(yba_gapset))
print("yb_yb  ⊆ yb_yba  =", subset(yb_yb_data, yb_yba_data))
print("yba_yb ⊆ yba_yba =", subset(yb_yb_data, yb_yba_data))
print("yb_yb  ⊆ yba_yb  =", subset(yb_yb_data, yb_yba_data))
print("yba_yb ⊆ yba_yba =", subset(yba_yb_data, yba_yba_data))

yb_yb_clusters = find_pivot_clusters(yb_yb,1000)
yb_yba_clusters = find_pivot_clusters(yb_yba,1000)
yba_yb_clusters = find_pivot_clusters(yba_yb,1000)
yba_yba_clusters = find_pivot_clusters(yba_yba,1000)