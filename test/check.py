import sys
from pivotpoint.pivotpoint import *
from pivotpoint.synth import *

seq = "DFPIANGERDFAD"
yb_spec = generate_spectrum_from_sequence(seq,parametize_yb_spectrum())
yba_spec = generate_spectrum_from_sequence(seq,parametize_yba_spectrum())
full_spec = generate_spectrum_from_sequence(seq,parametize_full_spectrum())
print("seq:", seq)
print("yb spec:",yb_spec)
print("yba spec:", yba_spec)
print("full spec:",full_spec)
assert all([yba_spec.count(y_or_b) > 0 for y_or_b in yb_spec]), "failed to prove tautology (Y-ions ∪ B-ions) ⊆ (Y-ions ∪ B-ions ∪ A-ions)"
assert all([full_spec.count(y_or_b_or_a) > 0 for y_or_b_or_a in yba_spec]), "failed to prove tautology (Y-ions ∪ B-ions ∪ A-ions) ⊆ (Full spectrum)"

max_gap = 150
yb_gapset = generate_gapset_from_sequence(seq,parametize_yb_spectrum(),max_gap=max_gap)
yba_gapset = generate_gapset_from_sequence(seq,parametize_yba_spectrum(),max_gap=max_gap)
full_gapset = generate_gapset_from_sequence(seq,parametize_full_spectrum(),max_gap=max_gap)
print("n yb gaps <", max_gap, ":\t",len(yb_gapset))
print("n yba gaps <", max_gap, ":\t",len(yba_gapset))
print("n full gaps <", max_gap, ":\t",len(full_gapset))

precision = 0.1
yb_pivots = find_disjoint_quasiisometric_interval_pairs(yb_spec,yb_gapset,precision)
yba_pivots = find_disjoint_quasiisometric_interval_pairs(yba_spec,yba_gapset,precision)
full_pivots = find_disjoint_quasiisometric_interval_pairs(full_spec,full_gapset,precision)
print("\nn pivots expected minimum:\t\t", fld(len(seq),2))
print("n pivots | yb, with precision", precision, ":\t", len(yb_pivots))
print("n pivots | yba, with precision", precision, ":\t", len(yba_pivots))
print("n pivots | full, with precision", precision, ":\t", len(full_pivots))

# lo = 0
# hi = 100
# nsample = 75
# gapset = [2,4,6,8,10]
# data = [random.randint(lo,hi) for _ in range(nsample)]
# precision = 0.5
# pivots = pivoting_interval_pairs(data,gapset,precision)
# print([p.data() for p in pivots])
