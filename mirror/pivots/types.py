from dataclasses import dataclass

@dataclass
class PivotParams:
    tolerance: float

class Pivot:
    """Interface to a pivot identified from a pair of gaps."""

    def __init__(self, pair_a, pair_b, indices_a, indices_b):
        self._indices_a = indices_a
        self._indices_b = indices_b
        self._pair_a = pair_a
        self._pair_b = pair_b
        
        self.index_data = sorted([*indices_a, *indices_b])
        self.data = sorted([*pair_a, *pair_b])
    
    def peaks(self):
        return self.data

    def indices(self):
        return self.index_data

    def peak_pairs(self):
        return self._pair_a, self._pair_b
    
    def index_pairs(self):
        return self._indices_a, self._indices_b
    
    def outer_left(self):
        return self.index_data[0]
    
    def inner_left(self):
        return self.index_data[1]
    
    def inner_right(self):
        return self.index_data[2]
    
    def outer_right(self):
        return self.index_data[3]

    def center(self):
        return sum(self.data) / 4

    def residue(self):
        return residue_lookup(self.gap())
    
    def gap(self):
        peak_pairs = self.peak_pairs()
        gap_a = peak_pairs[0][1] - peak_pairs[0][0]
        gap_b = peak_pairs[1][1] - peak_pairs[1][0]
        return (gap_a + gap_b) / 2
    
    def negative_index_pairs(self):
        """index pairs that should not be present in the gap set."""
        inds_a, inds_b = self.index_pairs()
        negative_pairs = [
            (inds_a[0], inds_b[0]),
            (inds_a[1], inds_b[1]),
            (inds_a[1], inds_b[0]),
            (inds_a[0], inds_b[1])]
        negative_pairs = [(min(e), max(e)) for e in negative_pairs]
        return negative_pairs

    def __repr__(self):
        return f"""Pivot(
\tpeaks = {self.peak_pairs()}
\tindices = {self.index_pairs()}
\tresidue = {self.residue()}
)"""
    
    def __eq__(self, other):
        return self.peaks() == other.peaks()