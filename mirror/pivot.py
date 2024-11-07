class Pivot:
    def __init__(self, pair_a, pair_b, indices_a, indices_b):
        self.index_data = sorted([*indices_a, *indices_b])
        self.data = sorted([*pair_a, *pair_b])

        self.indices_a = indices_a
        self.pair_a = pair_a
        self.gap_a = pair_a[1] - pair_a[0]
        
        self.indices_b = indices_b
        self.pair_b = pair_b
        self.gap_b = pair_b[1] - pair_b[0]

    def center(self):
        return sum(self.data) / 4
    
    def gap(self):
        return (self.gap_a + self.gap_b) / 2

    def __repr__(self):
        return f"gap:\t{self.gap()}\npeaks:\t{[round(x,3) for x in self.data]}"