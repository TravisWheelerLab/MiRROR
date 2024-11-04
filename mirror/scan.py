class ScanConstraint:

    def __init__(self, parameters):
        pass
    
    # `stop` and `match` use the same arguments.
    # to avoid redundancy, `evaluate` precomputes those inputs.
    def evaluate(self, state):
        return None

    def stop(self, val):
        return True

    def match(self, val):
        return False

# searches the product space `arr` x `arr` for pairs satisfying the condition 
# defined in `constraint`, which also defines the stop condition that prevents 
# quadratic time complexity.
def constrained_pair_scan(
    arr: list,
    constraint: ScanConstraint,
    outer_loop_range = lambda size: (0, size),
    inner_loop_range = lambda size,idx: (idx + 1, size),
):
    n = len(arr)
    outer_lo, outer_hi = outer_loop_range(n)
    for i in range(outer_lo, outer_hi):
        inner_lo, inner_hi = inner_loop_range(n, i)
        for j in range(inner_lo, inner_hi):
            val = constraint.evaluate((arr[i],arr[j]))
            if constraint.stop(val):
                break
            elif constraint.match(val):
                yield (i,j)