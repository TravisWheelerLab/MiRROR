from copy import deepcopy

def collect_data(pivotmap):
    return [pivot.data() for pivot in deepcopy(pivotmap)]

def unique(a):
    return list(set(a))
    
def subset(A,B):
    return all([B.count(a) > 0 for a in A])