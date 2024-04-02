def pivot_peaks(
    pivots: list[PivotingIntervalPair],
    precision: float,
    peak_finding_routine: str = "scipy"):

    # from a `list[PivotingIntervalPair]`, produce a `list[float]` of its centers.
# construct a histogram from the centers; find its peaks (either iteratively, or not)
# assign each `PivotingIntervalPair` to a peak in the centers histogram.
# sort each cluster by proximity to its peak.
# wrap the resulting data as `PivotPointPeakCluster` objects.

# from a `PivotPointPeakCluster`, perform branch and bound search for compatible amino sequences.