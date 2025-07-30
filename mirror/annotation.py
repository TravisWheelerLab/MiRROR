def _annotate(
    peaks: PeakList,
    params: AnnotationParams,
) -> AnnotationResult:
    # find pairs of peaks whose m/z difference has a ResidueState in the ResidueStateSpace.
    peak_pairs = find_pairs(
        peaks = peaks,
        match_threshold = params.match_threshold,
        residue_state_space = params.residue_state_space)
    # find pairs of pairs (or equivalent four-peak structures) that reflect about a common point of symmetry.
    pivots = find_pivots(
        pairs = peak_pairs,
        peaks = peaks,
        difference_threshold = params.difference_threshold,
        search_strategies = params.search_strategies)
    # find boundary peaks. 
    ## LeftBoundaryPeaks have m/z that is within a shift transformation of a ResidueState.    
    left_boundaries = find_left_boundaries(
        peaks = peaks,
        match_threshold = params.match_threshold,
        residue_state_space = params.residue_state_space)
    ## RightBoundaryPeaks have m/z that is within a reflection and shift of a ResidueState.
    ## the reflection is parametized by a pivot, so right_boundaries is a second-order collection.
    right_boundaries = find_right_boundaries(
        pivots = pivots,
        peaks = peaks,
        match_threshold = params.match_threshold,
        residue_state_space = params.residue_state_space)
    # score, reorder, and filter pivots according to their symmetry and right boundary quality.
    pivots = filter_pivots(
        pivots = pivots,
        left_boundaries = left_boundaries,
        right_boundaries = right_boundaries,
        symmetry_threshold = params.pivot_symmetry_threshold,
        score_threshold = params.pivot_score_threshold)

def annotate(
    peak_lists: Iterator[PeakList],
    params: AnnotationParams,
) -> Iterator[AnnotationResult]:
    with Pool() as pool:
        return pool.starmap(
            annotate,
            zip(peak_lists, repeat(params, len(peak_lists))
