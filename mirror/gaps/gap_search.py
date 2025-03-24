from .gap_types import *
from bisect import bisect_left, bisect_right

def _find_gaps_without_targets():
    """TODO"""
    pass

def duplicate_inverse_charges(mz, charge_values):
    charge_values = charge_values
    # create charge table
    mz_idx = np.arange(len(mz))
    charge_states = np.hstack([np.full_like(mz, 1.)] + [np.full_like(mz, c) for c in charge_values])
    indices = np.hstack([mz_idx for _ in range(len(charge_values) + 1)])
    charge_table = np.vstack((
        indices, 
        charge_states,
    ))
    # create duplicate peaks with inverse charges
    dup_mz = np.hstack([mz] + [mz * c for c in charge_values])
    # reorder
    order = np.argsort(dup_mz)
    return dup_mz[order], charge_table[:, order]

def _create_transformation_tensors(
    masses: list[float],
    losses: list[float],
    modifications: list[float],
):
    # mass tensor - one dimensional along axis 0.
    n_masses = len(masses)
    mass_tensor = masses.reshape(n_masses, 1, 1, 1)
    max_mass = masses.max()
    # inverse loss tensors - one dimensional, respectively along axes 1 and 2.
    n_losses = len(losses) + 1
    left_loss_tensor = -1 * np.concatenate(([-0.0], losses)).reshape(1, n_losses, 1, 1)
    right_loss_tensor = -1 * np.concatenate(([-0.0], losses)).reshape(1, 1, n_losses, 1)
    # inverse modification tensor - one dimensional along axis 3.
    n_modifications = len(modifications) + 1
    modification_tensor = -1 * np.concatenate(([-0.0], modifications)).reshape(1, 1, 1, n_modifications)
    # compute the extremal delta
    min_loss = -left_loss_tensor.max()
    max_loss = -right_loss_tensor.min()
    max_modification = -modification_tensor.min()
    extremal_delta = max_mass + max_loss + max_modification - min_loss
    return mass_tensor, left_loss_tensor, right_loss_tensor, modification_tensor, extremal_delta

def _get_solver(
    solver_type: Type[GapAbstractTransformationSolver],
    mz: np.ndarray,
    mass_tensor: np.ndarray,
    left_loss_tensor: np.ndarray,
    right_loss_tensor: np.ndarray,
    modification_tensor: np.ndarray,
    extremal_delta: float,
    tolerance: float,
) -> GapAbstractTransformationSolver:
    return solver_type(
        mz,
        mass_tensor,
        left_loss_tensor,
        right_loss_tensor,
        modification_tensor,
        extremal_delta,
        tolerance,
    )

class GapTensorTransformationSolver(GapAbstractTransformationSolver):
    def __post_init__(self):
        """self.modification_tensor can be very large, so there's no reason to compute self.right_loss_tensor + self.modification_tensor more than once."""
        self._left_tensor = self.left_loss_tensor.reshape(1, *self.left_loss_tensor.shape)
        self._right_tensor = self.right_loss_tensor + self.modification_tensor
        self._right_tensor = self._right_tensor.reshape(1, *self._right_tensor.shape)
        self._mass_tensor = self.mass_tensor.reshape(1, *self.mass_tensor.shape)
        # initialize dynamic fields
        self._outer_index = -1
        self._inner_index = -1
        self._lower_inner_index = -1
        self._upper_inner_index = -1
        self._results = None
        self._local_results = None
    
    def _inner_upper_bound(self) -> int:
        idx = bisect_right(self.mz, self.mz[self._outer_index] + self.extremal_delta)
        return max(0, min(self.mz.size - 1, idx + 1))
    
    def _construct_results(self) -> None:
        left_peaks = self.mz[self._outer_index] + self._left_tensor
        self._lower_inner_index = self._outer_index + 1
        self._upper_inner_index = self._inner_upper_bound()
        right_queries = self.mz[self._lower_inner_index: self._upper_inner_index + 1]
        right_queries = right_queries.reshape(right_queries.size, 1, 1, 1, 1)
        right_peaks = right_queries + self._right_tensor
        differences = right_peaks - left_peaks
        differences[differences < 0] = np.inf
        self._results = np.abs(differences - self._mass_tensor)
        
    def set_outer_index(self, outer_index: int) -> None:
        self._outer_index = outer_index
        self._construct_results()

    def _construct_local_results(self) -> None:
        self._local_results = self._results[self._local_inner_index, :, :, :, :]
        
    def set_inner_index(self, inner_index: int) -> None:
        self._local_inner_index = inner_index - self._lower_inner_index
        self._construct_local_results()
        
    def get_solutions(self) -> tuple[int, int, int, int]:
        soln_tensor = self._local_results <= self._local_results.min() + self.tolerance
        return self._filter_solution(zip(*(soln_tensor).nonzero()))
        
class GapBisectTransformationSolver(GapAbstractTransformationSolver):    
    def __post_init__(self):
        """construct the target mass values from the transformation tensors"""
        # transform target masses
        subtractive_transformations = self.modification_tensor + self.right_loss_tensor
        transformed_masses = self.mass_tensor + self.left_loss_tensor - subtractive_transformations
        self._n = transformed_masses.size
        self._shape = transformed_masses.shape
        self._transformed_target_masses = transformed_masses.flatten()
        indices = np.arange(self._n)
        self._unraveled_indices = np.array([np.unravel_index(i, self._shape) for i in indices])
        # reorder
        order = np.argsort(self._transformed_target_masses)
        self._transformed_target_masses = self._transformed_target_masses[order]
        self._unraveled_indices = [tuple(i) for i in self._unraveled_indices[order]]
        # initialize dynamic fields
        self._left_peak = np.nan
        self._match_lo = -1
        self._match_hi = -1
    
    def _unravel(self, i):
        return self._unraveled_indices[i]

    def _bound_bisection(self, i):
        return max(0, min(self._n - 1, i))
    
    def _bisect_range(self, query: float):
        return (self._bound_bisection(bisect_left(self._transformed_target_masses, query - self.tolerance)),
            self._bound_bisection(bisect_right(self._transformed_target_masses, query + self.tolerance)))

    def set_outer_index(self, outer_index: int) -> None:
        self._left_peak = self.mz[outer_index]

    def set_inner_index(self, inner_index: int) -> None:
        query = self.mz[inner_index] - self._left_peak
        self._match_lo, self._match_hi = self._bisect_range(query)
    
    def get_solutions(self) -> tuple[int, int, int, int]:
        return self._filter_solution(map(self._unravel, range(self._match_lo, self._match_hi + 1)))

def _find_gaps(
    solver_type: Type[GapAbstractTransformationSolver],
    mz: np.ndarray,
    masses: list[float],
    losses: list[float],
    modifications: list[float],
    charge_table: np.ndarray,
    tolerance: float,
    alphabet: list[str],
    barstr = '-'*48,
    verbose = False,
    wait_for_input = False,
):
    wait = input if wait_for_input else lambda: None
    n = len(mz)
    # construct transformation tensors
    mass_tensor, left_loss_tensor, right_loss_tensor, modification_tensor, extremal_delta = _create_transformation_tensors(masses, losses, modifications)
    transformation_solver = _get_solver(
        solver_type,
        mz, 
        mass_tensor, 
        left_loss_tensor, 
        right_loss_tensor,
        modification_tensor,
        extremal_delta,
        tolerance)
    for i in range(n - 1):
        transformation_solver.set_outer_index(i)
        deduplicated_i = charge_table[0,i].astype(int)
        for j in range(i + 1, n):
            min_delta = mz[j] - mz[i] - extremal_delta
            # early stopping criteria
            # - if `min_delta` is greater than the threshold, 
            #   all subsequent values for `j` will also be greater than the threshold,
            #   so there's no reason to continue the inner loop.
            if min_delta > tolerance:
                if verbose:
                    print(f"\nmin delta {min_delta} > tolerance {tolerance}; terminating inner loop.")
                    wait()
                break   
            # otherwise, continue         
            transformation_solver.set_inner_index(j)
            deduplicated_j = charge_table[0,j].astype(int)
            if deduplicated_i == deduplicated_j:
                continue
            min_delta = np.inf
            solns = list(transformation_solver.get_solutions())
            if verbose:
                print(f"\n# solutions[{deduplicated_i}, {deduplicated_j} = ({i}, {j})]\t = {len(solns)}")
            for (target_optimizer, left_optimizer, right_optimizer, modification_optimizer) in solns:
                # optimal target
                target_residue = alphabet[target_optimizer]
                target_mass = mass_tensor[target_optimizer, 0, 0, 0]

                # optimal left loss
                query_left_loss = left_loss_tensor[0, left_optimizer, 0, 0]
                
                # optimal right loss
                query_right_loss = right_loss_tensor[0, 0, right_optimizer, 0]
                
                # optimal modification
                query_modification = modification_tensor[0, 0, 0, modification_optimizer]

                # calculate query mass
                # - the left/right distinction is mainly for numeric stability.
                #   every subtraction introduces instability; do as few as possible.
                left_transformed_peak = mz[i] + query_left_loss
                right_transformed_peak = mz[j] + query_right_loss + query_modification
                query_mass = right_transformed_peak - left_transformed_peak

                # calculate deltas
                delta_mass = query_mass - target_mass
                
                if verbose:
                    print(f"\nsolution[{deduplicated_i}, {deduplicated_j} = ({i}, {j})]\ntarget:\t{target_residue, target_mass}\nquery:\t{query_mass}\nleft loss:\t{query_left_loss}\nright loss:\t{query_right_loss}\nmodification:\t{query_modification}\ndelta:\t{delta_mass}")
                # match criteria
                if abs(delta_mass) <= tolerance:
                    charge_state = tuple(charge_table[1,(i,j)])
                    target_idx = target_optimizer
                    left_loss_idx = left_optimizer - 1
                    right_loss_idx = right_optimizer - 1
                    modification_idx = modification_optimizer - 1
                    gap_match = GapMatch(
                        (-1,-1),
                        (deduplicated_i,deduplicated_j),
                        (i, j), 
                        charge_state,
                        target_residue, 
                        target_mass, 
                        target_idx,
                        query_mass, 
                        query_left_loss, 
                        left_loss_idx, 
                        query_right_loss, 
                        right_loss_idx, 
                        query_modification,
                        modification_idx,
                    )
                    if verbose:
                        print(f"\nmatch!\n{gap_match}")
                        wait()
                    yield gap_match
                elif verbose:
                    print(f"\nno match!")
                    wait()

def find_gaps(
    gap_params: GapSearchParameters,
    mz: np.ndarray,
    mode = "tensor",
    **kwargs
):
    mode, residues, masses, losses, modifications, charges, tolerance = gap_params.collect()
    dup_mz, charge_table = duplicate_inverse_charges(mz, charges)
    # branch to the appropriate subroutine.
    args = [dup_mz, masses, losses, modifications, charge_table, tolerance, residues]
    if mode == "tensor":
        match_generator = _find_gaps(GapTensorTransformationSolver, *args, **kwargs)
    elif mode == "bisect":
        match_generator = _find_gaps(GapBisectTransformationSolver, *args, **kwargs)
    else:
        raise ValueError(f"Unsupported search mode {mode}")
    # binsort outputs by match residue
    residues = list(residues)
    bins = {residue: [] for residue in residues + ['X']}
    for match in match_generator:
        match_residue = match.match_residue
        if match_residue in residues:
            bins[match_residue].append(match)
        else:
            bins['X'].append(match)
    # 
    for residue in residues + ['X']:
        for match in bins[residue]:
            i, j = match.inner_index
            assert dup_mz[j] - dup_mz[i] > 0 
            match.index_pair = match.inner_index
    # construct GapResult objects
    return dup_mz, [GapResult(bins[residue]) for residue in residues + ['X']]

def _find_gaps_tensor(
    mz: np.ndarray,
    masses: list[float],
    losses: list[float],
    modifications: list[float],
    charge_table: np.ndarray,
    tolerance: float,
    alphabet: list[str],
    barstr = '-'*48,
    verbose = False,
    wait_for_input = False,
):
    wait = input if wait_for_input else lambda: None
    n = len(mz)
    # construct transformation tensors
    mass_tensor, left_loss_tensor, right_loss_tensor, modification_tensor, __ = _create_transformation_tensors(masses, losses, modifications)
    max_mass = mass_tensor.max()
    if verbose:
        print(barstr)
        print(f"mz:\n{mz}")
        print(f"masses:\n{mass_tensor}")
        print(f"left losses:\n{left_loss_tensor}\nright losses:\n{right_loss_tensor}")
        print(f"modifications:\n{modification_tensor}")
        print(barstr)
        wait()
    for i in range(n - 1):
        deduplicated_i = charge_table[0,i].astype(int)
        # left peak tensor (one dimensional on axis 1).
        left_peaks = mz[i] + left_loss_tensor
        for j in range(i + 1, n):
            deduplicated_j = charge_table[0,j].astype(int)
            # right peak tensor (two dimensional on axes 2 and 3).
            right_peaks = mz[j] + right_loss_tensor + modification_tensor
            
            # right-left difference tensor (three dimensional on axes 1,2,3).
            differences = right_peaks - left_peaks
            differences[differences < 0] = np.inf
            
            # early termination criteria
            min_delta = differences.min() - max_mass

            # result tensor (four dimensional on axes 0,1,2,3).
            results = np.abs(differences - mass_tensor)
            
            # result tensor minimizer as a tuple indexing axes 0,1,2,3.
            optimizer = np.unravel_index(results.argmin(), results.shape)
            target_optimizer, left_optimizer, right_optimizer, modification_optimizer = optimizer
            optimum = results[optimizer]
            
            # optimal target 
            target_mass = mass_tensor[target_optimizer, 0, 0, 0]
            target_residue = alphabet[target_optimizer]
            
            # optimal difference
            query_mass = differences[0, left_optimizer, right_optimizer, modification_optimizer]

            # optimal left loss
            query_left_loss = left_loss_tensor[0, left_optimizer, 0, 0]
            
            # optimal right loss
            query_right_loss = right_loss_tensor[0, 0, right_optimizer, 0]
            
            # optimal modification
            query_modification = modification_tensor[0, 0, 0, modification_optimizer]
            
            if verbose:
                print(f"left peaks[{i}]:\n{left_peaks}")
                print(f"right peaks[{j}]:\n{right_peaks}")
                print(f"differences:\n{differences}")
                print(f"results:\n{results}")
                print(f"optimum[{i}, {j}] (topological: {deduplicated_i,deduplicated_j}):\n\tresults{optimizer} = {optimum}")
                print(barstr)
            delta_mass = query_mass - target_mass
            if abs(delta_mass) <= tolerance:
                charge_state = tuple(charge_table[1,(i,j)])
                target_idx = target_optimizer
                left_loss_idx = left_optimizer - 1
                right_loss_idx = right_optimizer - 1
                modification_idx = modification_optimizer - 1
                gap_match = GapMatch(
                    (-1,-1),
                    (deduplicated_i,deduplicated_j),
                    (i, j), 
                    charge_state,
                    target_residue, 
                    target_mass, 
                    target_idx,
                    query_mass, 
                    query_left_loss, 
                    left_loss_idx, 
                    query_right_loss, 
                    right_loss_idx, 
                    query_modification,
                    modification_idx,
                )
                if verbose:
                    print(f"\tmatch!")
                    print(gap_match)
                    print(barstr)
                    wait()
                yield gap_match
            elif min_delta > 0:
                if verbose:
                    print(f"\tno match; min_delta = {min_delta} > 0; terminating inner loop.")
                    print(barstr)
                    wait()
                break
            elif verbose:
                print(f"\tno match; min_delta = {min_delta} < 0; continuing inner loop.")
                print(barstr)
                wait()

def _bound_bisection(n, idx):
    return max(0, min(n - 1, idx))

def _bisect_range(
    sorted_arr: np.ndarray,
    query: float,
    tolerance: float,
):
    _l = bisect_left(sorted_arr, query - tolerance)
    l = _bound_bisection(sorted_arr.size, _l)
    _r  = bisect_right(sorted_arr, query + tolerance)
    r = _bound_bisection(sorted_arr.size, _r)
    #print(f"target values:\n{sorted_arr}\nquery value:\n{query} ≈ {query - tolerance, query + tolerance}\nraw range:\n{_l,_r}\nbounded range:\n{l, r}")
    return (l, r)

def _find_gaps_bisect(
    mz: np.ndarray,
    masses: np.ndarray,
    losses: np.ndarray,
    modifications: np.ndarray,
    charge_table: np.ndarray,
    tolerance: float,
    alphabet: list[str],
    verbose = False,
    barstr = '-'*48,
    wait_for_input = False,
):
    if not(np.all(np.diff(mz) >= 0)):
        raise ValueError("The spectrum array must be sorted in ascending order!")
    
    if not(tolerance > 1e-14):
        raise ValueError("The tolerance parameter must be greater than 1e-14, otherwise floating-point errors will induce false negatives.")
    
    n = len(mz)
    wait = input if wait_for_input else lambda: None
    # construct transformation tensors
    mass_tensor, left_loss_tensor, right_loss_tensor, modification_tensor, extremal_delta = _create_transformation_tensors(masses, losses, modifications)
    # create target space
    loss_matrix = left_loss_tensor - right_loss_tensor
    loss_matrix[:, 1:, 1:, :][loss_matrix[:, 1:, 1:, :] == 0] = np.inf
    transformed_masses = mass_tensor + loss_matrix - modification_tensor
    transformed_masses_shape = transformed_masses.shape
    # flatten and sort, keeping track of true indices
    flat_transformed_masses = transformed_masses.flatten()
    indices = np.arange(flat_transformed_masses.size)
    unraveled_indices = np.array([np.unravel_index(idx, transformed_masses_shape) for idx in indices])
    order = np.argsort(flat_transformed_masses)
    flat_transformed_masses = flat_transformed_masses[order]
    indices = indices[order]
    unraveled_indices = [tuple(i) for i in unraveled_indices[order]]
    for idx in range(flat_transformed_masses.size):
        unraveled_idx = unraveled_indices[idx]
        if verbose:
            print(f"targets[{unraveled_idx}] = \n{transformed_masses[unraveled_idx]}")
            print(f"targets[{idx}] =\n{flat_transformed_masses[idx]}")
        assert transformed_masses[unraveled_idx] == flat_transformed_masses[idx]
    unravel = lambda idx: unraveled_indices[idx]
    if verbose:
        print(barstr)
        print(f"mz:\n{mz}")
        print(f"masses:\n{mass_tensor}")
        print(f"left losses:\n{left_loss_tensor}\nright losses:\n{right_loss_tensor}")
        print(f"modifications:\n{modification_tensor}")
        print(f"target space:\n{flat_transformed_masses}")
        print(barstr)
        wait()
    for i in range(n - 1):
        deduplicated_i = charge_table[0,i].astype(int)
        for j in range(i + 1, n):
            deduplicated_j = charge_table[0,j].astype(int)

            query = mz[j] - mz[i]
            match_lo, match_hi = _bisect_range(flat_transformed_masses, query, tolerance)
            if verbose:
                print(f"gap[{i},{j}]")
                print(f"query:\n{query}")
                print(f"match range:\n{match_lo,match_hi}")
                print(f"targets matched:\n{[[flat_transformed_masses[idx] for idx in range(match_lo, match_hi + 1)]]}")
            min_delta = np.inf
            for match_idx in range(match_lo, match_hi + 1):
                optimizer = target_optimizer, left_optimizer, right_optimizer, modification_optimizer = unravel(match_idx)

                # optimal target
                target_mass = mass_tensor[target_optimizer, 0, 0, 0]
                target_residue = alphabet[target_optimizer]

                # optimal left loss
                query_left_loss = left_loss_tensor[0, left_optimizer, 0, 0]
                
                # optimal right loss
                query_right_loss = right_loss_tensor[0, 0, right_optimizer, 0]
                
                # optimal modification
                query_modification = modification_tensor[0, 0, 0, modification_optimizer]

                # calculate query mass
                left_transformed_peak = mz[i] + query_left_loss
                right_transformed_peak = mz[j] + query_right_loss + query_modification
                query_mass = right_transformed_peak - left_transformed_peak

                # calculate deltas
                delta_mass = query_mass - target_mass
                local_min_delta = query + extremal_delta
                if min_delta > local_min_delta:
                    min_delta = local_min_delta
                if verbose:                
                    print(f"optimizer {optimizer}")
                    print(f"left peak:\n{mz[i]}")
                    print(f"right peak:\n{mz[j]}")
                    print(f"left loss:\n{query_left_loss}")
                    print(f"right loss:\n{query_right_loss}")
                    print(f"modification loss:\n{query_modification}")
                    print(f"mass:\n{query_mass}")
                    print(f"target mass:\n{target_mass} ({target_residue})")
                    print(f"optimum[{i}, {j}] (topological: {deduplicated_i,deduplicated_j}):\n\tresults{optimizer} = {delta_mass}")
                    print(barstr)
                if abs(delta_mass) <= tolerance:
                    charge_state = tuple(charge_table[1,(i,j)])
                    target_idx = target_optimizer
                    left_loss_idx = left_optimizer - 1
                    right_loss_idx = right_optimizer - 1
                    modification_idx = modification_optimizer - 1
                    gap_match = GapMatch(
                        (-1,-1),
                        (deduplicated_i,deduplicated_j),
                        (i, j), 
                        charge_state,
                        target_residue, 
                        target_mass, 
                        target_idx,
                        query_mass, 
                        query_left_loss, 
                        left_loss_idx, 
                        query_right_loss, 
                        right_loss_idx, 
                        query_modification,
                        modification_idx,
                    )
                    if verbose:
                        print(f"\tmatch!")
                        print(gap_match)
                        print(barstr)
                        wait()
                    yield gap_match
            if min_delta > 0:
                if verbose:
                    print(f"\tno match; min_delta = {min_delta} > 0; terminating inner loop.")
                    print(barstr)
                    wait()
                break
            elif verbose:
                print(f"\tno match; min_delta = {min_delta} < 0; continuing inner loop.")
                print(barstr)
                wait()
    
def find_gaps_old(*args, **kwargs):
    # determine search mode.
    if "mode" not in kwargs:
        # kwarg "mode" not passed; defaults to tensor
        mode = "tensor"
    else:
        # kwarg "mode" was passed, get its value.
        mode = kwargs["mode"]
        # subroutines will crash if "mode" stays in the kwargs, so we delete it.
        del kwargs["mode"]
    # branch to the appropriate subroutine.
    if mode == "tensor":
        return _find_gaps_tensor(*args, **kwargs)
    elif mode == "bisect":
        return _find_gaps_bisect(*args, **kwargs)
    elif mode == "hybrid":
        raise NotImplementedError()
        return _find_gaps_hybrid(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported search mode {mode}")