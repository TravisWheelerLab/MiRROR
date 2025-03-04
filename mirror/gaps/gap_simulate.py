from .gap_types import np, MASSES, RESIDUES, LOSSES, MODIFICATIONS, CHARGES

def create_mass_sequence(masses, alphabet, seq_len):
    n = len(masses)
    assert n == len(alphabet)
    choices = np.random.choice(np.arange(n), seq_len)
    return [0, *np.array(masses)[choices]], np.array(alphabet)[choices]

def _parametize_subsequence_transformation(
    sequence: np.ndarray, 
    transformation_values: np.ndarray, 
    num_transformations: np.ndarray, 
    prefix_offset = 1,
    multiplicative = False,
):
    # todo: each transformation depends on the value of its position
    idx = np.random.choice(np.arange(len(transformation_values)), num_transformations, replace = True)
    pos = np.random.choice(np.arange(len(sequence))[prefix_offset:], num_transformations, replace = False)
    if multiplicative:
        transformations = np.ones_like(sequence)
        transformations[pos] *= transformation_values[idx]
    else:
        transformations = np.zeros_like(sequence)
        transformations[pos] += transformation_values[idx]
    return transformations

def create_losses(mass_sequence, loss_values, num_losses):
    # todo: each loss depends on the residue at its position
    return _parametize_subsequence_transformation(
        mass_sequence, 
        loss_values, 
        num_losses,
    )

def create_modifications(mass_sequence, mod_values, num_mods):
    # todo: each modification depends on the residue at its position
    return _parametize_subsequence_transformation(
        mass_sequence, 
        mod_values, 
        num_mods,
    )

def create_charges(mass_sequence, charge_values, num_charged):
    # todo: each charge depends on the residue at its position
    return _parametize_subsequence_transformation(
        mass_sequence, 
        charge_values, 
        num_charged,
        prefix_offset=0,
        multiplicative=True,
    )

def create_noise(mass_sequence, num_noise):
    return np.random.uniform(
        low = 0, 
        high = int(1.2 * max(mass_sequence)),
        size = num_noise,
    )

def create_mz(mass_sequence, loss_arr, mod_arr, charge_arr, noise_arr):
    n_masses = len(mass_sequence)
    # apply PTMs
    modified_mass_sequence = mass_sequence + mod_arr
    # create fragment peaks
    mz = np.add.accumulate(modified_mass_sequence)
    # apply losses
    mz += loss_arr
    # apply charges
    assert (charge_arr >= 1).all()
    mz /= charge_arr
    # combine with noise
    mz = np.hstack([mz, noise_arr])
    # sort and keep track of true peak index set
    indices = np.arange(mz.size)
    order = np.argsort(mz)
    mz = mz[order]
    indices = indices[order]
    true_series = []
    for i in range(mz.size):
        if indices[i] < n_masses:
            true_series.append(i)
    true_series.sort(key = lambda i: indices[i])
    true_gaps = [(true_series[i], true_series[i + 1]) for i in range(len(true_series) - 1)]
    position_lookup = dict()
    for (i,k) in enumerate(true_series):
        position_lookup[k] = i
    return mz, true_gaps, position_lookup

def random_data(seq_len, num_losses, num_mods, num_charged, num_noise, masses, residues, losses, modifications, charges):
    mass_seq, residues = create_mass_sequence(masseses, residu, seq_len)
    losses = create_losses(mass_seq, losses, num_losses)
    modifications = create_modifications(mass_seq, modifications, num_mods)
    charges = create_charges(mass_seq, charges, num_charged)
    noise = create_noise(mass_seq, num_noise)
    mz, true_gaps, position_lookup = create_mz(mass_seq, losses, modifications, charges, noise)
    return mass_seq, residues, losses, modifications, charges, noise, mz, true_gaps, position_lookup

def display_true_data(mass_sequence, residues, losses, modifications, charges, mz, true_gaps):
    pass