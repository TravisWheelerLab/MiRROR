from .gap_types import np, GapResult, GapSearchParameters

def read_gap_params(handle) -> GapSearchParameters:
    mode, residues, masses, losses, modifications, charges, tolerance = [line.split(',') for line in handle.readlines()]
    _as_floats = lambda x: list(map(float, x))
    return GapSearchParameters(
        mode[0],
        residues,
        _as_floats(masses),
        _as_floats(losses),
        _as_floats(modifications),
        _as_floats(charges),
        float(tolerance[0]),
    )

def write_gap_params(handle, gap_params: GapSearchParameters) -> None:
    _as_str = lambda x: ''.join(map(str, x))
    param_str = '\n'.join([
        gap_params.mode,
        _as_str(gap_params.residues),
        _as_str(gap_params.masses),
        _as_str(gap_params.losses),
        _as_str(gap_params.modifications),
        _as_str(gap_params.charges),
        str(gap_params.tolerance),
    ])
    handle.write(param_str)

def read_gap_result(handle) -> GapResult:
    return GapResult(
        None,
        gap_data = np.load(handle)
    )

def write_gap_result(handle, result: GapResult) -> None:
    np.save(
        handle,
        result._gap_data,
    )
