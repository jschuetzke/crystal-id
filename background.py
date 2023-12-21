import numpy as np


def decay(shape, seed=None):
    """
    Parameters
    ----------
    shape : list
        shape of the background array to generate. E.g. [100,2500] for
        100 scans with 2500 datapoints each.
    seed : int, optional
        input random seed for debugging purposes. The default is None.

    Returns
    -------
    bkg : 2D numpy array
        returns varied background signals with exponential decay.

    """
    assert len(shape) == 2
    n_scans, datapoints = shape

    rng = np.random.default_rng(seed)

    bkg = np.arange(datapoints, dtype=np.float64)[None, :]
    bkg = np.repeat(bkg, n_scans, axis=0)

    slope = rng.uniform(-0.001, -0.0003, size=(n_scans, 1))

    bkg *= slope
    bkg = np.exp(bkg)

    return bkg


def chebyshev(shape, coefs=0.1, factor=0.3, seed=None):
    """
    Parameters
    ----------
    shape : list
        shape of the background array to generate. E.g. [100,2500] for
        100 scans with 2500 datapoints each.
    coefs : float
        coefficient value (negative, positive) for the Chebyshev polynomial.
    seed : int, optional
        input random seed for debugging purposes. The default is None.

    Returns
    -------
    bkg : 2D numpy array
        returns varied background signals with exponential decay.

    """
    # background function using a chebyshev polynomial
    coef_min = -coefs
    coef_max = coefs

    rng = np.random.default_rng(seed)

    cheb = np.zeros(shape)
    for i in range(shape[0]):
        polynom_order = rng.integers(2, 5)
        ccoefs = rng.uniform(
            coef_min / polynom_order, coef_max / polynom_order, (polynom_order)
        )
        c = np.polynomial.chebyshev.Chebyshev(ccoefs)
        cheb[i, :] = c.linspace(shape[1])[1]
    # correction to avoid negative background
    negative = np.min(cheb, axis=1) < 0
    cheb = cheb + (np.abs(np.min(cheb, axis=1)) * negative)[:, None]
    # scale according to scans
    cheb /= np.max(cheb, axis=1, keepdims=True)
    cheb *= factor
    return cheb