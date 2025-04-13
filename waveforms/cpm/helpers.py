import numpy as np
from numpy.typing import NDArray


def normalize_cpm_filter(
    sps: int,
    g: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Normalizes a filter such that the integral is 0.5.

    Args:
        sps (int): Samples per symbol
        g (NDArray[np.float64]): Phase pulse

    Returns:
        NDArray[np.float64]: Original filter g multiplied by a nomalizing factyor
    """
    a_scalar = sps / (np.sum(g) * 2)
    return a_scalar * g
