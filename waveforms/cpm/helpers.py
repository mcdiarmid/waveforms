import numpy as np
from numpy.typing import NDArray


def normalize_cpm_filter(
    sps: int,
    g: NDArray[np.float64],
) -> NDArray[np.float64]:
    a_scalar = sps / (np.cumsum(g)[-1] * 2)
    return a_scalar * g
