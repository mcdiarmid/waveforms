import numpy as np
from numpy.typing import NDArray
from scipy.signal import kaiserord, firwin


def kaiser_fir_lpf(
    sps: int,
    f_cutoff: float,
    width: float = None,
    ripple_db: float = 80.0,
) -> NDArray[np.float64]:
    """
    Modified from snippet
    https://scipy.github.io/old-wiki/pages/Cookbook/FIRFilter.html

    :param sps: Samples per symbol
    :param f_cutoff: Cutoff frequency
    :param width: Transition width
    :param ripple_db: Supression after transition
    :return: Filter taps
    """
    nyq_rate = sps / 2
    N, beta = kaiserord(ripple_db, width or 1/sps)
    return firwin(
        numtaps=N, 
        cutoff=f_cutoff/nyq_rate,
        window=('kaiser', beta)
    )
