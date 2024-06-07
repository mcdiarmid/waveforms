from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import firwin, kaiserord

if TYPE_CHECKING:
    from numpy.typing import NDArray


def kaiser_fir_lpf(
    sps: int,
    f_cutoff: float,
    width: float | None = None,
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
    numtaps, beta = kaiserord(ripple_db, width or 1 / sps)
    return firwin(numtaps=numtaps, cutoff=f_cutoff / nyq_rate, window=("kaiser", beta))
