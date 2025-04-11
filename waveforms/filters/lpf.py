from __future__ import annotations

from typing import TYPE_CHECKING

from scipy.signal import firwin, kaiserord


if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


def kaiser_fir_lpf(
    sps: int,
    f_cutoff: float,
    width: float | None = None,
    ripple_db: float = 80.0,
) -> NDArray[np.float64]:
    """Generates a Kaiser FIR Low Pass Filter.

    Modified from: https://scipy.github.io/old-wiki/pages/Cookbook/FIRFilter.html

    Args:
        sps (int): Samples per symbol
        f_cutoff (float): Cutoff frequency
        width (float): Transition width
        ripple_db (float): Supression after transition

    Returns:
        NDArray[np.float64]: Filter taps
    """
    nyq_rate = sps / 2
    numtaps, beta = kaiserord(ripple_db, width or 1 / sps)
    return firwin(numtaps=numtaps, cutoff=f_cutoff / nyq_rate, window=("kaiser", beta))
