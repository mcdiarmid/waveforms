import numpy as np
from numpy.typing import NDArray
from scipy.signal import windows, butter, impulse, kaiserord, firwin


def hann_window(sps: int, f_cutoff: float) -> NDArray[np.float64]:
    return windows.hann(int(sps*f_cutoff))


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
    """
    # The Nyquist rate of the signal.
    nyq_rate = sps / 2
    if width is None:
        width = 1 / sps
    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = kaiserord(ripple_db, width)
    # The cutoff frequency of the filter.
    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    return firwin(N, f_cutoff/nyq_rate, window=('kaiser', beta))


def butterworth_lpf(sps: int, f_cutoff: float) -> NDArray[np.float64]:
    _t, g = impulse(
        butter(4, 0.5, analog=False),
        T=np.linspace(0, f_cutoff*2, num=sps*2)
    )
    return g
