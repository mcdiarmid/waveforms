import numpy as np
from numpy.typing import NDArray
from scipy.signal import windows, butter, impulse


def hann_window(sps: int, f_cutoff: float) -> NDArray[np.float64]:
    return windows.hann(int(sps*f_cutoff))


def butterworth_lpf(sps: int, f_cutoff: float) -> NDArray[np.float64]:
    _t, g = impulse(
        butter(4, f_cutoff, analog=False, fs=sps),
        T=np.linspace(0, f_cutoff*2, num=100)
    )
    return g
