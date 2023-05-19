import numpy as np
from numpy.typing import NDArray
from scipy.signal import hanning


def hann_window(sps: int, f_cutoff: float) -> NDArray[np.float64]:
    return hanning(int(sps*f_cutoff))
