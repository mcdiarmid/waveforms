import numpy as np
from numpy.typing import NDArray
from scipy.signal import besselap, impulse

from waveforms.cpm.helpers import normalize_cpm_filter


def freq_pulse_pcmfm(sps: int = 8, order: int = 4) -> NDArray[np.float64]:
    length = 3
    g1 = np.ones(sps) / (2*sps)
    _t, g2 = impulse(
        besselap(order, norm="mag"),
        T=np.linspace(0, length*2/0.7, num=(length-1)*sps+1)
    )
    g = np.convolve(g1, g2, mode="full")
    return normalize_cpm_filter(sps, g)
