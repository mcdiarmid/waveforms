import numpy as np
from numpy.typing import NDArray

from waveforms.cpm.pcmfm.bessel import bessel
from waveforms.cpm.helpers import normalize_cpm_filter


def freq_pulse_pcmfm(sps: int = 8, order: int = 4) -> NDArray[np.float64]:
    g1 = np.ones(sps) / (sps)
    g2 = bessel(sps, order=order)
    # g2 = np.ones(1)
    g = np.convolve(g1, g2, mode="full")
    return normalize_cpm_filter(sps, g)
