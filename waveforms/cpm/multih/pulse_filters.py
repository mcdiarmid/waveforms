import numpy as np
from numpy.typing import NDArray

from waveforms.cpm.helpers import normalize_cpm_filter


MULTIH_IRIG_NUMER = np.array([4, 5])
MULTIH_IRIG_DENOM = 16


def freq_pulse_multih_irig(sps: int = 8, length: float = 3) -> NDArray[np.float64]:
    """Generates a RC filter frequency pulse for the Multi-h CPM waveform.

    Args:
        sps (int): Samples per symbol time
        length (float): Pulse length (symbol time)

    Returns:
        NDArray[np.float64]: Multi-h CPM frequency pulse.
    """
    t_norm = np.linspace(0, length, num=length * sps + 1)
    g = (1 - np.cos(2 * np.pi * t_norm / length)) / (2 * length)
    return normalize_cpm_filter(sps, g)
